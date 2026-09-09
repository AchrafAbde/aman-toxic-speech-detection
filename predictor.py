import os
import json
import joblib
import numpy as np
from utils import clean_text, LABEL_COLS

# torch et transformers sont optionnels — uniquement requis pour le Teacher XLM-R
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── HuggingFace Hub — fallback si fichiers absents en local ─────────────────
HF_REPO_ID = 'BHM03/amane-models'

def _resolve(local_path: str, hf_filename: str) -> str:
    """Retourne le chemin local si le fichier existe, sinon télécharge depuis HF Hub."""
    if os.path.exists(local_path):
        return local_path
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            token=os.environ.get('HF_TOKEN'),
        )
    except Exception as e:
        raise FileNotFoundError(
            f'Fichier introuvable localement ({local_path}) '
            f'et téléchargement HF Hub échoué : {e}'
        )


# ── Chemins modèles ──────────────────────────────────────────────────────────
BENCHMARK_MODEL_PATH    = 'models/ovr_balanced.joblib'
BENCHMARK_TFIDF_PATH    = 'models/tfidf_vectorizer.joblib'
TEACHER_CKPT_PATH       = 'models/teacher/teacher_best.pt'
TEACHER_TOKENIZER_PATH  = 'models/teacher/tokenizer'
TEACHER_THRESH_PATH     = 'models/teacher/thresholds.json'
TEACHER_STAR_CKPT_PATH  = 'models/teacher_star/teacher_star.pt'
TEACHER_STAR_THRESH_PATH= 'models/teacher_star/threshold_darija.json'
STUDENT_CKPT_PATH       = 'models/student/student_best.pt'
STUDENT_THRESH_PATH     = 'models/student/threshold_student.json'

# ── Hyperparamètres d'inférence Teacher (cohérents avec l'entraînement) ─────
TEACHER_MAX_LEN      = 256
TEACHER_STAR_MAX_LEN = 128
STUDENT_MAX_LEN      = 128
TEACHER_MODEL_HF     = 'xlm-roberta-base'
STUDENT_MODEL_HF     = 'distilbert-base-multilingual-cased'


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARK — OneVsRest + LogisticRegression
# ════════════════════════════════════════════════════════════════════════════
class BenchmarkPredictor:
    def __init__(self):
        self.model    = None
        self.tfidf    = None
        self.loaded   = False
        self.error    = None

    def load(self):
        try:
            self.model  = joblib.load(_resolve(BENCHMARK_MODEL_PATH, 'ovr_balanced.joblib'))
            self.tfidf  = joblib.load(_resolve(BENCHMARK_TFIDF_PATH, 'tfidf_vectorizer.joblib'))
            self.loaded = True
        except Exception as e:
            self.error  = str(e)
            self.loaded = False

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError(f'Benchmark non chargé : {self.error}')
        cleaned  = clean_text(text).lower()
        vec      = self.tfidf.transform([cleaned])
        probs    = self.model.predict_proba(vec)[0]
        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    def predict_batch(self, texts: list) -> list[dict]:
        if not self.loaded:
            raise RuntimeError(f'Benchmark non chargé : {self.error}')
        cleaned = [clean_text(t).lower() for t in texts]
        vecs    = self.tfidf.transform(cleaned)
        probs   = self.model.predict_proba(vecs)
        return [
            {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, row)}
            for row in probs
        ]


# ════════════════════════════════════════════════════════════════════════════
# XLM-R TEACHER
# ════════════════════════════════════════════════════════════════════════════
class TeacherPredictor:
    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self.thresholds = None
        self.loaded     = False
        self.error      = None
        self.device     = None

    def load(self):
        if not TORCH_AVAILABLE:
            self.error  = (
                'PyTorch / Transformers non installés. '
                'Lance : pip install torch transformers sentencepiece'
            )
            self.loaded = False
            return

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 2. Charger le tokenizer — on utilise directement 'xlm-roberta-base'
            # car le tokenizer ne change pas lors du fine-tuning, et les fichiers
            # locaux sont incomplets (sentencepiece.bpe.model manquant).
            self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

            # 3. Reconstruire l'architecture (identique au notebook d'entraînement)
            class TeacherModel(nn.Module):
                def __init__(self, model_name: str, num_labels: int = 6, dropout: float = 0.1):
                    super().__init__()
                    self.encoder    = AutoModel.from_pretrained(model_name)
                    hidden_size     = self.encoder.config.hidden_size
                    self.dropout    = nn.Dropout(dropout)
                    self.classifier = nn.Linear(hidden_size, num_labels)

                def mean_pool(self, token_embeds, attention_mask):
                    mask   = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
                    summed = (token_embeds * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    return summed / counts

                def forward(self, input_ids, attention_mask):
                    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    pooled  = self.mean_pool(outputs.last_hidden_state, attention_mask)
                    pooled  = self.dropout(pooled)
                    return self.classifier(pooled)

            # 4. Charger le checkpoint
            ckpt  = torch.load(_resolve(TEACHER_CKPT_PATH, 'teacher_best.pt'), map_location=self.device, weights_only=False)
            model = TeacherModel(TEACHER_MODEL_HF, num_labels=len(LABEL_COLS))
            model.load_state_dict(ckpt['model_state'])
            model.to(self.device)
            model.eval()
            self.model = model

            # 5. Charger les seuils tunés
            if os.path.exists(TEACHER_THRESH_PATH):
                with open(TEACHER_THRESH_PATH) as f:
                    self.thresholds = json.load(f)
            else:
                # Fallback : seuil 0.5 par défaut
                self.thresholds = {col: 0.5 for col in LABEL_COLS}

            self.loaded = True

        except Exception as e:
            self.error  = f'{type(e).__name__}: {e}'
            self.loaded = False

    # ── Inférence : probabilités ───────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """Retourne les probabilités sigmoïdes par label."""
        if not self.loaded:
            raise RuntimeError(f'Teacher non chargé : {self.error}')

        cleaned  = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            max_length=TEACHER_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attn_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    # ── Inférence : labels binaires (avec seuils tunés) ────────────────────
    def predict_with_labels(self, text: str) -> dict:
        """
        Retourne probabilités + labels binaires selon les seuils tunés.
        Format : {'probs': {...}, 'labels': {...}, 'thresholds': {...}}
        """
        probs  = self.predict(text)
        labels = {col: int(probs[col] >= self.thresholds[col]) for col in LABEL_COLS}
        return {
            'probs'      : probs,
            'labels'     : labels,
            'thresholds' : self.thresholds,
        }

    # ── Inférence : batch ──────────────────────────────────────────────────
    def predict_batch(self, texts: list) -> list[dict]:
        """Inférence batch — plus efficace que predict() en boucle."""
        if not self.loaded:
            raise RuntimeError(f'Teacher non chargé : {self.error}')

        cleaned = [clean_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned,
            max_length=TEACHER_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attn_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()

        return [
            {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, row)}
            for row in probs
        ]


# ════════════════════════════════════════════════════════════════════════════
# XLM-R TEACHER* — adapté Darija (fine-tuné sur DarLoad, loss masquée)
# ════════════════════════════════════════════════════════════════════════════
class TeacherStarPredictor:
    def __init__(self):
        self.model          = None
        self.tokenizer      = None
        self.threshold      = 0.45   # seuil tunée sur DarLoad
        self.loaded         = False
        self.error          = None
        self.device         = None

    def load(self):
        if not TORCH_AVAILABLE:
            self.error  = 'PyTorch / Transformers non installés.'
            self.loaded = False
            return

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

            class TeacherModel(nn.Module):
                def __init__(self, model_name, num_labels=6, dropout=0.1):
                    super().__init__()
                    self.encoder    = AutoModel.from_pretrained(model_name)
                    self.dropout    = nn.Dropout(dropout)
                    self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

                def mean_pool(self, token_embeds, attention_mask):
                    mask   = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
                    summed = (token_embeds * mask).sum(dim=1)
                    return summed / mask.sum(dim=1).clamp(min=1e-9)

                def forward(self, input_ids, attention_mask):
                    out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    pooled = self.mean_pool(out.last_hidden_state, attention_mask)
                    return self.classifier(self.dropout(pooled))

            ckpt  = torch.load(_resolve(TEACHER_STAR_CKPT_PATH, 'teacher_star.pt'), map_location=self.device, weights_only=False)
            model = TeacherModel(TEACHER_MODEL_HF, num_labels=len(LABEL_COLS))
            model.load_state_dict(ckpt['model_state'])
            model.to(self.device)
            model.eval()
            self.model = model

            if os.path.exists(TEACHER_STAR_THRESH_PATH):
                with open(TEACHER_STAR_THRESH_PATH) as f:
                    self.threshold = json.load(f).get('toxic_darija', 0.45)

            self.loaded = True

        except Exception as e:
            self.error  = f'{type(e).__name__}: {e}'
            self.loaded = False

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError(f'Teacher* non chargé : {self.error}')
        cleaned  = clean_text(text)
        encoding = self.tokenizer(
            cleaned, max_length=TEACHER_STAR_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    def predict_batch(self, texts: list) -> list[dict]:
        if not self.loaded:
            raise RuntimeError(f'Teacher* non chargé : {self.error}')
        cleaned  = [clean_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned, max_length=TEACHER_STAR_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return [{col: float(round(p, 4)) for col, p in zip(LABEL_COLS, row)} for row in probs]


# ════════════════════════════════════════════════════════════════════════════
# FACTORY — chargement unique via st.cache_resource
# ════════════════════════════════════════════════════════════════════════════
def load_benchmark() -> BenchmarkPredictor:
    p = BenchmarkPredictor()
    p.load()
    return p

def load_teacher() -> TeacherPredictor:
    p = TeacherPredictor()
    p.load()
    return p

def load_teacher_star() -> TeacherStarPredictor:
    p = TeacherStarPredictor()
    p.load()
    return p


# ════════════════════════════════════════════════════════════════════════════
# STUDENT — DistilBERT multilingue spécialisé Darija
# ════════════════════════════════════════════════════════════════════════════
class StudentPredictor:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.threshold = 0.6
        self.loaded    = False
        self.error     = None
        self.device    = None

    def load(self):
        if not TORCH_AVAILABLE:
            self.error  = 'PyTorch / Transformers non installés.'
            self.loaded = False
            return

        try:
            self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_HF)

            class StudentModel(nn.Module):
                def __init__(self, model_name, num_labels=6, dropout=0.1):
                    super().__init__()
                    self.encoder    = AutoModel.from_pretrained(model_name)
                    self.dropout    = nn.Dropout(dropout)
                    self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

                def forward(self, input_ids, attention_mask):
                    out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    pooled = out.last_hidden_state[:, 0, :]  # token [CLS]
                    return self.classifier(self.dropout(pooled))

            ckpt  = torch.load(_resolve(STUDENT_CKPT_PATH, 'student_best.pt'), map_location=self.device, weights_only=False)
            model = StudentModel(STUDENT_MODEL_HF, num_labels=len(LABEL_COLS))
            model.load_state_dict(ckpt['model_state'])
            model.to(self.device)
            model.eval()
            self.model = model

            if os.path.exists(STUDENT_THRESH_PATH):
                with open(STUDENT_THRESH_PATH) as f:
                    self.threshold = json.load(f).get('toxic_darija', 0.35)

            self.loaded = True

        except Exception as e:
            self.error  = f'{type(e).__name__}: {e}'
            self.loaded = False

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError(f'Student non chargé : {self.error}')
        cleaned  = clean_text(text)
        encoding = self.tokenizer(
            cleaned, max_length=STUDENT_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    def predict_batch(self, texts: list) -> list[dict]:
        if not self.loaded:
            raise RuntimeError(f'Student non chargé : {self.error}')
        cleaned  = [clean_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned, max_length=STUDENT_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return [{col: float(round(p, 4)) for col, p in zip(LABEL_COLS, row)} for row in probs]


def load_student() -> StudentPredictor:
    p = StudentPredictor()
    p.load()
    return p
