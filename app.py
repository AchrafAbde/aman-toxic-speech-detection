import json
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import LABEL_COLS, LABEL_META, toxicity_level

API_URL = st.secrets.get("API_URL", "https://bhm03-amane-api.hf.space")


class APIPredictor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded     = False
        self.error      = None

    def predict(self, text: str) -> dict:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model": self.model_name},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, texts: list) -> list:
        resp = requests.post(
            f"{API_URL}/predict_batch",
            json={"texts": texts, "model": self.model_name},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def get_api_status() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AMANE — Détection de toxicité',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stTextArea textarea { font-size: 14px; }
.label-fr { font-size: 13px; font-weight: 500; }
.model-badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 500;
}
.badge-available { background: #E1F5EE; color: #085041; }
.badge-pending   { background: #FAEEDA; color: #633806; }
div[data-testid="stProgress"] > div { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Metadata des modèles ──────────────────────────────────────────────────────
MODEL_INFO = {
    'Benchmark (OVR LogReg)': {
        'short': 'Benchmark',   'badge': 'OVR LogReg + TF-IDF',
        'color': '#888780',     'fill':  'rgba(136,135,128,0.12)',
        'lang':  'Anglais',     'params': '—',    'latency': '< 10ms',
        'f1': '—',              'phase': 'Phase 1',
    },
    'XLM-R Teacher (EN)': {
        'short': 'Teacher EN',  'badge': 'XLM-RoBERTa-base',
        'color': '#534AB7',     'fill':  'rgba(83,74,183,0.15)',
        'lang':  'Anglais',     'params': '278M', 'latency': '~240ms',
        'f1': '0.623',          'phase': 'Phase 1',
    },
    'XLM-R Teacher* (Darija)': {
        'short': 'Teacher*',    'badge': 'XLM-R adapté Darija',
        'color': '#0F6E56',     'fill':  'rgba(15,110,86,0.15)',
        'lang':  'Darija + AN', 'params': '278M', 'latency': '~240ms',
        'f1': '0.815',          'phase': 'Phase 2',
    },
    'Student DistilBERT (Darija)': {
        'short': 'Student',     'badge': 'DistilBERT distillé',
        'color': '#D85A30',     'fill':  'rgba(216,90,48,0.15)',
        'lang':  'Darija',      'params': '66M',  'latency': '~60ms',
        'f1': '0.895',          'phase': 'Phase 3',
    },
}

PRESETS = {
    'EN — Toxique': 'You are a complete idiot and I hope something terrible happens to you.',
    'DA — Toxique': 'nta hmar bzzaf, ghadi ndarb rassek u nkhlik f7ala',
    'DA — Neutre':  'labas 3lik khoya? kif nta? mzyan lhamdulah',
    'FR — Neutre':  'Bonjour, comment allez-vous? Bonne journée à tous.',
}

# ── Statut API + initialisation des predictors ───────────────────────────────
benchmark    = APIPredictor('benchmark')
teacher      = APIPredictor('teacher')
teacher_star = APIPredictor('teacher_star')
student      = APIPredictor('student')

_status = get_api_status()
_api_online = bool(_status)
for _predictor, _key in [
    (benchmark,    'benchmark'),
    (teacher,      'teacher'),
    (teacher_star, 'teacher_star'),
    (student,      'student'),
]:
    _info = _status.get(_key, {})
    _predictor.loaded = _info.get('loaded', False)
    _predictor.error  = _info.get('error')

MODELS = {
    'Benchmark (OVR LogReg)':      benchmark,
    'XLM-R Teacher (EN)':          teacher,
    'XLM-R Teacher* (Darija)':     teacher_star,
    'Student DistilBERT (Darija)': student,
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛡️ AMANE')
    st.markdown('*Détection de discours toxiques*')
    st.divider()

    page = st.radio(
        'Navigation',
        ['Analyse', 'Comparaison 4 modèles', 'Batch CSV', 'YouTube Analyse', 'Pipeline & Métriques'],
        label_visibility='collapsed',
    )

    st.divider()
    if not _api_online:
        st.warning('API en démarrage — patientez quelques secondes puis rechargez.', icon='⏳')
    st.markdown('**Statut des modèles**')
    for key, predictor in MODELS.items():
        info = MODEL_INFO[key]
        cls  = 'badge-available' if predictor.loaded else 'badge-pending'
        icon = '✓' if predictor.loaded else '⏳'
        st.markdown(
            f'<span class="model-badge {cls}">{icon} {info["short"]}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption('Projet AMANE — Master 4 IA & Data Science')


# ══════════════════════════════════════════════════════════════════════════════
# Helpers visuels
# ══════════════════════════════════════════════════════════════════════════════

def render_probs(probs: dict, threshold: float = 0.5):
    level, lcolor = toxicity_level(probs)
    st.markdown(
        f'<div style="margin-bottom:10px;">'
        f'<span style="font-size:12px;color:#666;">Niveau : </span>'
        f'<span style="font-size:13px;font-weight:600;color:{lcolor};">{level}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for col in LABEL_COLS:
        prob = probs[col]
        meta = LABEL_META[col]
        flag = '●' if prob > threshold else '○'
        c1, c2, c3 = st.columns([3, 6, 1])
        with c1:
            st.markdown(
                f'<span class="label-fr" style="color:{meta["color"] if prob > threshold else "inherit"};">'
                f'{flag} {meta["fr"]}</span>',
                unsafe_allow_html=True,
            )
        with c2:
            st.progress(float(prob))
        with c3:
            st.markdown(
                f'<span style="font-size:12px;font-weight:500;">{prob*100:.0f}%</span>',
                unsafe_allow_html=True,
            )


def radar_single(probs: dict, color: str, fill: str) -> go.Figure:
    labels = [LABEL_META[c]['fr'] for c in LABEL_COLS]
    v = [probs[c] for c in LABEL_COLS] + [probs[LABEL_COLS[0]]]
    l = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v, theta=l, fill='toself',
        line=dict(color=color, width=2),
        fillcolor=fill,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=280,
    )
    return fig


def multi_radar(results: dict) -> go.Figure:
    labels = [LABEL_META[c]['fr'] for c in LABEL_COLS]
    l      = labels + [labels[0]]
    fig    = go.Figure()
    for key, probs in results.items():
        info = MODEL_INFO[key]
        v    = [probs[c] for c in LABEL_COLS] + [probs[LABEL_COLS[0]]]
        fig.add_trace(go.Scatterpolar(
            r=v, theta=l, fill='toself',
            name=info['short'],
            line=dict(color=info['color'], width=2),
            fillcolor=info['fill'],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=True,
        legend=dict(x=1.05, y=1.0),
        margin=dict(l=40, r=110, t=30, b=30),
        height=400,
    )
    return fig


# ── YouTube helpers ───────────────────────────────────────────────────────────
_YT_ID_RE = re.compile(
    r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})'
)

def extract_video_id(url: str) -> str | None:
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else (url.strip() if len(url.strip()) == 11 else None)


def fetch_youtube_comments(api_key: str, video_id: str, max_comments: int) -> list[str]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    youtube  = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page = None

    while len(comments) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                videoId=video_id,
                part='snippet',
                maxResults=min(100, max_comments - len(comments)),
                textFormat='plainText',
                pageToken=next_page,
            ).execute()
        except HttpError as e:
            if e.resp.status == 403:
                raise RuntimeError('Commentaires désactivés pour cette vidéo.')
            raise RuntimeError(f'Erreur API YouTube : {e}')

        for item in resp.get('items', []):
            text = item['snippet']['topLevelComment']['snippet'].get('textDisplay', '')
            if text.strip():
                comments.append(text.strip())

        next_page = resp.get('nextPageToken')
        if not next_page:
            break

    return comments


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse commentaire unique
# ══════════════════════════════════════════════════════════════════════════════
if page == 'Analyse':
    st.title('Analyse de commentaire')
    st.caption('Entrez un commentaire pour détecter la toxicité avec le modèle sélectionné.')

    model_options  = [k for k, p in MODELS.items() if p.loaded]
    selected_model = st.selectbox('Modèle', model_options)
    info           = MODEL_INFO[selected_model]

    st.markdown(
        f'<span class="model-badge badge-available">{info["badge"]}</span>'
        f'<span style="font-size:12px;color:#666;margin-left:10px;">'
        f'Langue : <b>{info["lang"]}</b> &nbsp;·&nbsp; Params : <b>{info["params"]}</b>'
        f' &nbsp;·&nbsp; Latence : <b>{info["latency"]}</b>'
        f' &nbsp;·&nbsp; Macro F1 : <b>{info["f1"]}</b></span>',
        unsafe_allow_html=True,
    )
    st.write('')

    comment = st.text_area(
        'Commentaire à analyser',
        value='You are a complete idiot and I hope something terrible happens to you.',
        height=110,
        placeholder='Anglais, français ou Darija...',
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze = st.button('Analyser', type='primary', use_container_width=True)
    with col_clear:
        clear = st.button('Effacer')
    if clear:
        comment = ''

    if analyze and comment.strip():
        with st.spinner('Analyse en cours...'):
            try:
                probs = MODELS[selected_model].predict(comment)
                st.divider()
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.subheader('Résultats par label')
                    render_probs(probs)
                with col_right:
                    st.subheader('Radar')
                    st.plotly_chart(
                        radar_single(probs, info['color'], info['fill']),
                        use_container_width=True,
                    )
                with st.expander('Probabilités brutes (JSON)'):
                    st.json({col: f'{v*100:.2f}%' for col, v in probs.items()})
            except Exception as e:
                st.error(f'Erreur : {e}')
    elif analyze:
        st.warning('Veuillez entrer un commentaire.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Comparaison 4 modèles
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Comparaison 4 modèles':
    st.title('Comparaison des 4 modèles AMANE')
    st.caption(
        "Testez un commentaire sur tous les modèles simultanément et visualisez "
        "l'évolution progressive des performances."
    )

    # Preset buttons
    if 'cmp_text' not in st.session_state:
        st.session_state['cmp_text'] = PRESETS['EN — Toxique']

    st.markdown('**Exemples rapides :**')
    preset_cols = st.columns(len(PRESETS))
    for i, (label, text) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            if st.button(label, use_container_width=True, key=f'preset_{i}'):
                st.session_state['cmp_text'] = text
                st.rerun()

    comment = st.text_area('Commentaire à analyser', key='cmp_text', height=90)
    run     = st.button('Analyser avec les 4 modèles', type='primary')

    if run and comment.strip():
        with st.spinner('Analyse en cours...'):
            results = {}
            for key, predictor in MODELS.items():
                if predictor.loaded:
                    try:
                        results[key] = predictor.predict(comment)
                    except Exception as e:
                        st.warning(f'{MODEL_INFO[key]["short"]} — erreur : {e}')

        if not results:
            st.error('Aucun modèle disponible.')
        else:
            st.divider()

            # ── Metric cards ─────────────────────────────────────────────
            metric_cols = st.columns(len(results))
            for i, (key, probs) in enumerate(results.items()):
                info          = MODEL_INFO[key]
                level, lcolor = toxicity_level(probs)
                top_prob      = max(probs.values())
                top_label     = LABEL_META[max(probs, key=probs.get)]['fr']
                with metric_cols[i]:
                    st.markdown(
                        f'<div style="border-left:4px solid {info["color"]};padding-left:10px;">'
                        f'<div style="font-weight:600;font-size:14px;">{info["short"]}</div>'
                        f'<div style="font-size:11px;color:#666;">'
                        f'{info["phase"]} &nbsp;·&nbsp; {info["params"]} params</div>'
                        f'<div style="font-size:20px;font-weight:700;color:{lcolor};margin-top:6px;">'
                        f'{level}</div>'
                        f'<div style="font-size:12px;color:#666;">'
                        f'Max : {top_prob*100:.0f}% ({top_label})</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.write('')

            # ── Prob bars — grille 2×2 ────────────────────────────────────
            keys = list(results.keys())
            for row_keys in [keys[:2], keys[2:]]:
                if not row_keys:
                    continue
                cols = st.columns(len(row_keys))
                for col_widget, key in zip(cols, row_keys):
                    info = MODEL_INFO[key]
                    with col_widget:
                        st.markdown(
                            f'<div style="font-size:13px;font-weight:600;color:{info["color"]};'
                            f'border-bottom:2px solid {info["color"]};padding-bottom:4px;'
                            f'margin-bottom:10px;">'
                            f'{info["short"]} — {info["badge"]}</div>',
                            unsafe_allow_html=True,
                        )
                        render_probs(results[key])
                st.write('')

            # ── Radar multi-modèles ───────────────────────────────────────
            st.divider()
            st.subheader('Radar comparatif')
            st.plotly_chart(multi_radar(results), use_container_width=True)

            # ── Tableau delta vs Benchmark ────────────────────────────────
            bench_key = 'Benchmark (OVR LogReg)'
            if bench_key in results:
                st.subheader('Delta vs Benchmark')
                bench      = results[bench_key]
                other_keys = [k for k in results if k != bench_key]
                rows = []
                for col in LABEL_COLS:
                    row = {'Label': LABEL_META[col]['fr'], 'Benchmark': f"{bench[col]*100:.1f}%"}
                    for key in other_keys:
                        d     = results[key][col] - bench[col]
                        arrow = '↑' if d > 0.03 else ('↓' if d < -0.03 else '≈')
                        row[MODEL_INFO[key]['short']] = (
                            f"{results[key][col]*100:.1f}% ({d*100:+.1f}% {arrow})"
                        )
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif run:
        st.warning('Veuillez entrer un commentaire.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Batch CSV':
    st.title('Analyse en lot — CSV')
    st.caption(
        'Uploadez un fichier CSV avec une colonne `comment_text`. '
        'Le système analysera chaque ligne et renverra un CSV annoté.'
    )

    with st.expander('Format du CSV attendu'):
        st.markdown('Le fichier doit contenir **au minimum** une colonne `comment_text` :')
        st.dataframe(pd.DataFrame({
            'comment_text': [
                'You are an idiot',
                'Have a great day!',
                'nta hmar bzzaf',
            ]
        }), use_container_width=True)

    batch_options = [k for k, p in MODELS.items() if p.loaded]
    batch_model   = st.selectbox('Modèle pour le batch', batch_options, key='batch_model')

    uploaded = st.file_uploader('Choisir un fichier CSV', type=['csv'])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        if 'comment_text' not in df_input.columns:
            st.error('La colonne `comment_text` est manquante dans le fichier.')
        else:
            st.success(f'Fichier chargé — {len(df_input):,} commentaires')
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button("Lancer l'analyse", type='primary'):
                progress_bar = st.progress(0, text='Analyse en cours...')
                texts = df_input['comment_text'].fillna('').tolist()
                n     = len(texts)

                try:
                    all_probs = MODELS[batch_model].predict_batch(texts)
                    progress_bar.progress(80, text='Assemblage des résultats...')

                    df_result = df_input.copy()
                    for col in LABEL_COLS:
                        df_result[f'prob_{col}'] = [p[col] for p in all_probs]
                    df_result['toxicity_level'] = [
                        toxicity_level(p)[0] for p in all_probs
                    ]
                    progress_bar.progress(100, text='Terminé ✓')

                    st.divider()
                    st.subheader('Résultats')

                    m1, m2, m3, m4 = st.columns(4)
                    n_toxic  = sum(1 for p in all_probs if p['toxic'] > 0.5)
                    n_threat = sum(1 for p in all_probs if p['threat'] > 0.5)
                    n_hate   = sum(1 for p in all_probs if p['identity_hate'] > 0.5)
                    m1.metric('Total analysés',    f'{n:,}')
                    m2.metric('Toxiques détectés', f'{n_toxic:,}',  f'{n_toxic/n*100:.1f}%')
                    m3.metric('Menaces détectées', f'{n_threat:,}')
                    m4.metric('Haine identitaire', f'{n_hate:,}')

                    prob_cols  = [f'prob_{c}' for c in LABEL_COLS]
                    display_df = df_result[['comment_text', 'toxicity_level'] + prob_cols].copy()
                    for c in prob_cols:
                        display_df[c] = display_df[c].apply(lambda x: f'{x*100:.1f}%')
                    st.dataframe(display_df, use_container_width=True, height=350)

                    level_counts = df_result['toxicity_level'].value_counts()
                    fig_bar = go.Figure(go.Bar(
                        x=level_counts.index,
                        y=level_counts.values,
                        marker_color=['#A32D2D', '#D85A30', '#EF9F27', '#854F0B', '#0F6E56'],
                    ))
                    fig_bar.update_layout(
                        title='Distribution des niveaux de toxicité',
                        xaxis_title='Niveau', yaxis_title='Commentaires',
                        height=280, margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    csv_bytes = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        '⬇️ Télécharger les résultats (CSV)',
                        data=csv_bytes,
                        file_name='amane_resultats.csv',
                        mime='text/csv',
                        type='primary',
                    )
                except Exception as e:
                    st.error(f"Erreur durant l'analyse : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — YouTube Analyse
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'YouTube Analyse':
    st.title('Analyse YouTube')
    st.caption(
        'Collez l\'URL d\'une vidéo YouTube pour extraire ses commentaires '
        'et analyser leur toxicité avec un modèle AMANE.'
    )

    col_url, col_key = st.columns([3, 2])
    with col_url:
        yt_url = st.text_input(
            'URL ou ID de la vidéo',
            placeholder='https://www.youtube.com/watch?v=...',
        )
    with col_key:
        yt_key = st.text_input(
            'Clé API YouTube Data v3',
            type='password',
            placeholder='AIza...',
        )

    col_max, col_model, col_btn = st.columns([2, 3, 1])
    with col_max:
        max_comments = st.slider('Max commentaires', 20, 500, 100, step=10)
    with col_model:
        yt_model_options = [k for k, p in MODELS.items() if p.loaded]
        yt_model = st.selectbox('Modèle', yt_model_options, key='yt_model')
    with col_btn:
        st.write('')
        st.write('')
        fetch_btn = st.button('Analyser', type='primary', use_container_width=True)

    if fetch_btn:
        if not yt_url.strip():
            st.warning('Veuillez entrer une URL YouTube.')
        elif not yt_key.strip():
            st.warning('Veuillez entrer votre clé API YouTube.')
        else:
            video_id = extract_video_id(yt_url.strip())
            if not video_id:
                st.error('URL non reconnue. Collez un lien YouTube valide ou un ID à 11 caractères.')
            else:
                with st.spinner(f'Récupération des commentaires (max {max_comments})...'):
                    try:
                        raw_comments = fetch_youtube_comments(yt_key.strip(), video_id, max_comments)
                    except RuntimeError as e:
                        st.error(str(e))
                        raw_comments = []

                if raw_comments:
                    st.success(f'{len(raw_comments)} commentaires récupérés.')

                    with st.spinner('Analyse de toxicité en cours...'):
                        try:
                            all_probs = MODELS[yt_model].predict_batch(raw_comments)
                        except Exception as e:
                            st.error(f"Erreur durant l'analyse : {e}")
                            all_probs = []

                    if all_probs:
                        df_yt = pd.DataFrame({'comment_text': raw_comments})
                        for col in LABEL_COLS:
                            df_yt[f'prob_{col}'] = [p[col] for p in all_probs]
                        df_yt['toxicity_level'] = [toxicity_level(p)[0] for p in all_probs]

                        st.divider()
                        n = len(df_yt)
                        n_toxic  = sum(1 for p in all_probs if p['toxic']         > 0.5)
                        n_threat = sum(1 for p in all_probs if p['threat']        > 0.5)
                        n_hate   = sum(1 for p in all_probs if p['identity_hate'] > 0.5)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric('Commentaires analysés', f'{n:,}')
                        m2.metric('Toxiques détectés',     f'{n_toxic:,}',  f'{n_toxic/n*100:.1f}%')
                        m3.metric('Menaces',               f'{n_threat:,}')
                        m4.metric('Haine identitaire',     f'{n_hate:,}')

                        level_counts = df_yt['toxicity_level'].value_counts()
                        fig_yt = go.Figure(go.Bar(
                            x=level_counts.index,
                            y=level_counts.values,
                            marker_color=['#A32D2D', '#D85A30', '#EF9F27', '#854F0B', '#0F6E56'],
                        ))
                        fig_yt.update_layout(
                            title='Distribution des niveaux de toxicité',
                            xaxis_title='Niveau', yaxis_title='Commentaires',
                            height=280, margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_yt, use_container_width=True)

                        prob_cols  = [f'prob_{c}' for c in LABEL_COLS]
                        display_df = df_yt[['comment_text', 'toxicity_level'] + prob_cols].copy()
                        for c in prob_cols:
                            display_df[c] = display_df[c].apply(lambda x: f'{x*100:.1f}%')
                        st.dataframe(display_df, use_container_width=True, height=380)

                        csv_bytes = df_yt.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            '⬇️ Télécharger les résultats (CSV)',
                            data=csv_bytes,
                            file_name=f'amane_youtube_{video_id}.csv',
                            mime='text/csv',
                            type='primary',
                        )
                elif not raw_comments and fetch_btn:
                    st.info('Aucun commentaire récupéré pour cette vidéo.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Pipeline & Métriques
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Pipeline & Métriques':
    st.title('Pipeline AMANE — Évolution des modèles')
    st.caption('Architecture Knowledge Distillation en 3 phases pour la Darija.')

    tab_roadmap, tab_curves, tab_archi = st.tabs([
        'Feuille de route', "Courbes d'entraînement", 'Architecture',
    ])

    # ── Tab 1 : Feuille de route ───────────────────────────────────────────
    with tab_roadmap:
        roadmap = {
            'Phase':          ['Phase 1', 'Phase 1', 'Phase 2', 'Phase 3'],
            'Modèle':         [
                'Benchmark (OVR LogReg)',
                'XLM-R Teacher',
                'XLM-R Teacher* (adapté Darija)',
                'Student DistilBERT (Darija)',
            ],
            'Langue':         ['Anglais', 'Anglais', 'Anglais + Darija', 'Darija'],
            'Macro F1 (val)': ['—', '0.623', '0.815', '0.895'],
            'Paramètres':     ['—', '278M', '278M', '66M'],
            'Latence CPU':    ['< 10ms', '~240ms', '~240ms', '~60ms'],
            'Compression':    ['—', '1×', '1×', '4.2×'],
            'Statut':         ['Déployé', 'Déployé', 'Déployé', 'Déployé'],
        }
        df_roadmap = pd.DataFrame(roadmap)

        def color_statut(val):
            if val == 'Déployé':
                return 'background-color:#E1F5EE;color:#085041;font-weight:500'
            return 'background-color:#F0F0F0;color:#666'

        st.dataframe(
            df_roadmap.style.map(color_statut, subset=['Statut']),
            use_container_width=True, hide_index=True,
        )

        st.write('')
        st.markdown('**Progression Macro F1 — Darija toxic detection**')
        f1_labels  = ['Benchmark\n(N/A)', 'Teacher EN\n(proxy)', 'Teacher*\n(Darija)', 'Student\n(Darija)']
        f1_values  = [0.0, 0.183, 0.815, 0.895]
        f1_colors  = ['#888780', '#534AB7', '#0F6E56', '#D85A30']
        f1_texts   = ['N/A', '0.183', '0.815', '0.895']
        fig_f1 = go.Figure(go.Bar(
            x=f1_labels, y=f1_values,
            marker_color=f1_colors,
            text=f1_texts, textposition='outside',
        ))
        fig_f1.update_layout(
            yaxis=dict(range=[0, 1.05], title='Score'),
            height=320, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Tab 2 : Courbes d'entraînement ────────────────────────────────────
    with tab_curves:
        history_paths = {
            'XLM-R Teacher (EN)':          'models/teacher/training_history.json',
            'XLM-R Teacher* (Darija)':     'models/teacher_star/training_history_star.json',
            'Student DistilBERT (Darija)': 'models/student/training_history_student.json',
        }
        histories = {}
        for key, path in history_paths.items():
            try:
                with open(path) as fh:
                    histories[key] = json.load(fh)
            except Exception:
                pass

        if not histories:
            st.info("Historiques d'entraînement non disponibles localement.")
        else:
            for key, history in histories.items():
                info   = MODEL_INFO[key]
                epochs = [h['epoch'] for h in history]

                st.markdown(
                    f'<div style="font-weight:600;font-size:15px;color:{info["color"]};'
                    f'border-left:4px solid {info["color"]};padding-left:8px;'
                    f'margin:20px 0 8px;">{info["short"]} — {info["phase"]}</div>',
                    unsafe_allow_html=True,
                )

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Loss (train / val)', 'Métriques val'),
                )
                fig.add_trace(go.Scatter(
                    x=epochs, y=[h['train_loss'] for h in history],
                    name='Train loss', mode='lines+markers',
                    line=dict(color=info['color'], width=2),
                    marker=dict(size=6),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=epochs, y=[h['val_loss'] for h in history],
                    name='Val loss', mode='lines+markers',
                    line=dict(color=info['color'], width=2, dash='dot'),
                    marker=dict(size=6),
                ), row=1, col=1)

                f1_key  = 'val_f1'  if 'val_f1'  in history[0] else 'macro_f1'
                auc_key = 'val_auc' if 'val_auc' in history[0] else None

                if f1_key in history[0]:
                    fig.add_trace(go.Scatter(
                        x=epochs, y=[h[f1_key] for h in history],
                        name='F1 val', mode='lines+markers',
                        line=dict(color=info['color'], width=2),
                        marker=dict(size=6),
                    ), row=1, col=2)

                if auc_key and auc_key in history[0]:
                    fig.add_trace(go.Scatter(
                        x=epochs, y=[h[auc_key] for h in history],
                        name='AUC val', mode='lines+markers',
                        line=dict(color=info['color'], width=2, dash='dot'),
                        marker=dict(size=6),
                    ), row=1, col=2)

                fig.update_xaxes(title_text='Époque', tickmode='linear', dtick=1)
                fig.update_layout(
                    height=260,
                    margin=dict(l=40, r=20, t=40, b=30),
                    showlegend=True,
                    legend=dict(x=1.02, y=1.0),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3 : Architecture ──────────────────────────────────────────────
    with tab_archi:
        st.markdown("""
### Pipeline Knowledge Distillation — AMANE

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Jigsaw EN                                             │
│  Jigsaw Dataset (EN, 159k) ──► XLM-R Teacher ──► 6 labels       │
│  BCEWithLogitsLoss + pos_weight   278M params     F1 = 0.623     │
└─────────────────────────────────────┬────────────────────────────┘
                                      │ teacher_best.pt
                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Adaptation domaine Darija (DarLoad)                   │
│  Teacher gelé + DarLoad ──► Masked Loss ──► Teacher* (Darija)    │
│  binary toxic/non-toxic      col 0 seulement   F1  = 0.815       │
│  2 epochs, LR = 1e-5         5 labels gelés    AUC = 0.914       │
└─────────────────────────────────────┬────────────────────────────┘
                                      │ pseudo-labels Teacher* → DarLoad
                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Distillation Student                                   │
│  Soft labels Teacher* ──► Hybrid Loss ──► Student DistilBERT     │
│  DarLoad (conf ≥ 0.3)       α=0.3 hard        66M params         │
│  4 epochs                   (1-α)=0.7 soft    F1  = 0.895        │
│                                                4× plus rapide    │
└──────────────────────────────────────────────────────────────────┘
```

### Hybrid Loss — Phase 3

| Composante | Poids | Description |
|---|---|---|
| BCE Hard | α = 0.30 | Supervision label `toxic` DarLoad ground-truth |
| BCE Soft | 1-α = 0.70 | Distillation sur 6 soft labels du Teacher* |
| **Total** | **1.0** | `L = 0.3 × BCE_hard + 0.7 × BCE_soft` |

### Architectures comparées

| Modèle | Backbone | Pooling | Layers | Hidden | Tokenizer |
|---|---|---|---|---|---|
| Teacher / Teacher* | XLM-RoBERTa-base | Mean pooling | 12 | 768 | SentencePiece 250k |
| Student | DistilBERT-multilingual | CLS token | 6 | 768 | WordPiece 119k |
| Benchmark | TF-IDF + LogReg OVR | — | — | 10k features | — |

### Datasets

| Dataset | Taille | Labels | Usage |
|---|---|---|---|
| Jigsaw Toxic Comments | 159k EN | 6 labels multi | Teacher train |
| DarLoad | ~10k DA | binary toxic | Teacher* fine-tune + pseudo-labels |
| Pseudo-labels DarLoad | ~8k DA (conf ≥ 0.3) | 6 soft labels | Student train |
""")
