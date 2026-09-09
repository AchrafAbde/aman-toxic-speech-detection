import re

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

LABEL_META = {
    'toxic'         : {'fr': 'Toxique',         'color': '#D85A30'},
    'severe_toxic'  : {'fr': 'Très toxique',     'color': '#993C1D'},
    'obscene'       : {'fr': 'Obscène',          'color': '#EF9F27'},
    'threat'        : {'fr': 'Menace',           'color': '#A32D2D'},
    'insult'        : {'fr': 'Insulte',          'color': '#534AB7'},
    'identity_hate' : {'fr': 'Haine identitaire','color': '#0F6E56'},
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[USER]', text)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'==+\s*.+?\s*==+', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'([!?.])\1{2,}', r'\1\1', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def toxicity_level(probs: dict) -> tuple[str, str]:
    """Retourne (niveau, couleur) selon les probabilités détectées."""
    max_prob = max(probs.values())
    if probs.get('threat', 0) > 0.5 or probs.get('severe_toxic', 0) > 0.5:
        return 'Très élevé', '#A32D2D'
    elif max_prob > 0.7:
        return 'Élevé', '#D85A30'
    elif max_prob > 0.4:
        return 'Modéré', '#EF9F27'
    elif max_prob > 0.2:
        return 'Faible', '#854F0B'
    else:
        return 'Aucun', '#0F6E56'