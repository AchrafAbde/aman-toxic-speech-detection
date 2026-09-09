"""
Script de test rapide pour valider l'intégration du Teacher XLM-R.

Usage :
    cd C:\\Projets\\Aman\\Aman_App
    python test_teacher.py

Ce script :
1. Charge le Teacher (vérifie checkpoint, tokenizer, seuils)
2. Lance des prédictions sur des exemples EN (Jigsaw-style) + Darija
3. Affiche probabilités + labels binaires (avec seuils tunés)
"""
import time
import sys
from predictor import load_teacher, load_benchmark
from utils import LABEL_COLS, LABEL_META


# ── Exemples de test ────────────────────────────────────────────────────────
TEST_TEXTS = [
    # ── Anglais (langue d'entraînement) ────────────────────────────────────
    ('EN-clean',  "Hello, I really enjoyed your article. Thank you for sharing!"),
    ('EN-toxic',  "You are such an idiot, nobody likes you here."),
    ('EN-threat', "I will find you and you will regret what you said."),
    ('EN-id-hate',"All people from that country are stupid and useless."),

    # ── Darija (transfert interlingue — vrai test du Teacher) ──────────────
    ('DA-clean',  "salam khouya, kifach hawalek? Inchallah la ssaha labas."),
    ('DA-insult', "nta hmar, ma3andek 3aql."),
    ('DA-toxic',  "khssek tskout 7it ma katfhem walou."),

    # ── Français ───────────────────────────────────────────────────────────
    ('FR-clean',  "Merci beaucoup pour cette explication très claire."),
    ('FR-toxic',  "T'es vraiment con, ferme ta gueule."),
]


def fmt_probs(probs: dict, labels: dict = None, thresholds: dict = None) -> str:
    """Formate les probabilités sur une ligne (avec marqueur si > seuil)."""
    parts = []
    for col in LABEL_COLS:
        p   = probs[col]
        fr  = LABEL_META[col]['fr']
        if labels and labels[col]:
            t = thresholds[col]
            parts.append(f'\033[91m{fr}={p:.3f}*\033[0m(t={t})')
        elif p > 0.3:
            parts.append(f'\033[93m{fr}={p:.3f}\033[0m')
        else:
            parts.append(f'{fr}={p:.3f}')
    return ' | '.join(parts)


def main():
    print('=' * 75)
    print('  AMANE — Test d\'intégration du Teacher XLM-R')
    print('=' * 75)

    # ── 1. Charger le Teacher ──────────────────────────────────────────────
    print('\n[1/3] Chargement du Teacher...')
    t0 = time.time()
    teacher = load_teacher()
    elapsed = time.time() - t0

    if not teacher.loaded:
        print(f'\n[ECHEC] Chargement echoue apres {elapsed:.1f}s')
        print(f'   Erreur : {teacher.error}')
        sys.exit(1)

    print(f'[OK] Teacher charge en {elapsed:.1f}s')
    print(f'   Device       : {teacher.device}')
    print(f'   Seuils tunes : {teacher.thresholds}')

    # ── 2. Charger le Benchmark (pour comparaison) ─────────────────────────
    print('\n[2/3] Chargement du Benchmark...')
    benchmark = load_benchmark()
    if benchmark.loaded:
        print('[OK] Benchmark charge')
    else:
        print(f'[WARN] Benchmark non disponible : {benchmark.error}')

    # ── 3. Prédictions sur les exemples ────────────────────────────────────
    print('\n[3/3] Prédictions sur les exemples de test')
    print('-' * 75)

    for tag, text in TEST_TEXTS:
        print(f'\n[{tag}] {text}')

        # Teacher avec seuils tunés
        t0 = time.time()
        result = teacher.predict_with_labels(text)
        infer_ms = (time.time() - t0) * 1000

        print(f'  Teacher  ({infer_ms:5.0f}ms): {fmt_probs(result["probs"], result["labels"], result["thresholds"])}')

        # Benchmark (si dispo)
        if benchmark.loaded:
            probs_bench = benchmark.predict(text)
            print(f'  Benchmark        : {fmt_probs(probs_bench)}')

    print('\n' + '=' * 75)
    print('  [OK] Test termine. Si les predictions Darija font du sens ->')
    print('     le transfert interlingue fonctionne, on peut passer au pseudo-labeling.')
    print('=' * 75)


if __name__ == '__main__':
    main()
