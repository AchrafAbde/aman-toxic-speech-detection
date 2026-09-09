---
title: AMANE Toxicity Detector
emoji: 🛡️
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
---

# AMANE — Détection de discours toxiques en Darija

Projet Master 4 IA & Data Science — Architecture Knowledge Distillation multi-label pour la détection de toxicité en Darija marocain.

## Modèles déployés

| Phase | Modèle | Params | Macro F1 |
|---|---|---|---|
| Phase 1 | Benchmark OVR LogReg | — | — |
| Phase 1 | XLM-R Teacher (EN) | 278M | 0.623 |
| Phase 2 | XLM-R Teacher* (Darija) | 278M | 0.815 |
| Phase 3 | Student DistilBERT (Darija) | 66M | 0.895 |

## Architecture

```
Jigsaw EN → XLM-R Teacher → DarLoad (masked loss) → Teacher* → pseudo-labels → Student DistilBERT
```
