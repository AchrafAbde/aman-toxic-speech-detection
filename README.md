# AMAN — Multi-label Toxic Speech Detection in Moroccan Darija

> A multi-label NLP system for detecting toxic speech in Moroccan Darija using transformer models and knowledge distillation.

## Overview

AMAN is a machine learning project focused on toxic-speech detection in Moroccan Darija.

The project explores transformer-based classification and knowledge distillation to build a smaller and faster inference model.

## Results

- Fine-tuned an XLM-RoBERTa teacher model on approximately 160,000 comments.
- Achieved a Macro F1 score of 0.623 with the initial teacher model.
- Trained a Darija teacher model reaching a Macro F1 of 0.815.
- Distilled the model into a DistilBERT student model reaching a Macro F1 of 0.895.
- Built an inference application with Streamlit.

## Architecture

```text
XLM-R Teacher
      ↓
Darija Teacher
      ↓
Knowledge Distillation
      ↓
DistilBERT Student

Technologies
Python
PyTorch
Hugging Face Transformers
NLP
Streamlit
Scikit-learn
Project Context

Team project (4 members) developed as part of the AI & Data Science engineering curriculum.

Repository Structure
.
├── app.py
├── predictor.py
├── test_teacher.py
├── utils.py
├── requirements.txt
└── models/
    ├── student/
    ├── teacher/
    └── teacher_star/

Highlights
Multi-label toxic-speech classification
Moroccan Darija NLP
Transformer-based modeling
Knowledge distillation
Smaller and faster student model
Streamlit inference application
