# Prediction-Fake-news-Project

Pipeline NLP complet pour la detection de desinformation sur des titres de presse, avec:
- pretraitement texte (regex, contractions, stopwords, lemmatisation),
- modelisation TensorFlow (Dense TF-IDF + BiLSTM embeddings),
- exposition REST via FastAPI.

## Objectif

Classer un titre d'article en:
- `REAL` (fiable)
- `FAKE` (trompeur)

Le projet suit l'enonce ECF "Detection automatique de desinformation dans les titres de presse".

## Structure du projet

```text
fake_news_nlp/
├── notebook/
│   ├── fake_news_prediction.ipynb
├── api/
│   └── main.py
├── models/
│   ├── best_model.keras
│   └── vectorizer.pkl
├── data/
│   ├── fake_or_real_news.csv
│   ├── news.csv
│   └── titles_clean.csv
└── requirements.txt
```

## Dataset

Source principale: Kaggle Fake or Real News Dataset  
Fichier attendu dans `data/`: `news.csv` (ou copie de `fake_or_real_news.csv` vers `news.csv`).

Colonnes utilisees:
- `title` (texte utilise)
- `label` (`REAL` / `FAKE`)

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Execution du notebook

Ouvrir et executer:
- `notebook/ecf_fake_news.ipynb`

Ou execution batch:

```bash
python -m jupyter nbconvert --to notebook --execute notebook/ecf_fake_news.ipynb --output ecf_fake_news.executed.ipynb --output-dir notebook
```

Le notebook:
- prepare les donnees,
- entraine/compare les modeles,
- evalue les performances,
- sauvegarde les artefacts:
  - `models/best_model.keras`
  - `models/vectorizer.pkl`
  - `data/titles_clean.csv`

## Lancer l'API

Depuis le dossier `fake_news_nlp`:

```bash
uvicorn api.main:app --reload
```

Documentation interactive:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Endpoints

### `GET /health`

Reponse:

```json
{"status":"ok","model":"fake_news_detector"}
```

### `POST /predict`

Entree:

```json
{"title":"Scientists discover new treatment"}
```

Sortie:

```json
{
  "title": "Scientists discover new treatment",
  "label": "REAL",
  "confidence": 0.87
}
```

### `POST /predict/batch`

Entree:

```json
{"titles":["title 1","title 2"]}
```

Sortie:

```json
{
  "predictions": [
    {"title":"title 1","label":"FAKE","confidence":0.92},
    {"title":"title 2","label":"REAL","confidence":0.71}
  ]
}
```

## Gestion des erreurs API

Conforme a l'enonce:
- titre vide ou espaces uniquement -> `422`
- titre > 300 caracteres -> `400`
- champ `title` absent -> `422`
- batch vide ou > 50 titres -> `400`

## Resultats obtenus (exemple)

Comparaison des 2 modeles:
- Dense (TF-IDF): rapide, simple a maintenir, bon compromis prod.
- BiLSTM (Embedding): leger gain en performance brute, cout de calcul plus eleve.

## Pistes d'amelioration

- ajustement du seuil de decision selon l'objectif metier (triage FAKE),
- calibration des probabilites,
- enrichissement des donnees et nettoyage plus robuste,
- suivi de performance en production.

## Auteur

Projet pedagogique ECF M2 - Concepteur Developpeur en Intelligence Artificielle.
