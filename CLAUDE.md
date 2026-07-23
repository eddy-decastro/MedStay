# CLAUDE.md — MedStay-CI

Prédiction de durée de séjour hospitalier (patients diabétiques) avec intervalles de prédiction
garantis à 90 % par Conformalized Quantile Regression (CQR), servie par une API FastAPI
conteneurisée et déployée automatiquement sur Render.

Répartition d'effort du projet : 40 % stats/ML, 60 % ingénierie/MLOps.
Documents de référence : `docs/SPEC.md` (énoncé complet) et `docs/ROADMAP.md` (phases + checklists).

## Contexte utilisateur (IMPORTANT — mode pédagogique)

- Je suis élève-ingénieur en maths appliquées. Ce projet est un projet d'apprentissage ET une
  pièce de portfolio : je dois être capable de défendre chaque choix en entretien technique.
- **Ne me livre pas de gros blocs de code que je n'ai pas demandés.** Pour la logique cœur
  (preprocessing, modélisation, MAPIE, validation statistique) : propose une approche, explique,
  puis laisse-moi écrire ou écris avec moi étape par étape. Review mon code de manière critique.
- Pour le boilerplate (Dockerfile, YAML GitHub Actions, config, squelettes Pydantic) : tu peux
  générer directement, mais commente chaque section pour que je comprenne.
- Docker : j'en ai fait mais j'ai oublié. GitHub Actions : jamais utilisé. Explique ces parties
  pas à pas, une étape à la fois.
- Quand je fais une erreur méthodologique (fuite de données, split mal fait, métrique
  mal interprétée), dis-le franchement.

## Stack technique (figée — ne pas proposer d'alternatives sans raison forte)

- Python 3.11, gestion d'env : `uv` (fallback venv + pip)
- Données : UCI Diabetes 130-US Hospitals (1999–2008), cible = `time_in_hospital` (régression)
- ML : LightGBM — baseline L2 + deux modèles quantiles (alpha=0.05 et 0.95)
- Incertitude : MAPIE `MapieQuantileRegressor` (méthode CQR), couverture cible 90 % (alpha=0.10)
- API : FastAPI + Pydantic v2 (validation stricte)
- Front : Streamlit (appelle l'API en HTTP, ne charge JAMAIS le modèle directement)
- Conteneur : Docker (simple en phase 1, multi-stage en phase 5), python:3.11-slim, non-root
- Tests : pytest + pytest-cov (cible ≥ 70 %) + httpx TestClient
- Lint/format : ruff
- CI/CD : GitHub Actions (ci.yml) ; déploiement = auto-deploy natif Render (pas de deploy.yml)
- Déploiement : Render UNIQUEMENT (web service Docker, port dynamique via $PORT)

## Contraintes non négociables

1. **0 € de coût.** Aucun service payant, aucune carte bancaire. Render free tier uniquement
   (le service s'endort après ~15 min d'inactivité → cold start au réveil, tradeoff assumé).
   Pas de déploiement AWS réel : le dossier `infra/` contient un Terraform/Dockerfile AWS
   documenté mais jamais exécuté en CI.
2. **Split des données fait UNE FOIS et sauvegardé** : train 60 / calibration 20 / test 20.
   Jamais recalculé aléatoirement. Seed globale fixée dans `src/config.py`.
3. **Le set de calibration ne sert qu'à MAPIE**, jamais à l'entraînement ni au choix
   d'hyperparamètres. Le test set ne sert qu'à l'évaluation finale.
4. **Front découplé** : Streamlit → HTTP → FastAPI → artefacts. Un seul service Render, deux
   processus lancés par `start.sh` (uvicorn port 8000 interne, Streamlit exposé sur $PORT).
5. **Pas de données commitées** : les CSV sont téléchargés par script (`src/data/load.py`).
   Les artefacts modèles (joblib) dans `models/` peuvent être versionnés (petits).
6. Aucune donnée personnelle réelle. Dataset public anonymisé uniquement.

## Structure du dépôt

```
├── src/
│   ├── data/          # load.py (téléchargement), preprocess.py, split.py
│   ├── models/        # train.py, calibrate.py (MAPIE), evaluate.py
│   ├── api/           # main.py, schemas.py (Pydantic), service.py
│   └── config.py      # chemins, seed, hyperparamètres, ALPHA=0.10
├── app/               # streamlit_app.py
├── tests/             # test_data.py, test_api.py, test_properties.py
├── notebooks/         # 01_eda.ipynb, 02_modeling.ipynb — exploration SEULEMENT
├── models/            # artefacts joblib exportés
├── infra/             # AWS documenté, non déployé
├── docs/              # SPEC.md, ROADMAP.md
├── .github/workflows/ # ci.yml (deploy = auto-deploy natif Render)
├── Dockerfile
├── docker-compose.yml
├── start.sh           # lance uvicorn + streamlit dans le conteneur Render
├── requirements.txt   # versions épinglées
└── README.md
```

Règle : aucun code de production dans les notebooks. Tout ce qui est réutilisé vit dans `src/`.

## Commandes

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
python -m src.data.load          # télécharge le dataset UCI
python -m src.data.preprocess    # nettoyage + features + split sauvegardé
python -m src.models.train       # baseline + quantiles + calibration MAPIE + export joblib
python -m src.models.evaluate    # couverture empirique, largeur, sous-groupes → figures
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
pytest --cov=src --cov-report=term-missing
ruff check . && ruff format .
docker build -t medstay . && docker run -p 7860:7860 medstay   # local : $PORT non défini → 7860
```

## Conventions de code

- Type hints partout dans `src/`. Docstrings courtes (une ligne) sur les fonctions publiques.
- Pydantic v2 : `Field(ge=..., le=...)` pour toutes les bornes physiologiques des inputs.
- Logging via `logging`, pas de `print` dans `src/`.
- Commits : format `type(scope): message` (feat, fix, test, ci, docs, refactor).
  Petits commits fréquents — l'historique fait partie du portfolio.
- Pas de `# type: ignore` ni de `except Exception: pass` sans justification en commentaire.

## Règles méthodologiques ML (à surveiller activement)

- Doublons `patient_nbr` : garder la PREMIÈRE admission par patient (indépendance des
  observations, requise par l'hypothèse d'échangeabilité du conformal). Documenter dans le README.
- `?` = valeur manquante. Drop `weight` (~97 % manquant) ; `payer_code` et `medical_specialty`
  à évaluer pendant l'EDA.
- Regrouper `diag_1/2/3` (codes ICD-9) en catégories cliniques larges (circulatoire,
  respiratoire, digestif, diabète, blessure, musculosquelettique, génito-urinaire, néoplasmes, autre).
- Encodage : ordinal/natif pour LightGBM (pas de one-hot massif).
- Tout preprocessing est fitté sur le train uniquement, appliqué aux trois splits.
- Métriques de validation obligatoires (phase 5) : couverture empirique test (≈ 0.90),
  largeur moyenne d'intervalle, couverture par sous-groupe (âge, type d'admission),
  courbe couverture vs alpha, comparaison avec quantiles NON calibrés.
- Propriétés à tester : lower ≤ point ≤ upper ; largeur > 0 ; alpha plus petit ⇒ intervalle
  plus large ; bornes ≥ 0.

## Déploiement Render

- Web service de type Docker, port dynamique : Render injecte `$PORT`, l'app DOIT bind dessus.
- `start.sh` : uvicorn en arrière-plan sur 8000, puis Streamlit sur `$PORT` (fallback 7860 en local).
  Streamlit appelle `http://localhost:8000`.
- Auto-deploy natif : Render connecte le repo GitHub et redéploie à chaque push sur la branche
  suivie. Pas de `deploy.yml`, pas de secret `HF_TOKEN`. La CI (ruff + pytest + build) reste le
  garde-fou avant que Render ne construise l'image.
- Free tier : le service s'endort après ~15 min d'inactivité (cold start ~30-50 s au réveil).
  Limite assumée, à documenter dans la section "Limites connues" du README.
- Construire le CI/CD par incréments : (1) ruff seul → (2) + pytest → (3) + docker build.
  Une étape verte avant d'ajouter la suivante (le déploiement lui-même est géré par Render).

## Ordre des phases (résumé — détail dans docs/ROADMAP.md)

0. Setup repo/env → 1. Squelette déployable (API factice + Docker simple + service Render créé) →
2. CI/CD précoce → 3. Données/EDA/split → 4. Modélisation + MAPIE → 5. Validation stats →
6. API réelle → 7. Tests → 8. Front Streamlit → 9. README/GIF/infra AWS/tag v1.0.0.

Principe : déployer tôt (lien public dès la phase 1), enrichir ensuite. Si un choix doit être
fait entre perfection locale et déploiement fonctionnel, le déploiement gagne toujours.
