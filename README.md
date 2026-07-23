# MedStay-CI

Prédiction de durée de séjour hospitalier (patients diabétiques) avec intervalles de
prédiction garantis à 90 % par Conformalized Quantile Regression (CQR), servie par une
API FastAPI conteneurisée et déployée automatiquement sur Render.

Projet d'apprentissage (élève-ingénieur maths appliquées), 40 % stats/ML — 60 %
ingénierie/MLOps. Voir [docs/SPEC.md](docs/SPEC.md) pour l'énoncé complet et
[docs/ROADMAP.md](docs/ROADMAP.md) pour le détail des phases.

Statut : Phase 0 (setup) en cours.

## Reproduction locale

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
python -m src.data.load
python -m src.data.preprocess
python -m src.models.train
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

## Licence

MIT — voir [LICENSE](LICENSE).
