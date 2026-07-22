# ROADMAP — MedStay-CI

Rythme : 10–20 h/semaine → **6 à 8 semaines calendaires** (~33 jours-effort).
Objectif : tag `v1.0.0` mi-septembre, avant la vague d'offres de stage.
Principe directeur : **déployer tôt, enrichir ensuite.** Lien public dès la phase 1.

---

## Phase 0 — Setup (2 j)

- [ ] Créer le repo GitHub public (`medstay-ci` ou équivalent), licence MIT
- [ ] Structure de dossiers complète (voir SPEC §5) + fichiers vides `__init__.py`
- [ ] `uv venv` + `requirements.txt` versions épinglées (fastapi, uvicorn, streamlit,
      lightgbm, mapie, scikit-learn, pandas, pydantic, pytest, pytest-cov, httpx, ruff, joblib)
- [ ] `src/config.py` : SEED=42, ALPHA=0.10, chemins, ratios de split
- [ ] `.gitignore` (données brutes, .venv, .env, __pycache__, .ruff_cache)
- [ ] Copier CLAUDE.md à la racine, SPEC.md et ROADMAP.md dans `docs/`
- [ ] Premier commit : `chore: project skeleton`

## Phase 1 — Squelette déployable (3 j, inclut réapprentissage Docker)

- [ ] Jour 1 : remise à niveau Docker en local (`hello-world`, puis un FastAPI minimal
      conteneurisé qui répond sur /health)
- [ ] `src/api/main.py` : `GET /health` + `POST /predict` FACTICE (renvoie un intervalle codé
      en dur au bon format JSON)
- [ ] `app/streamlit_app.py` minimal : un bouton qui appelle l'API et affiche la réponse
- [ ] `start.sh` : uvicorn (port 8000, arrière-plan) + streamlit (port 7860)
- [ ] Dockerfile SIMPLE (~10 lignes, pas de multi-stage ici), EXPOSE 7860
- [ ] Test local : `docker build` + `docker run -p 7860:7860` → l'UI répond
- [ ] Créer le Space HF (SDK Docker) à la main, push manuel → **lien public fonctionnel**

## Phase 2 — CI/CD précoce (3 j, GitHub Actions découvert ici)

Construire par incréments, chaque étape verte avant la suivante :
- [ ] (1) `ci.yml` : ruff check seul
- [ ] (2) + pytest sur 2 tests triviaux (health → 200, predict factice → schéma OK)
- [ ] (3) + docker build (sans push)
- [ ] (4) `deploy.yml` : push auto vers le Space HF sur main (secret `HF_TOKEN`)
- [ ] Badges CI + démo dans le README
- [ ] Vérifier la chaîne complète : un push modifie le Space en ligne sans action manuelle

## Phase 3 — Données (4 j)

- [ ] `src/data/load.py` : téléchargement UCI reproductible (pas de CSV commité)
- [ ] `notebooks/01_eda.ipynb` : distribution cible, manquants, cardinalités, corrélations
- [ ] `src/data/preprocess.py` : `?`→NaN ; drop `weight` ; décision documentée sur
      `payer_code`/`medical_specialty` ; première admission par `patient_nbr` (justifier
      dans le README) ; regroupement ICD-9 en catégories cliniques ; features dérivées
      (nb médicaments modifiés, ratio procédures/diagnostics) ; encodage natif LightGBM
- [ ] `src/data/split.py` : split 60/20/20 stratifié sur cible discrétisée, SAUVEGARDÉ
      (parquet + indices), jamais recalculé
- [ ] Tests : déterminisme du preprocessing, schéma de sortie, gestion NaN

## Phase 4 — Modélisation + conformal (4 j)

- [ ] `src/models/train.py` : baseline LightGBM L2 (MAE/RMSE/R² loggés)
- [ ] Deux LGBM quantiles (alpha 0.05 / 0.95) sur le train
- [ ] `src/models/calibrate.py` : `MapieQuantileRegressor`, calibration sur le calib set,
      alpha=0.10
- [ ] Export artefacts joblib dans `models/` + métadonnées (date, versions, métriques)
- [ ] Brancher l'API sur les vrais artefacts (remplacer le predict factice)

## Phase 5 — Validation statistique (3 j) ⭐ cœur différenciant

- [ ] `src/models/evaluate.py` produit et sauvegarde en figures :
  - [ ] Couverture empirique test (chiffre exact, attendu ≈ 0.90)
  - [ ] Largeur moyenne + distribution des largeurs
  - [ ] Couverture par sous-groupe (âge, type d'admission, nb diagnostics)
  - [ ] Courbe couverture empirique vs alpha (0.01 → 0.30) vs diagonale
  - [ ] Comparaison quantiles calibrés vs NON calibrés (preuve d'utilité du conformal)
- [ ] Passage au multi-stage Docker : python:3.11-slim, non-root, HEALTHCHECK,
      image < 1 Go (taille notée dans le README)

## Phase 6 — API réelle (4 j)

- [ ] `src/api/schemas.py` : `PatientInput` complet avec bornes physiologiques,
      catégories validées, exemples OpenAPI ; `PredictionOutput`
- [ ] Endpoints définitifs : /health, /model-info, /predict (alpha optionnel borné),
      /predict/batch
- [ ] Chargement via `lifespan`, logging structuré (latence, largeur, rien d'identifiant)
- [ ] 422 propres et messages d'erreur explicites

## Phase 7 — Tests sérieux (3 j)

- [ ] Propriétés métier : lower ≤ point ≤ upper ; largeur > 0 ; alpha↓ ⇒ largeur↑ ; bornes ≥ 0
- [ ] Validation : hors bornes → 422 ; champs manquants → 422 ; types faux → 422
- [ ] Non-régression : prédictions figées sur patients de référence
- [ ] `pytest --cov` ≥ 70 %, rapport dans la CI

## Phase 8 — Front Streamlit (4 j)

- [ ] Formulaire patient pré-rempli (cas réaliste, test en un clic)
- [ ] Slider alpha → intervalle qui s'élargit en direct (moment fort de la démo)
- [ ] Graphique point + intervalle, distribution de la cible en fond
- [ ] Onglet Performance (figures de la phase 5)
- [ ] Onglet À propos (conformal vulgarisé en 5 lignes + limites)
- [ ] Uniquement des appels HTTP à l'API — aucun chargement de modèle côté front

## Phase 9 — Finalisation (3 j)

- [ ] README complet : lien démo + GIF 15–20 s, problème métier, pourquoi le conformal,
      schéma d'archi, résultats chiffrés, **Limites connues** (dataset daté, couverture
      marginale seulement, pas de validation clinique, échangeabilité), reproduction
      `docker compose up`
- [ ] `infra/` : Dockerfile App Runner + Terraform commenté, note "non déployé par choix"
- [ ] Nettoyage historique de commits si besoin, tag `v1.0.0`
- [ ] GIF LinkedIn + post

---

## Jalons de contrôle

| Fin de semaine | État attendu |
|---|---|
| S1 | Lien public HF en ligne (API factice) + CI/CD complet vert |
| S3 | Modèle réel calibré branché sur l'API, validation de couverture faite |
| S5 | API définitive testée + front complet |
| S6–S8 | README, GIF, tag v1.0.0, post LinkedIn |

## Règle en cas de retard

Couper dans cet ordre : SHAP → features dérivées avancées → onglet Performance du front →
mypy. Ne jamais couper : déploiement, CI/CD, validation de couverture, tests de propriétés.
