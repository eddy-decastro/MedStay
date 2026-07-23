# SPEC — MedStay-CI

Prédiction de durée de séjour hospitalier avec intervalles garantis (Conformal Prediction),
pipeline MLOps complet, 100 % tiers gratuits.

## 1. Objectif

Livrer une application web publique qui, pour un patient diabétique, prédit la durée
d'hospitalisation sous forme d'**intervalle avec garantie statistique de couverture à 90 %**
(Conformalized Quantile Regression), servie par une API conteneurisée, testée et déployée
automatiquement à chaque push sur `main`.

Effort : **40 % stats/ML — 60 % ingénierie (API, Docker, CI/CD, déploiement)**.

## 2. Problème métier

Les hôpitaux doivent anticiper la durée de séjour pour gérer lits et ressources. Un modèle
classique renvoie une valeur unique ("4,2 jours") sans information de fiabilité. La Conformal
Prediction fournit un intervalle garanti mathématiquement ("entre 3 et 5 jours avec 90 % de
couverture"), qui s'élargit automatiquement pour les cas complexes ou rares. Sans incertitude
quantifiée, une prédiction est inutilisable pour une décision clinique.

## 3. Données

- **UCI Diabetes 130-US Hospitals (1999–2008)** : ~101 766 séjours, 50 variables. Accès libre.
- Cible : `time_in_hospital` (1–14 jours) — régression.
- Pièges connus à traiter et documenter :
  - `?` = manquant ; `weight` ~97 % manquant → drop.
  - Doublons `patient_nbr` → garder la première admission (indépendance/échangeabilité).
  - `diag_1/2/3` en codes ICD-9 → regrouper en catégories cliniques larges.
- **Split unique sauvegardé : train 60 % / calibration 20 % / test 20 %**, seed fixée.

## 4. Méthode statistique

1. Baseline : LightGBM L2 (MAE, RMSE, R²) — référence, pas le produit.
2. Régression quantile : deux LightGBM (`objective='quantile'`, alpha 0.05 et 0.95).
3. Calibration CQR : `MapieQuantileRegressor` (MAPIE), fit sur train, calibration sur le set
   de calibration, `alpha=0.10`.
4. **Validation (le cœur différenciant du projet)** :
   - Couverture empirique sur le test set (attendu ≈ 0.90, chiffre exact rapporté)
   - Largeur moyenne des intervalles (métrique d'utilité clinique)
   - Couverture conditionnelle par sous-groupe (âge, type d'admission, nb diagnostics)
   - Courbe couverture empirique vs alpha (0.01 → 0.30), doit suivre la diagonale
   - Comparaison avec quantiles non calibrés → preuve que la calibration sert
5. Optionnel (fin de projet) : SHAP + analyse des features corrélées à la largeur d'intervalle.

## 5. Architecture

```
                    ┌─────────────────────────────┐
                    │   GitHub repo (public)      │
                    │  push main                  │
                    └──────┬───────────────┬──────┘
                           │               │ webhook push
              GitHub Actions CI            ▼  (auto-deploy natif)
           lint + tests + build   ┌────────────────────────────────────┐
              (garde-fou)         │   Render web service (Docker)      │
                                  │                                    │
                                  │  ┌───────────┐  HTTP  ┌─────────┐  │
                                  │  │ Streamlit │ ─────► │ FastAPI │  │
                                  │  │  $PORT    │        │ port    │  │
                                  │  │ (exposé)  │ ◄───── │ 8000    │  │
                                  │  └───────────┘  JSON  └────┬────┘  │
                                  │       lancés par start.sh  │       │
                                  │                     ┌──────▼─────┐ │
                                  │                     │ Artefacts  │ │
                                  │                     │ LGBM q05/95│ │
                                  │                     │ + MAPIE CQR│ │
                                  │                     └────────────┘ │
                                  └────────────────────────────────────┘

Hors runtime (local) :
UCI CSV ─► preprocess ─► split 60/20/20 ─► LGBM quantiles (train)
        ─► MAPIE calibration (calib) ─► validation (test) ─► export joblib
```

Règles :
- Le front ne charge jamais le modèle : il appelle l'API.
- L'entraînement est hors ligne ; seule l'inférence est déployée.
- Un seul service Render, deux processus (`start.sh` : uvicorn 8000 interne + Streamlit `$PORT` exposé).

## 6. API (FastAPI + Pydantic v2)

- `GET /health` — liveness.
- `GET /model-info` — version, date d'entraînement, couverture mesurée sur test.
- `POST /predict` — un patient ; `alpha` optionnel (défaut 0.10, borné [0.01, 0.50]).
- `POST /predict/batch` — liste de patients.
- Schémas : `PatientInput` (bornes physiologiques via `Field(ge=..., le=...)`, catégories
  validées) → 422 propre sur input aberrant ; `PredictionOutput` (point_estimate, lower_bound,
  upper_bound, interval_width, coverage_level).
- Modèles chargés au démarrage via `lifespan`. Logging structuré (latence, largeur, pas de
  données identifiantes).

## 7. Tests (pytest, cible ≥ 70 %)

- Données : preprocessing déterministe, bon schéma de sortie, NaN gérés.
- API : codes HTTP, schémas de réponse, 422 sur inputs invalides.
- **Propriétés métier** : lower ≤ point ≤ upper ; largeur > 0 ; alpha↓ ⇒ largeur↑ ;
  bornes ≥ 0.
- Non-régression : prédictions figées sur un jeu de patients de référence.

## 8. Conteneurisation

- Phase 1 : Dockerfile simple (~10 lignes) pour déployer vite.
- Phase 5 : multi-stage, `python:3.11-slim`, non-root, HEALTHCHECK sur `/health`,
  LightGBM CPU-only, image < 1 Go (taille documentée dans le README).
- `docker-compose.yml` pour API + front en local en une commande.

## 9. CI/CD (GitHub Actions) — construit par incréments

- `ci.yml` (push + PR) : ruff check/format → pytest + couverture → docker build (sans push).
- Déploiement : auto-deploy natif Render (webhook sur push `main`). Pas de `deploy.yml` ni de
  secret `HF_TOKEN` : Render reconstruit l'image à partir du `Dockerfile` du repo.
- Incréments : (1) ruff seul → (2) + pytest → (3) + build. Vert avant d'empiler ; le déploiement
  est délégué à Render, pas à GitHub Actions.
- Badges README : CI, couverture, lien démo.

## 10. Contrainte de coût : 100 % gratuit

- **Production : Render free tier uniquement** (web service Docker, 0 €). Tradeoff assumé :
  le service s'endort après ~15 min d'inactivité → cold start ~30-50 s au premier appel.
- **AWS : documenté, jamais exécuté.** Free tier post-juillet 2025 = 200 $ / 6 mois puis frais.
  Dossier `infra/` : Dockerfile compatible App Runner + Terraform commenté, avec note README
  expliquant ce choix délibéré (raisonnement d'ingénieur, pas une lacune).
- GitHub Actions gratuit (repo public). Aucune carte bancaire nulle part.

## 11. Frontend (Streamlit)

1. Formulaire patient pré-rempli avec un cas réaliste (test en un clic).
2. Slider de confiance (alpha) — l'élargissement de l'intervalle en direct est le moment fort.
3. Graphique : point + intervalle sur axe temporel, distribution de la cible en fond.
4. Onglet "Performance" : couverture empirique, largeur moyenne, courbe couverture vs alpha,
   couverture par sous-groupe.
5. Onglet "À propos" : conformal prediction vulgarisé en 5 lignes + limites du dataset.

## 12. Livrables finaux

- Dépôt GitHub public structuré, historique de commits propre, tag `v1.0.0`.
- Démo publique Render (lien) + GIF 15–20 s pour LinkedIn.
- README : démo, problème métier, pourquoi le conformal, schéma d'architecture, résultats
  chiffrés, **section "Limites connues"** (dataset 1999–2008 ; couverture marginale et non
  conditionnelle ; pas de validation clinique ; hypothèse d'échangeabilité discutable en
  multi-sites), reproduction en une commande (`docker compose up`).
- `infra/` AWS documenté non déployé.

## 13. Ligne CV cible

> Application MLOps end-to-end de prédiction de durée de séjour hospitalier avec intervalles
> de confiance garantis (Conformal Prediction, MAPIE) — FastAPI, Docker, GitHub Actions,
> déployée sur Render. Couverture empirique validée à 90 % sur ~20 000 patients de test.

## 14. Priorités si le temps manque

Déploiement fonctionnel > CI/CD > validation de couverture > tests > feature engineering
avancé > SHAP. Un projet déployé imparfait vaut mieux qu'un projet parfait en local.
