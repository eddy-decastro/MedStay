# Journal de développement — MedStay-CI

Résumé complet du contexte projet et de tout ce qui a été construit, phase par
phase, avec les décisions prises et les bugs trouvés en cours de route.

---

## 1. Contexte du projet

**Objectif** : prédire la durée de séjour hospitalier de patients diabétiques
avec un **intervalle de confiance garanti à 90 %** (Conformalized Quantile
Regression), servi par une API conteneurisée et déployée automatiquement.

**Profil de l'utilisateur** : élève-ingénieur en maths appliquées. Le projet
est à la fois un apprentissage et une pièce de portfolio — chaque choix doit
pouvoir se défendre en entretien technique.

**Répartition d'effort visée** : 40 % stats/ML, 60 % ingénierie/MLOps.

**Contraintes non négociables** :
- **0 € de coût**, aucune carte bancaire.
- Split des données calculé **une seule fois** et sauvegardé.
- Le set de calibration ne sert **qu'à MAPIE**, jamais à l'entraînement.
- Front Streamlit **découplé** de l'API (communication HTTP uniquement).
- Pas de données commitées (téléchargement par script).

**Stack figée** : Python 3.11, `uv`, LightGBM, MAPIE (CQR), FastAPI + Pydantic
v2, Streamlit, Docker, GitHub Actions, déploiement Render.

**Changement de cap en cours de route** : le déploiement était initialement
prévu sur Hugging Face Spaces. HF a retiré son SDK Docker gratuit → migration
vers **Render** (web service Docker, port `$PORT` dynamique, auto-deploy natif
sur CI verte).

---

## 2. Phase 0 — Setup

Structure de dossiers, `uv venv`, `requirements.txt` épinglé, `src/config.py`
(seed=42, alpha=0.10, ratios de split), `.gitignore`, premier commit.

---

## 3. Phase 1 — Squelette déployable

- `src/api/main.py` : `/health` + `/predict` **factice** (JSON codé en dur, au
  bon format).
- `app/streamlit_app.py` minimal : un bouton qui appelle l'API.
- `start.sh` : uvicorn (8000, interne) + Streamlit (exposé).
- `Dockerfile` simple (~10 lignes).
- Build et run testés en local, puis **service Render créé** (web service
  Docker, connecté au repo GitHub, `Auto-Deploy: After CI checks pass`).

**Bug rencontré** : `--retry-connrefused` de curl ne suffisait pas à attendre
le démarrage du conteneur — avec `-p`, `docker-proxy` accepte la connexion dès
le `docker run` puis la réinitialise tant que l'app n'écoute pas, ce qui ne
déclenche jamais un vrai "connection refused". Corrigé avec
`--retry-all-errors`.

---

## 4. Phase 2 — CI/CD (`.github/workflows/ci.yml`)

Construit par incréments, chaque étape verte avant la suivante :

1. `lint` — ruff check + format.
2. `test` — pytest + couverture, job parallèle.
3. `docker` — build de l'image + smoke test du conteneur lancé.

**Décision** : pas de job `deploy` custom avec secret GitHub. Render propose
nativement `Auto-Deploy: After CI checks pass`, qui écoute directement les
status checks GitHub — même garde-fou, sans secret à gérer.

**Incident de sécurité évité** : un Deploy Hook Render (URL secrète) a été
collé en clair dans le chat par l'utilisateur → régénéré côté Render, jamais
utilisé, l'option native lui a été préférée.

---

## 5. Phase 3 — Données

- `src/data/load.py` : téléchargement du CSV UCI (l'URL `.zip` historique
  renvoyait 404 après une refonte du site UCI — corrigée vers l'endpoint
  `data.csv` actuel).
- `notebooks/01_eda.ipynb` : EDA exécutée et commitée avec ses 7 figures.
- `src/data/preprocess.py` : nettoyage complet.
- `src/data/split.py` : split 60/20/20 stratifié, sauvegardé, **jamais
  recalculé** sans `--force`.

### Décisions méthodologiques clés

- **Règle temporelle (la plus structurante du projet)** : ne garder que les
  variables connues **à l'admission**. Écartées : `discharge_disposition_id`,
  `readmitted` (postérieures à la sortie), `num_medications`,
  `num_lab_procedures`, `num_procedures`, les 21 colonnes de médicaments
  (mesurées **pendant** le séjour — elles reflètent la durée plus qu'elles ne
  la prédisent). **Coût assumé : 49 → 20 colonnes.** Un test verrouille cette
  règle et casse la CI si une variable postérieure est réintroduite.
- **Une seule admission par patient** (première occurrence) : requis par
  l'hypothèse d'échangeabilité du conformal. Coût : **29,7 % des lignes**
  écartées — le plus lourd du pipeline, non négociable.
- **Exclusion des décès / soins palliatifs** (2,4 % des lignes) : leur durée
  est déterminée par le décès, pas par la guérison.
- **`payer_code` conservé**, contre ma proposition initiale de le retirer.
  L'utilisateur a tranché : le retirer ne supprime pas l'inégalité d'accès aux
  soins, cela la rend invisible (le modèle la reconstruit via des proxys). On
  la garde et on mesure la couverture par catégorie d'assurance.
- Regroupement ICD-9 (~800 codes → 9 catégories cliniques), avec gestion des
  1 645 codes V/E non numériques et priorité du diabète sur la plage
  endocrinienne générale.
- `max_glu_serum` / `A1Cresult` traités comme `"not_measured"`, pas comme des
  manquants — l'absence de test est une décision clinique.

**Bug d'environnement** : une politique Windows (Smart App Control) bloquait
le chargement de la DLL `pyarrow`, empêchant tout import de pandas en local.
Contournement : `pyarrow` désinstallé du venv local (sans impact sur la CI
Linux ni sur la prod).

---

## 6. Phase 4 — Modélisation + conformal

- `src/models/train.py` : baseline LightGBM L2 vs modèle naïf (médiane), en
  validation croisée **sur le train uniquement**.
- `src/models/calibrate.py` : deux LightGBM quantiles (α=0,05 / 0,95) +
  `MapieQuantileRegressor` (méthode CQR), calibré sur le set de calibration.
- `src/models/conformal.py` : `ConformalPredictor`, objet de service **conçu
  spécifiquement** parce que MAPIE fige α à l'entraînement — or la SPEC exige
  un curseur de confiance variable. Recalcule la marge conforme pour tout α à
  partir des scores de conformité stockés (validé à 5e-4 près contre MAPIE).
- `src/models/benchmark.py` : comparaison **LightGBM vs XGBoost vs Random
  Forest vs réseau de neurones (MLP)**, protocole identique (CV 5 folds sur
  le train).

### Résultats baseline (CV train)

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Naïf (médiane) | 2,220 | 3,199 | −0,189 |
| **LightGBM** | **2,057** | **2,691** | **+0,159** |
| XGBoost | 2,060 | 2,697 | +0,155 |
| MLP (64,32) | 2,083 | 2,717 | +0,142 |
| Random Forest | 2,131 | 2,754 | +0,119 |

L'écart LightGBM/XGBoost est du bruit (0,0035 de R²). L'argument décisif :
`objective='quantile'` natif (indispensable à la CQR), support catégoriel
natif (19 colonnes vs 175 après one-hot pour MLP/RF), 5× plus rapide que le
MLP.

### Deux bugs critiques trouvés et corrigés

1. **Croisement de quantiles / incohérence point-intervalle.** MAPIE émettait
   un avertissement "predictions are ill-sorted". Investigation : 57,9 % des
   bornes basses tombaient sous 1 jour (minimum physiologique), et 0,3 % des
   estimations ponctuelles sortaient de leur propre intervalle. Corrigé par
   troncature à [1, 14] (prouvé : ne peut jamais faire **baisser** la
   couverture) + recentrage du point dans ses bornes.
2. **Catégories entières vs chaînes (le plus dangereux du projet).**
   `admission_type_id` / `admission_source_id` avaient des catégories
   entières. LightGBM mémorise les *valeurs* de catégories pour le remapping
   à la prédiction : l'entier `1` ne correspondait pas à la chaîne `"1"` reçue
   en JSON par l'API → **prédictions silencieusement fausses** (4,45 j au lieu
   de 4,77 j), **sans aucune erreur levée**. Corrigé à la source :
   `astype(str)` avant `astype("category")` dans `preprocess.py`.

---

## 7. Phase 5 — Validation statistique (le cœur du projet)

`src/models/evaluate.py` produit 5 analyses et figures sur le **test set**
(jamais touché avant cette phase) :

| Résultat | Valeur |
|---|---|
| Couverture visée | 90,00 % |
| **Couverture mesurée** | **90,21 %** (+0,84 erreur type — dans le bruit) |
| Largeur moyenne | 8,08 jours |
| Patients de test | 13 998 |
| Apport de la calibration | 87,36 % (brut) → 90,21 % (calibré), pour +0,02 j |
| Écart couverture vs α (11 niveaux, 50–99 %) | 0,0024 en moyenne |
| Couverture par sous-groupe | 86,36 % à 92,70 % |

**Limite théorique importante, mesurée et documentée** : la prédiction
conforme split ne garantit qu'une couverture **marginale**. La variation par
sous-groupe (6,3 points d'écart) en est la preuve empirique — ce n'est pas un
défaut d'implémentation.

---

## 8. Phase 6 — API réelle

- `src/api/schemas.py` : `PatientInput` à **16 champs saisis**, tous connus à
  l'admission. Les 3 features dérivées (`n_prior_visits`,
  `has_prior_inpatient`, `diagnoses_per_prior_visit`) sont des
  `computed_field` Pydantic — calculées, jamais saisies (évite les
  incohérences total/composantes).
- `src/api/service.py` : logique métier séparée des routes HTTP, testable sans
  serveur.
- `src/api/main.py` : `/health`, `/model-info` (couverture **réellement
  mesurée**, jamais la valeur visée), `/categories`, `/predict`,
  `/predict/batch`. Chargement du modèle via `lifespan` (une fois, pas par
  requête).

**Bug évité avant qu'il n'atteigne la prod** : `service.py` importait des
constantes depuis `calibrate.py` (dépend de `mapie`) et `evaluate.py` (dépend
de `matplotlib`) — deux dépendances de **développement**, absentes de
`requirements.txt`. L'API aurait planté au démarrage sur Render. Corrigé en
centralisant les chemins d'artefacts dans `config.py`, sans dépendance lourde.
Conséquence : `mapie` et `matplotlib` sortent définitivement de l'image de
production (allégement + robustesse).

---

## 9. Phase 7 — Tests

**99 tests** au total, dont les propriétés métier de la SPEC :

- `lower ≤ point ≤ upper` (réellement violée à 0,3 % avant correction).
- Largeur strictement positive, bornes dans [1, 14].
- **α plus petit ⇒ intervalle plus large.**
- Déterminisme, 15 cas de validation 422, contrat batch = unitaire.
- Verrou anti-fuite temporelle (28 colonnes surveillées).

---

## 10. Phase 8 — Front Streamlit

3 onglets : **Prédiction** (formulaire 16 champs + curseur de confiance
50–99 % qui élargit l'intervalle en direct — moment fort de la démo),
**Performance** (les 5 figures de la phase 5, servies statiquement, aucune
dépendance matplotlib côté prod), **À propos** (conformal vulgarisé + limites
connues). Graphique de l'intervalle en HTML/CSS pur (pas de matplotlib en
prod). Aucun chargement de modèle côté front — uniquement des appels HTTP.

---

## 11. Phase 9 — Docker multi-stage, infra AWS, README

- **Dockerfile multi-stage** : étage `builder` jeté après installation,
  utilisateur non-root, `HEALTHCHECK` sur `/health`.
- **`infra/`** : Terraform (ECR + App Runner) et `Dockerfile.aws` documentés,
  **jamais appliqués** — le free tier AWS post-2025 facture après 200 $ de
  crédits, incompatible avec la contrainte 0 €.
- **README complet** : résultats chiffrés, justification du choix de modèle,
  4 décisions méthodologiques expliquées, limites connues.

### Trois bugs Docker en cascade (débogage long, instructif)

1. **`pip install --user` dépend de `HOME`**, que l'instruction `USER` de
   Docker ne redéfinit pas de façon fiable → import échouait au runtime.
   Remplacé par un **venv à chemin absolu** (`/opt/venv`), indépendant de
   l'utilisateur.
2. **`HOME` toujours nécessaire** même avec le venv : Streamlit écrit sa
   config dans `~/.streamlit`, et sans `HOME` explicite il pointait vers
   `/root`, non accessible en écriture par l'utilisateur non-root.
3. **`libgomp1` manquant.** `python:3.11-slim` n'embarque pas la bibliothèque
   OpenMP dont LightGBM a besoin. L'image se construisait sans erreur mais
   plantait au **chargement du modèle** :
   `OSError: libgomp.so.1: cannot open shared object file`. Invisible en
   Phase 1 (le `/predict` était factice, aucun modèle chargé). Corrigé par
   `apt-get install libgomp1`.

### Débogage CI sans accès aux logs

Les logs de job GitHub Actions exigent des droits admin sur le repo
(inaccessibles ici). Diagnostic réalisé via :
- Les **annotations** (`::error::`), lisibles par l'API publique.
- Une trace complète du script (`set -x` + `tee` vers un fichier), restituée
  en annotation — a permis de localiser précisément la ligne fautive sans
  jamais voir les vrais logs du runner.

Piège shell rencontré : dans un bloc `run:` de GitHub Actions (`set -e`
implicite), une chaîne `[ condition ] && commande` dont le test est **faux**
renvoie 1 et **tue le script**, même quand la condition faisait exprès de ne
pas se déclencher. Remplacé par des `if` explicites partout.

Dernier bug, résolu en tout dernier : le test CI vérifiait `/health` par
égalité stricte sur l'ancienne réponse de la Phase 1 (`{"status":"ok"}`), alors
que le schéma s'était enrichi en Phase 6 (`model_loaded` ajouté) — le test
vérifiait une API obsolète, pas l'application réelle.

---

## 12. État final

| | |
|---|---|
| Commits | 30 |
| Tests | 99 (+ propriétés métier, verrous anti-fuite, anti-régression) |
| CI | 3 jobs verts (lint, test, docker) |
| Couverture conforme mesurée | **90,21 %** sur 13 998 patients de test |
| Déploiement | Render, `Auto-Deploy: After CI checks pass`, vérifié en ligne |
| Coût | 0 € |

**Reproduire** : voir [README.md](../README.md) section "Reproduire".
**Démo** : [medstay.onrender.com](https://medstay.onrender.com).
