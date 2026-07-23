"""Chemins, seed et hyperparamètres partagés par tout le pipeline."""

from pathlib import Path

SEED = 42

# Couverture cible de l'intervalle de prédiction (CQR), voir SPEC §4.
ALPHA = 0.10

# Split unique, non recalculé (voir CLAUDE.md contrainte 2).
TRAIN_RATIO = 0.60
CALIBRATION_RATIO = 0.20
TEST_RATIO = 0.20

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Chemins des artefacts, definis ICI et non dans les modules qui les produisent.
# Raison : src/models/evaluate.py importe matplotlib, absent de requirements.txt
# (dependance de dev). Si l'API importait ses constantes depuis evaluate.py,
# elle planterait au demarrage sur Render. Les constantes partagees vivent donc
# dans ce module, qui n'a aucune dependance lourde.
MODEL_PATH = MODELS_DIR / "cqr_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation.json"

# Quantiles des deux modeles encadrants. Choisis pour couvrir 1 - ALPHA = 90 %
# AVANT calibration : MAPIE corrigera ensuite ces bornes pour garantir la
# couverture reelle (les quantiles bruts d'un modele sont presque toujours
# trop etroits, c'est precisement ce que le conformal repare).
QUANTILE_LOW = 0.05
QUANTILE_HIGH = 0.95

# Nombre de folds pour la validation croisee sur le TRAIN. Les metriques de
# developpement se mesurent la : le set de calibration est reserve a MAPIE et
# le test a l'evaluation finale (contrainte 3 de CLAUDE.md).
CV_FOLDS = 5

# Hyperparametres LightGBM, volontairement sobres : sans tuning agressif, le
# risque de surapprentissage sur le protocole reste faible. Un tuning eventuel
# devra se faire en validation croisee sur le TRAIN uniquement.
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
