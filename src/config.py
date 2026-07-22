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
