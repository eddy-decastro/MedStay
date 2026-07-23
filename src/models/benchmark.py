"""Comparaison de modeles -- experience, pas production.

Objectif : justifier PAR LA MESURE le choix de LightGBM inscrit dans la SPEC,
plutot que par un argument d'autorite. Le resultat alimente le README.

Cinq modeles, protocole identique a la baseline (validation croisee 5 folds sur
le TRAIN uniquement), sans quoi la comparaison ne voudrait rien dire :

  - naive_median   : reference plancher (predire toujours la mediane)
  - lightgbm       : boosting, categorielles NATIVES
  - xgboost        : boosting, categorielles natives (enable_categorical)
  - random_forest  : bagging, exige un ONE-HOT prealable
  - mlp            : reseau de neurones, exige one-hot ET normalisation

Aucun modele alternatif n'est deploye. xgboost vit dans requirements-dev.txt :
le garder hors de l'image Render evite ~200 Mo inutiles.

Usage : python -m src.models.benchmark
"""

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import LGBM_PARAMS, MODELS_DIR, SEED
from src.data.split import load_split
from src.models.train import cross_validate, split_xy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = MODELS_DIR / "benchmark.json"

# Modeles capables d'ingerer les dtypes "category" sans pretraitement.
CATEGORICAL_NATIVE = {"lightgbm", "xgboost"}


def build_onehot_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """One-hot des categorielles, avec normalisation optionnelle des numeriques.

    C'est le coeur de la demonstration : LightGBM et XGBoost avalent le
    DataFrame tel quel, la ou RandomForest et le MLP imposent cette etape.
    """
    categorielles = X.select_dtypes(include="category").columns.tolist()
    numeriques = X.select_dtypes(exclude="category").columns.tolist()

    # Normalisation utile au MLP (convergence de la descente de gradient),
    # inutile aux forets qui sont invariantes aux transformations monotones.
    transformateur_num = StandardScaler() if scale else "passthrough"

    return ColumnTransformer(
        [
            ("num", transformateur_num, numeriques),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorielles,
            ),
        ]
    )


def build_models(X: pd.DataFrame) -> dict:
    """Construit les cinq modeles compares."""
    return {
        # Reference plancher : sans elle, un R2 de 0,16 est ininterpretable.
        "naive_median": DummyRegressor(strategy="median"),
        # Le choix de la SPEC. Categorielles natives, aucun pretraitement.
        "lightgbm": LGBMRegressor(objective="regression", **LGBM_PARAMS),
        # Meme famille (boosting). enable_categorical + tree_method="hist"
        # activent le support natif des categorielles depuis XGBoost 1.6.
        "xgboost": XGBRegressor(
            n_estimators=LGBM_PARAMS["n_estimators"],
            learning_rate=LGBM_PARAMS["learning_rate"],
            max_depth=6,
            subsample=LGBM_PARAMS["subsample"],
            colsample_bytree=LGBM_PARAMS["colsample_bytree"],
            enable_categorical=True,
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        # Bagging plutot que boosting. Aucun support categoriel : one-hot
        # obligatoire, sans normalisation (les arbres n'en tirent rien).
        "random_forest": Pipeline(
            [
                ("prep", build_onehot_preprocessor(X, scale=False)),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # Reseau de neurones. Architecture modeste : sur 42 000 lignes et un
        # signal faible, plus profond surapprendrait sans gagner.
        "mlp": Pipeline(
            [
                ("prep", build_onehot_preprocessor(X, scale=True)),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=120,
                        early_stopping=True,
                        n_iter_no_change=10,
                        validation_fraction=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def run_benchmark() -> dict:
    """Compare les cinq modeles en validation croisee sur le train."""
    X, y = split_xy(load_split()["train"])
    n_cat = X.select_dtypes(include="category").shape[1]
    n_num = X.shape[1] - n_cat
    n_onehot = sum(X[c].nunique() for c in X.select_dtypes(include="category").columns)

    logger.info("Train : %d lignes x %d features", *X.shape)
    logger.info(
        "Sans support categoriel : %d colonnes -> %d apres one-hot (x%.1f)",
        X.shape[1],
        n_onehot + n_num,
        (n_onehot + n_num) / X.shape[1],
    )

    resultats = {}
    for nom, modele in build_models(X).items():
        logger.info("Validation croisee : %s...", nom)
        debut = time.perf_counter()
        scores = cross_validate(modele, X, y)
        duree = time.perf_counter() - debut

        scores["fit_seconds"] = round(duree, 1)
        scores["categorical_native"] = nom in CATEGORICAL_NATIVE
        resultats[nom] = scores

        logger.info(
            "   MAE %.4f | RMSE %.4f | R2 %+.4f | %.1fs",
            scores["mae"]["mean"],
            scores["rmse"]["mean"],
            scores["r2"]["mean"],
            duree,
        )

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "validation croisee 5 folds sur le train uniquement",
        "n_train": len(X),
        "n_features_native": X.shape[1],
        "n_features_after_onehot": n_onehot + n_num,
        "results": resultats,
    }


def main() -> None:
    benchmark = run_benchmark()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    logger.info("Resultats : %s", BENCHMARK_PATH)


if __name__ == "__main__":
    main()
