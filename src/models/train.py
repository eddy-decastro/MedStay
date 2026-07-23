"""Baseline LightGBM L2 : reference de performance avant le conformal.

Ce modele n'est PAS le produit final. Il sert a repondre a une question :
"quelle precision peut-on esperer avec les seules variables connues a
l'admission ?" Les modeles quantiles et la calibration MAPIE viennent ensuite.

Regle de mesure (contrainte 3 de CLAUDE.md) : les metriques de developpement
sont obtenues par VALIDATION CROISEE SUR LE TRAIN. Le set de calibration est
reserve a MAPIE, le test a l'evaluation finale. Aucun des deux n'est touche ici.

Usage : python -m src.models.train
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from src.config import CV_FOLDS, LGBM_PARAMS, MODELS_DIR, SEED
from src.data.preprocess import TARGET
from src.data.split import load_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH = MODELS_DIR / "baseline_l2.joblib"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separe les features de la cible."""
    return df.drop(columns=[TARGET]), df[TARGET]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE et R2 -- les trois metriques exigees par la SPEC."""
    return {
        # MAE : erreur moyenne en JOURS, directement interpretable pour un
        # gestionnaire de lits. La metrique la plus parlante des trois.
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # RMSE : penalise davantage les grosses erreurs (sejours longs rates).
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        # R2 : part de variance expliquee. 0 = aussi bon que predire la moyenne.
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """Evalue un modele par validation croisee sur le train uniquement."""
    # KFold simple (pas de stratification) : la cible est numerique et la
    # validation croisee ne sert ici qu'a estimer l'erreur, pas a garantir
    # une couverture. Le shuffle est indispensable, les donnees pouvant etre
    # ordonnees par encounter_id.
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    scores: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": []}
    for train_idx, val_idx in kfold.split(X):
        # .iloc : on indexe par POSITION, les index du split n'etant pas
        # contigus apres la deduplication par patient.
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])

        for name, value in compute_metrics(y.iloc[val_idx], preds).items():
            scores[name].append(value)

    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in scores.items()
    }


def train_baseline() -> tuple[LGBMRegressor, dict]:
    """Entraine la baseline L2 et la compare a un modele naif."""
    splits = load_split()
    X_train, y_train = split_xy(splits["train"])
    logger.info("Train : %d lignes x %d features", *X_train.shape)

    # Reference naive : predire toujours la mediane. Sans ce point de
    # comparaison, un R2 de 0,05 est ininterpretable -- impossible de dire si le
    # modele est mauvais ou si le probleme est intrinsequement difficile.
    logger.info("Validation croisee : modele naif (mediane)...")
    naive_scores = cross_validate(DummyRegressor(strategy="median"), X_train, y_train)

    logger.info("Validation croisee : LightGBM L2 (%d folds)...", CV_FOLDS)
    model = LGBMRegressor(objective="regression", **LGBM_PARAMS)
    lgbm_scores = cross_validate(model, X_train, y_train)

    for name in ("mae", "rmse", "r2"):
        naif = naive_scores[name]["mean"]
        lgbm = lgbm_scores[name]["mean"]
        logger.info(
            "%-5s naif %7.4f | LightGBM %7.4f (+/- %.4f)",
            name.upper(),
            naif,
            lgbm,
            lgbm_scores[name]["std"],
        )

    # Modele final : reentraine sur la TOTALITE du train. La validation croisee
    # n'a servi qu'a estimer l'erreur, ses modeles intermediaires sont jetes.
    logger.info("Reentrainement sur l'integralite du train...")
    model.fit(X_train, y_train)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(X_train),
        "n_features": X_train.shape[1],
        "cv_folds": CV_FOLDS,
        "evaluated_on": "cross-validation sur le train (calibration et test intouches)",
        "naive_median": naive_scores,
        "lightgbm_l2": lgbm_scores,
        "params": LGBM_PARAMS,
    }
    return model, metrics


def main() -> None:
    """Entraine la baseline, la sauvegarde avec ses metriques."""
    model, metrics = train_baseline()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_PATH)
    BASELINE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Modele  : %s", BASELINE_PATH)
    logger.info("Metriques : %s", BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()
