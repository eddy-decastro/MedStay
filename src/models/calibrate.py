"""Calibration conforme des intervalles par CQR (Conformalized Quantile Regression).

C'est le coeur du projet. Deux etapes distinctes :

  1. REGRESSION QUANTILE -- deux LightGBM apprennent les quantiles 0,05 et 0,95
     de la duree de sejour. Leurs bornes sont deja adaptatives (plus larges pour
     les cas complexes), mais elles n'offrent AUCUNE garantie : un modele
     quantile sous-couvre presque toujours, parce qu'il est ajuste sur les
     donnees d'entrainement.

  2. CALIBRATION CONFORME -- sur un jeu JAMAIS VU a l'entrainement (le set de
     calibration), MAPIE mesure de combien les bornes se trompent, puis les
     corrige d'une marge unique. Le resultat porte une garantie de couverture
     a 1 - alpha, valable en distribution finie et sans hypothese sur la loi
     des donnees -- a la seule condition que les observations soient
     echangeables. C'est pour cela que la deduplication par patient
     (preprocess.py) n'etait pas negociable.

MAPIE entraine trois estimateurs : alpha/2, 1 - alpha/2 et la mediane. Avec
alpha = 0,10, cela donne les quantiles 0,05 / 0,95 fixes dans config.py.

Usage : python -m src.models.calibrate
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from mapie.regression import MapieQuantileRegressor

from src.config import (
    ALPHA,
    LGBM_PARAMS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    QUANTILE_HIGH,
    QUANTILE_LOW,
)
from src.data.split import load_split
from src.models.conformal import ConformalPredictor, FeatureSpec
from src.models.train import split_xy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Chemins definis dans config.py : l'API doit pouvoir les importer sans
# tirer mapie ni matplotlib (absents de l'image de production).


# Bornes physiologiques de la cible dans ce dataset : un sejour dure de 1 a
# 14 jours par construction (au-dela, le dataset agrege a 14).
TARGET_MIN = 1.0
TARGET_MAX = 14.0


def postprocess(
    point: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rend les predictions coherentes : bornes dans [1, 14] et point dedans.

    Corrige deux defauts observes sur les sorties brutes de MAPIE :

    1. 57,9 % des bornes basses tombaient sous 1 jour et 0,16 % des bornes
       hautes depassaient 14, alors que la cible est bornee par construction.
       Tronquer a [1, 14] NE PEUT JAMAIS FAIRE BAISSER LA COUVERTURE : la vraie
       valeur etant toujours dans [1, 14], si elle etait dans l'intervalle
       d'origine elle reste dans l'intervalle tronque. La garantie conforme est
       donc preservee, avec des intervalles plus serres.

       Cas limite : un intervalle situe ENTIEREMENT hors de [1, 14] (ex.
       [-3 ; 0,5]) devient degenere sur la borne ([1 ; 1]) et peut se mettre a
       couvrir y = 1. La couverture peut donc augmenter. La propriete exacte
       est une inegalite, pas une egalite -- meme si sur ce modele l'egalite
       est observee (0,928294 avant et apres), aucun intervalle ne tombant
       entierement hors des bornes.

    2. 0,3 % des estimations ponctuelles tombaient HORS de leur propre
       intervalle. Les trois quantiles sont appris independamment et la
       correction conforme decale les bornes sans decaler la mediane : rien ne
       garantit leur ordre. Annoncer "4,2 jours, intervalle [4,5 ; 9]" serait
       incoherent pour l'utilisateur. On ramene donc le point dans ses bornes.
    """
    low = np.clip(low, TARGET_MIN, TARGET_MAX)
    high = np.clip(high, TARGET_MIN, TARGET_MAX)

    # Garde-fou : si les deux quantiles se croisent malgre tout, on retablit
    # l'ordre plutot que de renvoyer un intervalle vide.
    low, high = np.minimum(low, high), np.maximum(low, high)

    point = np.clip(point, low, high)
    return point, low, high


def predict_intervals(
    model: MapieQuantileRegressor, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predit (point, borne basse, borne haute) deja post-traitees."""
    point, intervals = model.predict(X)
    return postprocess(point, intervals[:, 0, 0], intervals[:, 1, 0])


def build_quantile_estimator() -> LGBMRegressor:
    """Construit le regresseur quantile que MAPIE clonera pour chaque borne."""
    # objective="quantile" : LightGBM optimise la pinball loss et non la MSE.
    # C'est ce qui lui fait estimer un QUANTILE conditionnel plutot qu'une
    # moyenne conditionnelle.
    #
    # alpha=0,5 n'est qu'une valeur d'amorce : MAPIE la remplace par 0,05,
    # 0,95 et 0,5 sur ses trois clones. Elle doit toutefois EXISTER, sans quoi
    # MAPIE refuse l'estimateur ("matching parameter alpha_name does not exist").
    return LGBMRegressor(objective="quantile", alpha=0.5, **LGBM_PARAMS)


def build_feature_spec(X: pd.DataFrame) -> FeatureSpec:
    """Fige le schema d'entrainement pour que l'API reconstruise le bon DataFrame.

    Sans cela, un DataFrame construit depuis du JSON n'aurait ni le bon ordre de
    colonnes, ni les memes categories : LightGBM produirait des predictions
    silencieusement fausses, sans lever la moindre erreur.
    """
    categorielles = X.select_dtypes(include="category").columns
    return FeatureSpec(
        columns=list(X.columns),
        categories={
            col: [str(c) for c in X[col].cat.categories] for col in categorielles
        },
        numeric=list(X.select_dtypes(exclude="category").columns),
    )


def calibrate() -> tuple[MapieQuantileRegressor, dict]:
    """Entraine les modeles quantiles et les calibre par CQR."""
    splits = load_split()
    X_train, y_train = split_xy(splits["train"])
    X_calib, y_calib = split_xy(splits["calibration"])

    logger.info("Train       : %d lignes (apprentissage des quantiles)", len(X_train))
    logger.info("Calibration : %d lignes (MAPIE uniquement)", len(X_calib))

    model = MapieQuantileRegressor(
        build_quantile_estimator(),
        # "split" : l'unique methode disponible pour la CQR. Elle exige des jeux
        # d'entrainement et de calibration disjoints -- ce que notre split
        # garantit deja explicitement.
        cv="split",
        alpha=ALPHA,
    )

    # On passe NOTRE set de calibration plutot que de laisser MAPIE en decouper
    # un (calib_size=0.3 par defaut). Sans cela il taillerait dans le train et
    # notre split soigneusement stratifie ne servirait a rien.
    logger.info("Entrainement des 3 quantiles + calibration conforme...")
    model.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)

    quantiles_appris = [float(est.alpha) for est in model.estimators_]
    logger.info("Quantiles appris : %s", quantiles_appris)

    # Verification de sante sur le TRAIN. Ce chiffre sera optimiste (le modele
    # connait ces donnees) : il ne sert qu'a detecter une anomalie grossiere.
    # La vraie couverture se mesure sur le TEST, en phase 5.
    _, low, high = predict_intervals(model, X_train)
    couverture_train = float(((y_train >= low) & (y_train <= high)).mean())
    largeur_train = float(np.mean(high - low))
    logger.info(
        "Sante (train, optimiste) : couverture %.4f | largeur moyenne %.2f j",
        couverture_train,
        largeur_train,
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "CQR (Conformalized Quantile Regression)",
        "alpha": ALPHA,
        "target_coverage": round(1 - ALPHA, 4),
        "quantiles": {"low": QUANTILE_LOW, "high": QUANTILE_HIGH},
        "learned_quantiles": quantiles_appris,
        "n_train": len(X_train),
        "n_calibration": len(X_calib),
        "n_features": X_train.shape[1],
        "features": sorted(X_train.columns),
        # Sanity check, PAS la metrique du projet : la couverture officielle
        # est mesuree sur le test en phase 5.
        "sanity_train_coverage": round(couverture_train, 4),
        "sanity_train_mean_width": round(largeur_train, 4),
        "lgbm_params": LGBM_PARAMS,
        "versions": {
            "lightgbm": lightgbm.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return model, metadata


def build_predictor(
    model: MapieQuantileRegressor, X_train: pd.DataFrame, metadata: dict
) -> ConformalPredictor:
    """Extrait de MAPIE un predicteur autonome a niveau de confiance libre.

    On ne deploie pas l'objet MAPIE tel quel : il fige alpha a l'entrainement,
    alors que la SPEC demande un curseur de confiance. ConformalPredictor
    conserve les trois modeles quantiles et les scores de conformite, ce qui
    suffit a recalculer la marge pour n'importe quel alpha (voir conformal.py).
    """
    # estimators_ : [0] quantile bas, [1] quantile haut, [2] mediane.
    # conformity_scores_ : ligne 2 = max(bas, haut), le score CQR.
    scores = np.asarray(model.conformity_scores_)[2]

    return ConformalPredictor(
        model_low=model.estimators_[0],
        model_high=model.estimators_[1],
        model_median=model.estimators_[2],
        conformity_scores=scores,
        feature_spec=build_feature_spec(X_train),
        metadata=metadata,
    )


def main() -> None:
    """Calibre le modele et exporte l'artefact deploye par l'API."""
    model, metadata = calibrate()

    X_train, _ = split_xy(load_split()["train"])
    predictor = build_predictor(model, X_train, metadata)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, MODEL_PATH)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    taille_mo = MODEL_PATH.stat().st_size / 1e6
    logger.info("Predicteur conforme : %s (%.2f Mo)", MODEL_PATH, taille_mo)
    logger.info("Metadonnees : %s", MODEL_METADATA_PATH)


if __name__ == "__main__":
    main()
