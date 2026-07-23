"""Couche de service : chargement du modele et logique de prediction.

Separee des routes HTTP volontairement. main.py ne s'occupe que du protocole
(codes de statut, validation, journalisation) ; toute la logique metier vit ici
et se teste sans serveur.
"""

import json
import logging
from pathlib import Path

import joblib

from src.api.schemas import PatientInput, PredictionOutput

# Constantes importees de config et NON des modules qui produisent ces fichiers :
# src/models/evaluate.py importe matplotlib et src/models/calibrate.py importe
# mapie, deux dependances absentes de l'image de production. Les importer ici
# ferait planter l'API au demarrage sur Render.
from src.config import EVALUATION_REPORT_PATH, MODEL_METADATA_PATH, MODEL_PATH
from src.models.conformal import ConformalPredictor

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsule le predicteur conforme et ses metadonnees."""

    def __init__(self) -> None:
        self._predictor: ConformalPredictor | None = None
        self._metadata: dict = {}
        self._evaluation: dict = {}

    # --- Cycle de vie ------------------------------------------------------

    def load(self, model_path: Path = MODEL_PATH) -> None:
        """Charge l'artefact. Appele UNE FOIS au demarrage, via le lifespan.

        Charger a chaque requete couterait ~200 ms et saturerait les 512 Mo de
        RAM du service gratuit Render.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele introuvable : {model_path}. "
                "Lancer : python -m src.models.calibrate"
            )

        self._predictor = joblib.load(model_path)
        logger.info(
            "Modele charge : %d features, %d points de calibration",
            len(self._predictor.feature_spec.columns),
            self._predictor.n_calibration,
        )

        self._metadata = self._read_json(MODEL_METADATA_PATH)
        # Rapport d'evaluation optionnel : l'API doit demarrer meme si la
        # validation statistique n'a pas encore ete lancee.
        self._evaluation = self._read_json(EVALUATION_REPORT_PATH)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            logger.warning("Fichier absent, ignore : %s", path)
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def predictor(self) -> ConformalPredictor:
        if self._predictor is None:
            raise RuntimeError("Modele non charge : appeler load() d'abord")
        return self._predictor

    # --- Prediction --------------------------------------------------------

    def predict(
        self, patients: list[PatientInput], alpha: float = 0.10
    ) -> list[PredictionOutput]:
        """Predit les intervalles pour un ou plusieurs patients."""
        # model_dump() inclut les computed_field : les 3 features derivees
        # arrivent donc automatiquement, sans recalcul manuel ici.
        records = [p.model_dump() for p in patients]

        # build_frame remet les colonnes dans l'ordre du train et restaure les
        # dtypes categoriels. Sans cette etape, LightGBM predirait n'importe
        # quoi sans lever d'erreur.
        X = self.predictor.feature_spec.build_frame(records)

        point, low, high = self.predictor.predict(X, alpha=alpha)

        return [
            PredictionOutput(
                # Arrondi au centieme : afficher 4,448372 jours donnerait une
                # illusion de precision que le modele n'a pas.
                point_estimate=round(float(p), 2),
                lower_bound=round(float(lo), 2),
                upper_bound=round(float(hi), 2),
                interval_width=round(float(hi - lo), 2),
                coverage_level=round(1 - alpha, 4),
            )
            for p, lo, hi in zip(point, low, high)
        ]

    # --- Informations ------------------------------------------------------

    def model_info(self) -> dict:
        """Carte d'identite du modele, incluant la couverture reellement mesuree."""
        global_ = self._evaluation.get("global", {})
        return {
            "method": self._metadata.get("method", "CQR"),
            "trained_at": self._metadata.get("trained_at", "inconnu"),
            "target_coverage": self._metadata.get("target_coverage", 0.9),
            # None tant que src.models.evaluate n'a pas tourne : on n'invente
            # jamais un chiffre de couverture.
            "measured_test_coverage": global_.get("empirical_coverage"),
            "mean_interval_width": global_.get("mean_width"),
            "n_train": self._metadata.get("n_train", 0),
            "n_calibration": self._metadata.get("n_calibration", 0),
            "n_features": self._metadata.get("n_features", 0),
            "features": self._metadata.get("features", []),
            "versions": self._metadata.get("versions", {}),
        }

    def categories(self) -> dict[str, list[str]]:
        """Modalites connues du modele, pour que le front puisse peupler ses menus."""
        return self.predictor.feature_spec.categories


# Instance unique partagee par les routes, remplie au demarrage par le lifespan.
service = ModelService()
