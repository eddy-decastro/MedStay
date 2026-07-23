"""API FastAPI : prediction de duree de sejour avec intervalle garanti a 90 %.

Routes (SPEC section 6) :
  GET  /health        etat du service
  GET  /model-info    carte d'identite du modele et couverture mesuree
  GET  /categories    modalites acceptees (le front y peuple ses menus)
  POST /predict       un patient
  POST /predict/batch plusieurs patients

Ce module ne fait que du protocole HTTP. La logique vit dans service.py, ce qui
la rend testable sans serveur.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthOutput,
    ModelInfoOutput,
    PatientInput,
    PredictionOutput,
)
from src.api.service import service
from src.models.conformal import ALPHA_MAX, ALPHA_MIN

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele au demarrage, une seule fois pour tout le processus.

    Le charger a chaque requete couterait ~200 ms et saturerait les 512 Mo de
    RAM du service gratuit Render.
    """
    logger.info("Demarrage : chargement du modele...")
    service.load()
    logger.info("Pret.")
    yield
    # Rien a liberer : joblib ne detient aucune ressource systeme.
    logger.info("Arret.")


app = FastAPI(
    title="MedStay-CI",
    version="1.0.0",
    description=(
        "Prediction de duree de sejour hospitalier avec intervalles garantis "
        "par Conformalized Quantile Regression. Couverture validee "
        "empiriquement sur 13 998 patients de test."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput, tags=["Service"])
def health() -> HealthOutput:
    """Sonde de vie. Doit rester triviale : aucun calcul, aucun acces disque."""
    return HealthOutput(status="ok", model_loaded=service.is_loaded)


@app.get("/model-info", response_model=ModelInfoOutput, tags=["Service"])
def model_info() -> ModelInfoOutput:
    """Carte d'identite du modele, dont la couverture REELLEMENT mesuree."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return ModelInfoOutput(**service.model_info())


@app.get("/categories", tags=["Service"])
def categories() -> dict[str, list[str]]:
    """Modalites connues du modele.

    Expose surtout medical_specialty et payer_code, trop nombreuses pour tenir
    dans une enumeration Pydantic lisible : le front peuple ses menus ici.
    """
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return service.categories()


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(
    patient: PatientInput,
    alpha: float = Query(
        default=0.10,
        ge=ALPHA_MIN,
        le=ALPHA_MAX,
        description="Risque accepte ; la couverture visee vaut 1 - alpha",
    ),
) -> PredictionOutput:
    """Predit la duree de sejour d'un patient, avec son intervalle."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    debut = time.perf_counter()
    resultat = service.predict([patient], alpha=alpha)[0]
    latence_ms = (time.perf_counter() - debut) * 1000

    # Journalisation SANS donnee identifiante : ni age, ni diagnostic, ni
    # assurance. Uniquement des indicateurs de fonctionnement.
    logger.info(
        "predict alpha=%.2f largeur=%.2f latence=%.1fms",
        alpha,
        resultat.interval_width,
        latence_ms,
    )
    return resultat


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch(payload: BatchPredictionInput) -> BatchPredictionOutput:
    """Predit pour une liste de patients (1000 maximum)."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    debut = time.perf_counter()
    predictions = service.predict(payload.patients, alpha=payload.alpha)
    latence_ms = (time.perf_counter() - debut) * 1000

    logger.info(
        "predict_batch n=%d alpha=%.2f latence=%.1fms (%.2fms/patient)",
        len(predictions),
        payload.alpha,
        latence_ms,
        latence_ms / len(predictions),
    )
    return BatchPredictionOutput(predictions=predictions, count=len(predictions))
