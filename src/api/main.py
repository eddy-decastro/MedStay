"""API FastAPI — squelette déployable (Phase 1).

/predict renvoie un intervalle CODE EN DUR : aucun modèle chargé pour l'instant.
But de cette phase : valider la chaîne API -> Docker -> HF Space avant de brancher
la vraie logique (Phase 4/6). Le format JSON de sortie est déjà celui de la SPEC.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="MedStay-CI API", version="0.1.0")


class PatientInput(BaseModel):
    # Squelette minimal : les bornes physiologiques réelles (Field(ge=..., le=...))
    # arrivent en Phase 6 avec le vrai schéma PatientInput de la SPEC §6.
    age: str = Field(examples=["[50-60)"])
    time_in_hospital_hint: int | None = Field(
        default=None, description="Champ placeholder, ignoré par le predict factice."
    )


class PredictionOutput(BaseModel):
    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    coverage_level: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput, alpha: float = 0.10) -> PredictionOutput:
    # Valeurs factices : remplacées en Phase 4 par MapieQuantileRegressor.
    return PredictionOutput(
        point_estimate=4.2,
        lower_bound=3.0,
        upper_bound=5.5,
        interval_width=2.5,
        coverage_level=1 - alpha,
    )
