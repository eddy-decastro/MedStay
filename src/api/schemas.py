"""Schemas Pydantic v2 de l'API : validation stricte des entrees et sorties.

Principe : tout ce qui est invalide doit etre rejete par un 422 EXPLICITE
plutot que de produire une prediction silencieusement fausse. C'est la lecon
tiree du bug de categories entieres (voir conformal.py) : une erreur bruyante
vaut infiniment mieux qu'un resultat faux.

L'utilisateur saisit 16 champs. Les 3 features derivees
(n_prior_visits, has_prior_inpatient, diagnoses_per_prior_visit) sont calculees
par l'API : les demander exposerait a des incoherences entre un total et ses
composantes.
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from src.models.conformal import ALPHA_MAX, ALPHA_MIN

# --- Modalites autorisees ----------------------------------------------------
# Reprises telles quelles des categories vues a l'entrainement. Un test
# (test_api.py) verifie qu'elles restent alignees sur le modele : toute derive
# casse la CI plutot que de degrader silencieusement les predictions.

Race = Literal["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"]
Gender = Literal["Female", "Male"]
AgeBracket = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
# Categories cliniques issues du regroupement ICD-9 (voir preprocess.py).
DiagnosisGroup = Literal[
    "Circulatory",
    "Diabetes",
    "Digestive",
    "Genitourinary",
    "Injury",
    "Missing",
    "Musculoskeletal",
    "Neoplasms",
    "Other",
    "Respiratory",
]
# "not_measured" n'est pas une valeur manquante : c'est le fait que le medecin
# n'a pas juge utile de doser, ce qui est en soi un signal clinique.
GlucoseSerum = Literal[">200", ">300", "Norm", "not_measured"]
A1CResult = Literal[">7", ">8", "Norm", "not_measured"]


class PatientInput(BaseModel):
    """Un patient a l'admission. Aucun champ n'exige d'information posterieure."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "race": "Caucasian",
                    "gender": "Female",
                    "age": "[70-80)",
                    "admission_type_id": "1",
                    "admission_source_id": "7",
                    "payer_code": "MC",
                    "medical_specialty": "InternalMedicine",
                    "diag_1": "Circulatory",
                    "diag_2": "Diabetes",
                    "diag_3": "Circulatory",
                    "max_glu_serum": "not_measured",
                    "A1Cresult": ">8",
                    "number_outpatient": 0,
                    "number_emergency": 1,
                    "number_inpatient": 2,
                    "number_diagnoses": 9,
                }
            ]
        }
    }

    # --- Demographie ---
    race: Race = Field(description="Groupe declare ; 'Unknown' si non renseigne")
    gender: Gender
    age: AgeBracket = Field(description="Tranche d'age de 10 ans")

    # --- Circonstances de l'admission ---
    # Identifiants administratifs transmis en CHAINE : ce sont des categories,
    # pas des quantites. Une valeur inconnue du modele devient NaN, que LightGBM
    # traite nativement -- plutot qu'un decalage d'encodage silencieux.
    admission_type_id: str = Field(
        description="Type d'admission (1=urgence, 2=urgent, 3=programme...)"
    )
    admission_source_id: str = Field(
        description="Origine de l'admission (7=urgences, 1=adresse par un medecin...)"
    )
    payer_code: str = Field(
        default="Unknown", description="Code assurance ; 'Unknown' si absent"
    )
    medical_specialty: str = Field(
        default="Unknown",
        description="Specialite du medecin admetteur ; 'Unknown' si non renseignee",
    )

    # --- Diagnostics (categories cliniques larges) ---
    diag_1: DiagnosisGroup = Field(description="Diagnostic principal")
    diag_2: DiagnosisGroup = Field(
        default="Missing", description="Diagnostic secondaire"
    )
    diag_3: DiagnosisGroup = Field(
        default="Missing", description="Diagnostic tertiaire"
    )

    # --- Biologie ---
    max_glu_serum: GlucoseSerum = Field(default="not_measured")
    A1Cresult: A1CResult = Field(default="not_measured")

    # --- Historique de recours aux soins (bornes issues du train) ---
    # Les bornes hautes valent environ 1,5 fois le maximum observe : assez
    # larges pour ne pas rejeter un patient reel atypique, assez serrees pour
    # attraper une faute de saisie evidente.
    number_outpatient: int = Field(
        ge=0, le=50, description="Consultations externes sur l'annee ecoulee"
    )
    number_emergency: int = Field(
        ge=0, le=50, description="Passages aux urgences sur l'annee ecoulee"
    )
    number_inpatient: int = Field(
        ge=0, le=25, description="Hospitalisations sur l'annee ecoulee"
    )
    number_diagnoses: int = Field(
        ge=1, le=16, description="Nombre de diagnostics enregistres"
    )

    # --- Features derivees, calculees ici et non saisies ---

    @computed_field
    @property
    def n_prior_visits(self) -> int:
        """Total des recours aux soins anterieurs, toutes voies confondues."""
        return self.number_outpatient + self.number_emergency + self.number_inpatient

    @computed_field
    @property
    def has_prior_inpatient(self) -> int:
        """Marqueur de fragilite : au moins une hospitalisation dans l'annee."""
        return int(self.number_inpatient > 0)

    @computed_field
    @property
    def diagnoses_per_prior_visit(self) -> float:
        """Polypathologie rapportee au suivi anterieur (+1 evite la division par 0)."""
        return self.number_diagnoses / (self.n_prior_visits + 1)


class PredictionOutput(BaseModel):
    """Intervalle de prediction avec sa garantie de couverture."""

    point_estimate: float = Field(description="Duree la plus probable, en jours")
    lower_bound: float = Field(description="Borne basse de l'intervalle")
    upper_bound: float = Field(description="Borne haute de l'intervalle")
    interval_width: float = Field(description="Largeur de l'intervalle, en jours")
    coverage_level: float = Field(
        description="Couverture garantie (1 - alpha), validee empiriquement"
    )


class BatchPredictionInput(BaseModel):
    """Plusieurs patients en une requete."""

    patients: list[PatientInput] = Field(
        min_length=1,
        max_length=1000,
        description="Limite a 1000 : au-dela, le service gratuit Render sature",
    )
    alpha: float = Field(
        default=0.10,
        ge=ALPHA_MIN,
        le=ALPHA_MAX,
        description="Risque accepte ; la couverture visee vaut 1 - alpha",
    )


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    count: int


class HealthOutput(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfoOutput(BaseModel):
    """Carte d'identite du modele servi."""

    method: str
    trained_at: str
    target_coverage: float
    # Couverture REELLEMENT mesuree sur le test, pas la valeur visee.
    measured_test_coverage: float | None
    mean_interval_width: float | None
    n_train: int
    n_calibration: int
    n_features: int
    features: list[str]
    versions: dict[str, str]
