"""Tests du preprocessing : determinisme, schema de sortie, gestion des manquants."""

import pandas as pd
import pytest

from src.data.preprocess import (
    DRUG_COLS,
    DURING_STAY_COLS,
    POST_DISCHARGE_COLS,
    TARGET,
    group_icd9,
    preprocess,
)


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Mini-dataset synthetique reproduisant les pieges du CSV UCI."""
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            # Le patient 100 apparait deux fois : une seule doit survivre.
            "patient_nbr": [100, 100, 200, 300, 400],
            "race": ["Caucasian", "Caucasian", "?", "AfricanAmerican", "Other"],
            "gender": ["Male", "Male", "Female", "Unknown/Invalid", "Female"],
            "age": ["[50-60)", "[50-60)", "[70-80)", "[60-70)", "[40-50)"],
            "weight": ["?", "?", "?", "?", "?"],
            "admission_type_id": [1, 1, 2, 3, 1],
            # 11 = deces : cette ligne doit etre exclue.
            "discharge_disposition_id": [1, 1, 11, 1, 1],
            "admission_source_id": [7, 7, 1, 7, 7],
            TARGET: [3, 5, 8, 2, 6],
            "payer_code": ["MC", "MC", "?", "BC", "?"],
            "medical_specialty": ["?", "?", "Surgery", "?", "Cardiology"],
            "num_lab_procedures": [40, 45, 60, 20, 35],
            "num_procedures": [1, 2, 3, 0, 2],
            "num_medications": [10, 12, 20, 5, 15],
            "number_outpatient": [0, 1, 2, 0, 0],
            "number_emergency": [0, 0, 1, 0, 0],
            "number_inpatient": [0, 1, 3, 0, 1],
            # V57 = code non numerique, doit tomber dans "Other" sans planter.
            "diag_1": ["250.83", "428", "V57", "?", "715"],
            "diag_2": ["401", "401", "486", "530", "250.01"],
            "diag_3": ["?", "272", "E884", "?", "401"],
            "number_diagnoses": [5, 6, 9, 3, 7],
            "max_glu_serum": ["None", "None", ">200", "?", "Norm"],
            "A1Cresult": ["?", ">7", "None", "?", "Norm"],
            **{
                drug: ["No", "Up", "Steady", "No", "Down"]
                for drug in [
                    "metformin",
                    "repaglinide",
                    "nateglinide",
                    "chlorpropamide",
                    "glimepiride",
                    "acetohexamide",
                    "glipizide",
                    "glyburide",
                    "tolbutamide",
                    "pioglitazone",
                    "rosiglitazone",
                    "acarbose",
                    "miglitol",
                    "troglitazone",
                    "tolazamide",
                    "insulin",
                    "glyburide-metformin",
                    "glipizide-metformin",
                    "glimepiride-pioglitazone",
                    "metformin-rosiglitazone",
                    "metformin-pioglitazone",
                ]
            },
            "examide": ["No"] * 5,
            "citoglipton": ["No"] * 5,
            "change": ["No", "Ch", "Ch", "No", "Ch"],
            "diabetesMed": ["Yes", "Yes", "Yes", "No", "Yes"],
            "readmitted": ["NO", ">30", "<30", "NO", "NO"],
        }
    )


# --- Regroupement ICD-9 -----------------------------------------------------


@pytest.mark.parametrize(
    ("code", "attendu"),
    [
        ("250.83", "Diabetes"),  # diabete avec complications
        ("250", "Diabetes"),
        ("428", "Circulatory"),  # insuffisance cardiaque
        ("786", "Respiratory"),  # symptomes respiratoires (hors plage 460-519)
        ("530", "Digestive"),
        ("585", "Genitourinary"),
        ("820", "Injury"),
        ("715", "Musculoskeletal"),
        ("199", "Neoplasms"),
        ("276", "Other"),  # endocrinien non diabetique
    ],
)
def test_group_icd9_categories_cliniques(code, attendu):
    assert group_icd9(code) == attendu


@pytest.mark.parametrize("code", ["V57", "E884", "V45", "E code invalide"])
def test_group_icd9_codes_non_numeriques(code):
    """Les codes V/E ne sont pas convertibles en nombre : ils ne doivent pas planter."""
    assert group_icd9(code) == "Other"


def test_group_icd9_manquant():
    assert group_icd9(None) == "Missing"
    assert group_icd9(pd.NA) == "Missing"


def test_group_icd9_diabete_prioritaire_sur_endocrinien():
    """250.x tombe dans la plage endocrinienne : la regle Diabetes doit primer."""
    assert group_icd9("250.01") == "Diabetes"
    assert group_icd9("249") != "Diabetes"


# --- Preprocessing complet --------------------------------------------------


def test_preprocess_est_deterministe(raw_sample):
    """Deux executions sur la meme entree donnent un resultat identique."""
    a = preprocess(raw_sample.copy())
    b = preprocess(raw_sample.copy())
    pd.testing.assert_frame_equal(a, b)


def test_preprocess_ne_laisse_aucun_manquant(raw_sample):
    assert preprocess(raw_sample.copy()).isna().sum().sum() == 0


def test_preprocess_garde_une_seule_admission_par_patient(raw_sample):
    """Le patient 100 a deux sejours : un seul doit subsister (echangeabilite)."""
    out = preprocess(raw_sample.copy())
    # patient_nbr est supprime en sortie : on verifie via l'effectif attendu.
    # 5 lignes - 1 deces - 1 genre invalide - 1 doublon = 2.
    assert len(out) == 2


def test_preprocess_exclut_les_deces(raw_sample):
    """Le seul sejour termine par un deces (id 11) ne doit pas survivre."""
    out = preprocess(raw_sample.copy())
    # discharge_disposition_id sert au filtrage puis est supprime (inconnu a
    # l'admission) : on verifie l'exclusion par la duree propre a cette ligne.
    assert 8 not in out[TARGET].to_numpy()


def test_preprocess_ne_garde_aucune_variable_posterieure_a_l_admission(raw_sample):
    """Verrou anti-fuite : le modele doit etre utilisable DES l'admission.

    Ce test echoue si quelqu'un reintroduit une variable renseignee pendant ou
    apres le sejour. Il protege le cas d'usage annonce dans la SPEC (anticiper
    l'occupation des lits au moment ou le patient arrive).
    """
    out = preprocess(raw_sample.copy())
    interdites = set(POST_DISCHARGE_COLS) | set(DURING_STAY_COLS) | set(DRUG_COLS)
    fuites = interdites & set(out.columns)
    assert not fuites, f"Variables non disponibles a l'admission : {sorted(fuites)}"


def test_preprocess_supprime_identifiants_et_colonnes_inutiles(raw_sample):
    out = preprocess(raw_sample.copy())
    for col in ["encounter_id", "patient_nbr", "weight", "examide", "citoglipton"]:
        assert col not in out.columns


def test_preprocess_cree_les_features_derivees(raw_sample):
    out = preprocess(raw_sample.copy())
    for col in ["n_prior_visits", "has_prior_inpatient", "diagnoses_per_prior_visit"]:
        assert col in out.columns
    # Bornes de coherence : ces compteurs ne peuvent pas etre negatifs.
    assert (out["n_prior_visits"] >= 0).all()
    assert (out["diagnoses_per_prior_visit"] >= 0).all()
    assert out["has_prior_inpatient"].isin([0, 1]).all()


def test_preprocess_encode_en_categoriel_pour_lightgbm(raw_sample):
    """Aucune colonne texte ne doit subsister : LightGBM exige du categoriel."""
    out = preprocess(raw_sample.copy())
    assert out.select_dtypes(include="object").empty


def test_preprocess_conserve_la_cible_intacte(raw_sample):
    out = preprocess(raw_sample.copy())
    assert TARGET in out.columns
    assert out[TARGET].between(1, 14).all()
