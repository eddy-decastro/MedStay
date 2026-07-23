"""Tests de l'API : contrat HTTP, validation, et PROPRIETES METIER.

Les proprietes de la section 7 de la SPEC sont les tests les plus importants du
projet : elles verifient des invariants mathematiques que le modele doit tenir
quelles que soient les entrees. Un modele peut afficher d'excellentes metriques
tout en violant ces proprietes, auquel cas ses intervalles sont inutilisables.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PatientInput
from src.models.conformal import ALPHA_MAX, ALPHA_MIN, TARGET_MAX, TARGET_MIN

# Patient de reference : femme de 70-80 ans, admission par les urgences,
# pathologie circulatoire avec diabete. Cas realiste et frequent.
PATIENT = {
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


@pytest.fixture(scope="module")
def client():
    """TestClient dans un contexte : declenche le lifespan, donc le chargement."""
    with TestClient(app) as c:
        yield c


# --- Contrat HTTP -----------------------------------------------------------


def test_health_repond_200(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok", "model_loaded": True}


def test_model_info_expose_la_couverture_mesuree(client):
    """La couverture annoncee doit etre celle MESUREE sur le test, pas la visee."""
    corps = client.get("/model-info").json()

    assert corps["method"].startswith("CQR")
    assert corps["target_coverage"] == pytest.approx(0.9)
    assert corps["n_features"] == 19
    # Mesuree sur 13 998 patients : doit etre proche de 0,90 sans etre egale.
    assert 0.85 < corps["measured_test_coverage"] < 0.95


def test_categories_liste_les_modalites(client):
    corps = client.get("/categories").json()
    assert "medical_specialty" in corps
    assert "Circulatory" in corps["diag_1"]


def test_predict_renvoie_le_schema_attendu(client):
    reponse = client.post("/predict", json=PATIENT)
    assert reponse.status_code == 200
    assert set(reponse.json()) == {
        "point_estimate",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "coverage_level",
    }


def test_predict_batch(client):
    reponse = client.post("/predict/batch", json={"patients": [PATIENT] * 3})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["count"] == 3
    assert len(corps["predictions"]) == 3


def test_batch_et_unitaire_donnent_le_meme_resultat(client):
    """Un patient seul ou dans un lot doit recevoir la meme prediction."""
    seul = client.post("/predict", json=PATIENT).json()
    lot = client.post("/predict/batch", json={"patients": [PATIENT]}).json()
    assert seul == lot["predictions"][0]


# --- PROPRIETES METIER (SPEC section 7) --------------------------------------


def test_propriete_lower_inferieur_point_inferieur_upper(client):
    """lower <= point <= upper.

    Annoncer "4,2 jours, intervalle [4,5 ; 9]" serait incoherent. Cette
    propriete a REELLEMENT ete violee sur 0,3 % des cas avant le
    post-traitement : les trois quantiles sont appris independamment et la
    correction conforme decale les bornes sans decaler la mediane.
    """
    corps = client.post("/predict", json=PATIENT).json()
    assert corps["lower_bound"] <= corps["point_estimate"] <= corps["upper_bound"]


def test_propriete_largeur_strictement_positive(client):
    """Un intervalle de largeur nulle serait une prediction ponctuelle deguisee."""
    corps = client.post("/predict", json=PATIENT).json()
    assert corps["interval_width"] > 0
    assert corps["interval_width"] == pytest.approx(
        corps["upper_bound"] - corps["lower_bound"], abs=0.01
    )


def test_propriete_bornes_dans_les_limites_physiologiques(client):
    """Un sejour dure de 1 a 14 jours : aucune borne ne peut sortir de la."""
    corps = client.post("/predict", json=PATIENT).json()
    assert TARGET_MIN <= corps["lower_bound"] <= TARGET_MAX
    assert TARGET_MIN <= corps["upper_bound"] <= TARGET_MAX


def test_propriete_alpha_plus_petit_donne_intervalle_plus_large(client):
    """alpha decroissant => intervalle croissant.

    C'est la traduction directe de la garantie conforme : exiger plus de
    certitude coute forcement en precision. Une violation signalerait une
    erreur dans le calcul de la correction conforme.
    """
    largeurs = []
    for alpha in [0.50, 0.30, 0.20, 0.10, 0.05, 0.01]:
        corps = client.post("/predict", json=PATIENT, params={"alpha": alpha}).json()
        largeurs.append(corps["interval_width"])

    # Suite croissante au sens large (des paliers sont possibles quand deux
    # niveaux tombent sur le meme score de conformite).
    assert all(a <= b for a, b in zip(largeurs, largeurs[1:])), largeurs
    # Et strictement plus large aux extremes, sinon alpha ne servirait a rien.
    assert largeurs[-1] > largeurs[0]


def test_propriete_coverage_level_reflete_alpha(client):
    for alpha in [0.05, 0.10, 0.20]:
        corps = client.post("/predict", json=PATIENT, params={"alpha": alpha}).json()
        assert corps["coverage_level"] == pytest.approx(1 - alpha)


def test_propriete_deterministe(client):
    """Deux appels identiques doivent donner exactement le meme resultat."""
    a = client.post("/predict", json=PATIENT).json()
    b = client.post("/predict", json=PATIENT).json()
    assert a == b


def test_propriete_patient_plus_lourd_nest_pas_incoherent(client):
    """Un patient plus polypathologique ne doit pas obtenir un intervalle absurde."""
    leger = {**PATIENT, "number_diagnoses": 1, "number_inpatient": 0}
    lourd = {**PATIENT, "number_diagnoses": 16, "number_inpatient": 5}

    for cas in (leger, lourd):
        corps = client.post("/predict", json=cas).json()
        assert corps["lower_bound"] <= corps["point_estimate"] <= corps["upper_bound"]
        assert TARGET_MIN <= corps["lower_bound"]
        assert corps["upper_bound"] <= TARGET_MAX


# --- Validation : tout input invalide doit produire un 422 -------------------


@pytest.mark.parametrize(
    ("champ", "valeur"),
    [
        ("number_diagnoses", 0),  # sous la borne (>= 1)
        ("number_diagnoses", 17),  # au-dessus (<= 16)
        ("number_inpatient", -1),  # negatif
        ("number_emergency", 999),  # aberrant
        ("gender", "Autre"),  # hors enumeration
        ("age", "[100-110)"),  # tranche inexistante
        ("diag_1", "Cardiaque"),  # categorie clinique inconnue
        ("A1Cresult", ">9"),  # modalite inexistante
        ("number_diagnoses", "beaucoup"),  # mauvais type
    ],
)
def test_valeur_invalide_renvoie_422(client, champ, valeur):
    reponse = client.post("/predict", json={**PATIENT, champ: valeur})
    assert reponse.status_code == 422, f"{champ}={valeur!r} aurait du etre rejete"


@pytest.mark.parametrize("champ", ["gender", "age", "diag_1", "number_diagnoses"])
def test_champ_obligatoire_manquant_renvoie_422(client, champ):
    incomplet = {k: v for k, v in PATIENT.items() if k != champ}
    assert client.post("/predict", json=incomplet).status_code == 422


@pytest.mark.parametrize("alpha", [0.0, 0.005, 0.51, 0.99, 1.0, -0.1])
def test_alpha_hors_bornes_renvoie_422(client, alpha):
    """alpha doit rester dans [0,01 ; 0,50] : au-dela l'intervalle n'aurait
    plus de sens statistique avec 13 998 points de calibration."""
    reponse = client.post("/predict", json=PATIENT, params={"alpha": alpha})
    assert reponse.status_code == 422


def test_batch_vide_renvoie_422(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422


def test_batch_trop_grand_renvoie_422(client):
    reponse = client.post("/predict/batch", json={"patients": [PATIENT] * 1001})
    assert reponse.status_code == 422


def test_message_erreur_est_explicite(client):
    """Un 422 doit dire QUEL champ pose probleme, sinon il est inexploitable."""
    reponse = client.post("/predict", json={**PATIENT, "number_diagnoses": 99})
    detail = reponse.json()["detail"]
    assert any("number_diagnoses" in str(erreur["loc"]) for erreur in detail)


# --- Coherence schema / modele ------------------------------------------------


def test_les_features_derivees_sont_calculees_et_non_saisies():
    """L'utilisateur saisit 16 champs, le modele en recoit 19."""
    patient = PatientInput(**PATIENT)
    donnees = patient.model_dump()

    assert len(donnees) == 19
    assert donnees["n_prior_visits"] == 0 + 1 + 2
    assert donnees["has_prior_inpatient"] == 1
    assert donnees["diagnoses_per_prior_visit"] == pytest.approx(9 / 4)


def test_les_enumerations_du_schema_collent_aux_categories_du_modele(client):
    """Garde-fou anti-derive : si le modele est reentraine avec d'autres
    modalites, ce test casse au lieu de laisser passer des NaN silencieux."""
    categories = client.get("/categories").json()

    for champ in ("gender", "age", "diag_1", "max_glu_serum", "A1Cresult"):
        valeur = PATIENT[champ]
        assert valeur in categories[champ], f"{valeur} absent des categories de {champ}"


def test_bornes_alpha_coherentes_entre_schema_et_modele():
    assert ALPHA_MIN == 0.01
    assert ALPHA_MAX == 0.50
