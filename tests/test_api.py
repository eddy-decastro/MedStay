"""Smoke tests de l'API (Phase 2) : la chaine repond et respecte le contrat JSON.

Les vrais tests de proprietes metier (lower <= point <= upper, alpha plus petit =>
intervalle plus large, ...) arrivent en Phase 7, une fois le modele reel branche.
"""

from fastapi.testclient import TestClient

from src.api.main import app

# TestClient appelle l'app en memoire, sans lancer uvicorn ni ouvrir de port :
# les tests restent rapides et n'ont besoin d'aucun serveur en CI.
client = TestClient(app)


def test_health_renvoie_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_respecte_le_schema():
    response = client.post("/predict", json={"age": "[50-60)"})
    assert response.status_code == 200

    body = response.json()
    # On verifie le CONTRAT (cles + types), pas les valeurs : elles sont
    # factices en Phase 2 et changeront quand le vrai modele sera branche.
    assert set(body) == {
        "point_estimate",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "coverage_level",
    }
    assert all(isinstance(v, float) for v in body.values())
