"""Front Streamlit — squelette minimal (Phase 1).

Un seul bouton : appelle l'API et affiche la réponse brute. Le vrai formulaire
patient + graphiques arrivent en Phase 8. Règle non négociable : ce fichier ne
charge JAMAIS de modèle, il ne parle qu'en HTTP à l'API (CLAUDE.md contrainte 4).
"""

import os

import httpx
import streamlit as st

# En local : API sur localhost:8000. Dans le conteneur HF, start.sh lance les deux
# processus dans le même Space donc localhost fonctionne aussi côté prod.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("MedStay-CI")
st.caption("Prédiction de durée de séjour hospitalier — squelette Phase 1")

if st.button("Prédire (patient factice)"):
    response = httpx.post(
        f"{API_URL}/predict",
        json={"age": "[50-60)"},
        timeout=10.0,
    )
    st.json(response.json())
