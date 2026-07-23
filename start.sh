#!/bin/sh
# Lance les deux process du conteneur Render : uvicorn (interne, 8000) puis Streamlit
# (expose sur $PORT). Un seul conteneur, deux process, Streamlit parle a l'API en HTTP
# localhost (CLAUDE.md contrainte 4).
set -e

# "&" = arriere-plan : uvicorn ne bloque pas le script, sinon streamlit ne demarrerait jamais.
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Render injecte $PORT ; en local (docker run sans $PORT) on retombe sur 7860.
# Pas de "&" ici : streamlit reste au premier plan pour garder le conteneur vivant.
streamlit run app/streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0
