#!/bin/sh
# Lance les deux process du conteneur : uvicorn (interne, 8000) puis Streamlit
# (expose sur $PORT). Streamlit parle a l'API en HTTP localhost, il ne charge
# jamais le modele (contrainte 4 de CLAUDE.md).
set -e

# "&" = arriere-plan : uvicorn ne bloque pas le script, sinon streamlit ne
# demarrerait jamais.
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Attente ACTIVE que l'API reponde. Le chargement du modele prend quelques
# secondes ; sans cette attente, les premieres requetes de Streamlit
# echoueraient et l'interface afficherait une erreur au demarrage.
echo "Attente du demarrage de l'API..."
i=0
while [ $i -lt 60 ]; do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
        echo "API prete."
        break
    fi
    # Si uvicorn est mort, inutile d'attendre : on echoue tout de suite avec un
    # message clair plutot que de servir une interface sans backend.
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "ERREUR : l'API s'est arretee au demarrage." >&2
        wait "$API_PID"
        exit 1
    fi
    i=$((i + 1))
    sleep 1
done

if [ $i -ge 60 ]; then
    echo "ERREUR : l'API n'a pas repondu en 60 secondes." >&2
    exit 1
fi

# Render injecte $PORT ; en local (docker run sans $PORT) on retombe sur 7860.
# Pas de "&" ici : streamlit reste au premier plan pour garder le conteneur vivant.
exec streamlit run app/streamlit_app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0
