# Dockerfile MULTI-STAGE (phase 5). La version simple de la phase 1 installait
# tout dans une seule image ; ici on separe la CONSTRUCTION de l'EXECUTION.
#
# Interet : pip, ses caches et ses outils de compilation restent dans l'etage
# "builder" et ne sont jamais copies dans l'image finale. Seules les
# bibliotheques installees le sont.

# ============================================================================
# ETAGE 1 : builder -- installe les dependances, puis sera jete
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# --user installe dans /root/.local, un dossier unique et facile a copier
# ensuite dans l'image finale. --no-cache-dir evite de conserver les archives
# telechargees (~100 Mo inutiles).
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# ETAGE 2 : runtime -- l'image reellement deployee
# ============================================================================
FROM python:3.11-slim

# Utilisateur NON-ROOT. Si l'application est compromise, l'attaquant n'obtient
# pas les droits administrateur du conteneur. C'est la mesure de securite la
# plus rentable d'un Dockerfile.
RUN useradd --create-home --shell /bin/bash medstay

WORKDIR /app

# On recupere UNIQUEMENT les bibliotheques installees a l'etage precedent.
# pip lui-meme, ses caches et les outils de build restent derriere.
COPY --from=builder /root/.local /home/medstay/.local

# Les executables installes avec --user (uvicorn, streamlit) vivent la.
ENV PATH=/home/medstay/.local/bin:$PATH \
    # Evite l'ecriture de fichiers .pyc dans le conteneur.
    PYTHONDONTWRITEBYTECODE=1 \
    # Affiche les logs immediatement au lieu de les tamponner : sans cela les
    # journaux Render arrivent par blocs, voire disparaissent en cas de crash.
    PYTHONUNBUFFERED=1 \
    # Streamlit ne cherche pas a ouvrir de navigateur ni a collecter de
    # statistiques d'usage dans un conteneur.
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --chown evite un chmod ulterieur : les fichiers appartiennent directement au
# bon utilisateur.
COPY --chown=medstay:medstay src/ ./src/
COPY --chown=medstay:medstay app/ ./app/
COPY --chown=medstay:medstay models/ ./models/
COPY --chown=medstay:medstay reports/ ./reports/
COPY --chown=medstay:medstay start.sh ./

RUN chmod +x start.sh

USER medstay

# Informatif : Render route le trafic via $PORT. Utile au `docker run -p` local.
EXPOSE 7860

# Sonde de sante interrogeant l'API interne. Docker marque le conteneur
# "unhealthy" si elle echoue 3 fois de suite. --start-period laisse 40 s de
# demarrage avant de commencer a compter les echecs (chargement du modele).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["./start.sh"]
