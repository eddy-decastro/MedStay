# Dockerfile MULTI-STAGE (phase 5). La version simple de la phase 1 installait
# tout dans une seule image ; ici on separe la CONSTRUCTION de l'EXECUTION.
#
# Interet : pip, ses caches et ses outils de compilation restent dans l'etage
# "builder" et ne sont jamais copies dans l'image finale.

# ============================================================================
# ETAGE 1 : builder -- installe les dependances, puis sera jete
# ============================================================================
FROM python:3.11-slim AS builder

# Environnement virtuel dans un chemin ABSOLU et neutre.
# On prefere cela a "pip install --user" : ce dernier installe dans ~/.local,
# donc dans un dossier qui depend de la variable HOME. Or l'instruction USER de
# Docker ne redefinit pas HOME de maniere fiable -- Python cherche alors ses
# paquets au mauvais endroit et l'import echoue, alors meme que PATH semble
# correct. Un venv a chemin fixe elimine ce piege.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# requirements.txt copie SEUL et en premier : tant que ce fichier ne change pas,
# Docker reutilise le cache de cette couche et ne reinstalle rien.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# ETAGE 2 : runtime -- l'image reellement deployee
# ============================================================================
FROM python:3.11-slim

# libgomp1 : bibliotheque OpenMP de GNU, exigee par LightGBM pour son calcul
# multi-thread. python:3.11-slim ne l'embarque pas, et son absence ne se voit
# qu'au CHARGEMENT du modele :
#     OSError: libgomp.so.1: cannot open shared object file
# L'image se construit sans erreur, elle plante au demarrage. C'est le piege
# classique de LightGBM en conteneur slim.
#
# rm -rf /var/lib/apt/lists/* dans la MEME instruction RUN : dans une couche
# separee, les fichiers resteraient dans l'historique de l'image malgre la
# suppression.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur NON-ROOT. Si l'application est compromise, l'attaquant n'obtient
# pas les droits administrateur du conteneur. C'est la mesure de securite la
# plus rentable d'un Dockerfile.
RUN useradd --create-home --shell /bin/bash medstay

# On recupere UNIQUEMENT l'environnement virtuel : ni pip, ni ses caches, ni
# les outils de compilation de l'etage precedent.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    # HOME defini EXPLICITEMENT : l'instruction USER de Docker ne le fait pas de
    # maniere fiable. Sans cela HOME reste /root, dossier que l'utilisateur
    # medstay ne peut pas ecrire -- et Streamlit, qui ecrit sa configuration
    # dans ~/.streamlit, refuse alors de demarrer. Le port expose ne repond
    # jamais, sans message d'erreur evident.
    HOME=/home/medstay \
    # Pas de fichiers .pyc ecrits dans le conteneur.
    PYTHONDONTWRITEBYTECODE=1 \
    # Logs emis immediatement au lieu d'etre tamponnes : sans cela les journaux
    # Render arrivent par blocs, voire disparaissent en cas de crash.
    PYTHONUNBUFFERED=1 \
    # Streamlit ne cherche ni a ouvrir un navigateur ni a collecter de
    # statistiques d'usage dans un conteneur.
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

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

# Sonde de sante interrogeant l'API INTERNE : c'est elle qui porte le modele.
# --start-period laisse 40 s de demarrage (chargement du modele) avant de
# commencer a compter les echecs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["./start.sh"]
