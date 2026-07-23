"""Telechargement reproductible du dataset UCI Diabetes 130-US Hospitals.

Aucun CSV n'est commite (contrainte 5 de CLAUDE.md) : ce script reconstruit
data/raw/ a l'identique sur n'importe quelle machine.

Usage : python -m src.data.load
"""

import logging
import urllib.request
from pathlib import Path

from src.config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# URL stable exposee par l'API UCI pour le dataset 296 (le lien .zip historique
# renvoie 404 depuis la refonte du site : UCI sert desormais le CSV brut).
UCI_URL = "https://archive.ics.uci.edu/static/public/296/data.csv"

DATA_FILE = "diabetic_data.csv"

# Garde-fou : le fichier attendu fait ~19,5 Mo. En dessous, on a probablement
# telecharge une page d'erreur HTML plutot que les donnees.
MIN_EXPECTED_BYTES = 15_000_000


def download(force: bool = False) -> Path:
    """Telecharge le dataset UCI dans data/raw/ et renvoie le chemin du CSV."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DATA_DIR / DATA_FILE

    # Idempotent : on ne retelecharge pas si le fichier est deja la.
    if target.exists() and not force:
        logger.info("Deja present : %s", target)
        return target

    logger.info("Telechargement depuis %s", UCI_URL)
    with urllib.request.urlopen(UCI_URL, timeout=180) as response:
        content = response.read()

    if len(content) < MIN_EXPECTED_BYTES:
        raise ValueError(
            f"Fichier suspect ({len(content)} octets < {MIN_EXPECTED_BYTES}) : "
            "l'URL UCI a probablement encore change."
        )

    target.write_bytes(content)
    logger.info("Ecrit : %s (%.1f Mo)", target, len(content) / 1e6)
    return target


if __name__ == "__main__":
    download()
