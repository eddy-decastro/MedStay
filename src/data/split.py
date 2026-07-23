"""Split train / calibration / test, calcule UNE SEULE FOIS et sauvegarde.

C'est le fichier le plus critique du projet pour la validite statistique.

Trois jeux disjoints, aux roles strictement separes (contrainte 3 de CLAUDE.md) :
  - train (60 %)       : entrainement des modeles LightGBM
  - calibration (20 %) : MAPIE UNIQUEMENT. Jamais d'entrainement, jamais de choix
                         d'hyperparametres dessus. C'est ce qui rend la garantie
                         de couverture a 90 % valide.
  - test (20 %)        : evaluation finale UNIQUEMENT, regarde le plus tard possible.

Le split n'est JAMAIS recalcule : une fois les fichiers ecrits, ce script refuse
de les regenerer sans --force. Un split qui bouge d'une execution a l'autre
invaliderait toute comparaison de metriques et, pire, permettrait de choisir
implicitement le decoupage qui donne les meilleurs chiffres.

Usage : python -m src.data.split [--force]
"""

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CALIBRATION_RATIO,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
)
from src.data.load import DATA_FILE
from src.data.preprocess import TARGET, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLIT_DIR = PROCESSED_DATA_DIR / "splits"
SPLIT_NAMES = ("train", "calibration", "test")

# joblib et non CSV : le CSV perd les dtypes "category" poses par preprocess(),
# que LightGBM utilise pour son traitement natif des variables categorielles.
# Un aller-retour CSV les transformerait en texte et changerait le modele.
SPLIT_SUFFIX = ".joblib"

METADATA_FILE = SPLIT_DIR / "metadata.json"


def _split_paths() -> dict[str, object]:
    return {name: SPLIT_DIR / f"{name}{SPLIT_SUFFIX}" for name in SPLIT_NAMES}


def make_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Decoupe le DataFrame en train / calibration / test stratifies sur la cible."""
    # Stratification sur les VALEURS BRUTES de la cible (14 modalites entieres).
    # Le plus petit strate compte 642 individus, soit ~128 en test : aucun risque
    # de strate vide, donc inutile de regrouper en tranches. C'est la
    # stratification la plus fine possible.
    #
    # Pourquoi stratifier : sans cela, les sejours longs (1 % de la cible) se
    # repartiraient au hasard. Un set de calibration pauvre en sejours longs
    # produirait des quantiles mal calibres dans la queue de distribution --
    # precisement la ou l'intervalle importe le plus.
    strata = df[TARGET]

    # Le split se fait en DEUX temps : sklearn ne sait couper qu'en deux.
    # Temps 1 : on isole le test.
    rest, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=strata,
        random_state=SEED,
        shuffle=True,
    )

    # Temps 2 : on coupe le reste (80 % du total) en train et calibration.
    # La proportion demandee est relative a ce reste, pas au total :
    #   0,20 / (1 - 0,20) = 0,25  ->  0,25 x 80 % = 20 % du total. CQFD.
    calibration_share = CALIBRATION_RATIO / (1 - TEST_RATIO)
    train, calibration = train_test_split(
        rest,
        test_size=calibration_share,
        stratify=rest[TARGET],
        random_state=SEED,
        shuffle=True,
    )

    return {"train": train, "calibration": calibration, "test": test}


def _check_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Verifie qu'aucune ligne n'appartient a deux jeux (garde-fou anti-fuite)."""
    index_sets = {name: set(part.index) for name, part in splits.items()}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b and index_sets[a] & index_sets[b]:
                raise ValueError(f"Chevauchement entre {a} et {b}")


def save_split(splits: dict[str, pd.DataFrame]) -> None:
    """Sauvegarde les jeux, leurs indices et les metadonnees de reproductibilite."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in _split_paths().items():
        joblib.dump(splits[name], path)

        # Indices sauvegardes a part, en texte : permet de tracer a quel jeu
        # appartient une ligne sans charger les donnees, et de verifier
        # l'integrite du split par simple diff.
        pd.Series(splits[name].index, name="index").to_csv(
            SPLIT_DIR / f"{name}_indices.csv", index=False
        )

    total = sum(len(part) for part in splits.values())
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "calibration": CALIBRATION_RATIO,
            "test": TEST_RATIO,
        },
        "stratified_on": TARGET,
        "n_total": total,
        "sizes": {name: len(part) for name, part in splits.items()},
        "observed_shares": {
            name: round(len(part) / total, 4) for name, part in splits.items()
        },
        "n_features": splits["train"].shape[1] - 1,
        "columns": sorted(splits["train"].columns),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Metadonnees ecrites : %s", METADATA_FILE)


def load_split() -> dict[str, pd.DataFrame]:
    """Recharge le split sauvegarde. Leve une erreur s'il n'a jamais ete calcule."""
    paths = _split_paths()
    manquants = [str(p) for p in paths.values() if not p.exists()]
    if manquants:
        raise FileNotFoundError(
            f"Split introuvable ({manquants[0]}...). Lancer : python -m src.data.split"
        )
    return {name: joblib.load(path) for name, path in paths.items()}


def main(force: bool = False) -> None:
    """Calcule le split s'il n'existe pas encore, et refuse de l'ecraser sinon."""
    if all(p.exists() for p in _split_paths().values()) and not force:
        logger.warning(
            "Split deja present dans %s : rien a faire. "
            "Le recalculer invaliderait toutes les metriques deja mesurees. "
            "Utiliser --force uniquement si le preprocessing a change.",
            SPLIT_DIR,
        )
        return

    raw = pd.read_csv(RAW_DATA_DIR / DATA_FILE, low_memory=False)
    # On repart du CSV brut plutot que de clean.csv : l'aller-retour par CSV
    # perdrait les dtypes categoriels poses par preprocess().
    df = preprocess(raw)

    splits = make_split(df)
    _check_disjoint(splits)

    total = sum(len(part) for part in splits.values())
    for name, part in splits.items():
        logger.info(
            "%-12s : %6d lignes (%.1f %%)", name, len(part), len(part) / total * 100
        )

    save_split(splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcule et ECRASE le split existant. A n'utiliser que si le "
        "preprocessing a change : toutes les metriques anterieures deviennent "
        "incomparables.",
    )
    main(force=parser.parse_args().force)
