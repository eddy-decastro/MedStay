"""Tests du split : proportions, disjonction, stratification, determinisme.

Ces tests protegent la validite statistique du projet. Un split qui fuit ou qui
bouge invaliderait la garantie de couverture a 90 % sans qu'aucun code ne plante.
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.config import CALIBRATION_RATIO, SEED, TEST_RATIO, TRAIN_RATIO
from src.data import split as split_module
from src.data.preprocess import TARGET
from src.data.split import SPLIT_NAMES, load_split, make_split, save_split


@pytest.fixture
def df_synthetique() -> pd.DataFrame:
    """Jeu synthetique de 7 000 lignes reproduisant l'asymetrie de la cible reelle."""
    rng = np.random.default_rng(0)
    n = 7000
    # Poids decroissants : imite la distribution observee (mode a 3 j, queue a 14 j).
    poids = np.array([15, 17, 18, 13, 10, 7, 6, 4, 3, 2, 2, 1, 1, 1], dtype=float)
    poids /= poids.sum()
    return pd.DataFrame(
        {
            TARGET: rng.choice(np.arange(1, 15), size=n, p=poids),
            "num_feature": rng.normal(size=n),
            "cat_feature": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )


def test_proportions_respectees(df_synthetique):
    """Les trois jeux doivent faire 60 / 20 / 20 a moins d'un point de pourcentage."""
    splits = make_split(df_synthetique)
    total = sum(len(part) for part in splits.values())

    attendu = {
        "train": TRAIN_RATIO,
        "calibration": CALIBRATION_RATIO,
        "test": TEST_RATIO,
    }
    for name, ratio in attendu.items():
        assert abs(len(splits[name]) / total - ratio) < 0.01


def test_aucune_ligne_perdue_ni_dupliquee(df_synthetique):
    """La reunion des trois jeux doit redonner exactement le jeu de depart."""
    splits = make_split(df_synthetique)

    total = sum(len(part) for part in splits.values())
    reunion = set().union(*(set(part.index) for part in splits.values()))

    # Egalite des effectifs ET des index : detecte a la fois une ligne perdue
    # et une ligne dupliquee dans deux jeux.
    assert total == len(df_synthetique)
    assert reunion == set(df_synthetique.index)


@pytest.mark.parametrize(
    ("a", "b"), [("train", "calibration"), ("train", "test"), ("calibration", "test")]
)
def test_jeux_disjoints(df_synthetique, a, b):
    """LE test critique : une ligne partagee invaliderait la garantie conforme."""
    splits = make_split(df_synthetique)
    assert not set(splits[a].index) & set(splits[b].index)


def test_stratification_preserve_la_distribution(df_synthetique):
    """Chaque jeu doit refleter la distribution de la cible du jeu complet."""
    splits = make_split(df_synthetique)
    reference = df_synthetique[TARGET].value_counts(normalize=True).sort_index()

    for name in SPLIT_NAMES:
        observe = splits[name][TARGET].value_counts(normalize=True).sort_index()
        # Tolerance large (2 points) : sur 7 000 lignes les strates rares sont petites.
        assert (observe - reference).abs().max() < 0.02


def test_split_est_deterministe(df_synthetique):
    """Meme entree et meme seed doivent redonner exactement le meme decoupage.

    Sans cela, chaque execution changerait les metriques et il deviendrait
    possible de choisir, meme involontairement, le decoupage le plus flatteur.
    """
    a = make_split(df_synthetique)
    b = make_split(df_synthetique)
    for name in SPLIT_NAMES:
        assert list(a[name].index) == list(b[name].index)


def test_dtypes_categoriels_preserves(df_synthetique):
    """LightGBM exploite les dtypes 'category' : le split ne doit pas les degrader."""
    splits = make_split(df_synthetique)
    for name in SPLIT_NAMES:
        assert isinstance(splits[name]["cat_feature"].dtype, pd.CategoricalDtype)


def test_toutes_les_modalites_de_cible_presentes_partout(df_synthetique):
    """Aucun jeu ne doit manquer un sejour long : ce sont les cas critiques."""
    attendu = set(df_synthetique[TARGET].unique())
    splits = make_split(df_synthetique)
    for name in SPLIT_NAMES:
        assert set(splits[name][TARGET].unique()) == attendu


# --- Persistance ------------------------------------------------------------


@pytest.fixture
def split_dir_temporaire(tmp_path, monkeypatch):
    """Redirige les ecritures vers un dossier jetable, jamais le vrai split."""
    monkeypatch.setattr(split_module, "SPLIT_DIR", tmp_path)
    monkeypatch.setattr(split_module, "METADATA_FILE", tmp_path / "metadata.json")
    return tmp_path


def test_sauvegarde_puis_rechargement_identique(df_synthetique, split_dir_temporaire):
    """Un aller-retour disque ne doit alterer ni les lignes ni les dtypes."""
    splits = make_split(df_synthetique)
    save_split(splits)
    recharges = load_split()

    for name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(splits[name], recharges[name])


def test_load_split_echoue_si_jamais_calcule(split_dir_temporaire):
    """Message explicite plutot qu'un fichier introuvable cryptique."""
    with pytest.raises(FileNotFoundError, match="python -m src.data.split"):
        load_split()


def test_metadonnees_tracent_la_reproductibilite(df_synthetique, split_dir_temporaire):
    """La seed et les proportions doivent etre archivees avec le split."""
    splits = make_split(df_synthetique)
    save_split(splits)

    metadata = json.loads((split_dir_temporaire / "metadata.json").read_text())
    assert metadata["seed"] == SEED
    assert metadata["stratified_on"] == TARGET
    assert metadata["n_total"] == len(df_synthetique)
    assert metadata["sizes"]["train"] == len(splits["train"])


def test_main_refuse_d_ecraser_un_split_existant(
    df_synthetique, split_dir_temporaire, caplog
):
    """Garde-fou central : recalculer invaliderait toutes les metriques mesurees."""
    save_split(make_split(df_synthetique))

    # Sans --force, main() doit sortir sans toucher au disque ni lire le CSV brut.
    empreintes_avant = {
        p.name: p.stat().st_mtime_ns for p in split_dir_temporaire.iterdir()
    }
    with caplog.at_level(logging.WARNING):
        split_module.main(force=False)

    empreintes_apres = {
        p.name: p.stat().st_mtime_ns for p in split_dir_temporaire.iterdir()
    }
    assert empreintes_avant == empreintes_apres
    assert "deja present" in caplog.text
