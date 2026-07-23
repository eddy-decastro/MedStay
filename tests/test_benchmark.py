"""Tests du banc d'essai comparatif.

On verifie la STRUCTURE des pipelines, pas la performance des modeles : aucun
entrainement lourd ici, la comparaison chiffree vit dans models/benchmark.json.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.benchmark import (
    CATEGORICAL_NATIVE,
    build_models,
    build_onehot_preprocessor,
)


@pytest.fixture
def X_mixte() -> pd.DataFrame:
    """Features melant numerique et categoriel, comme le vrai jeu."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.integers(0, 20, size=n),
            "cat_x": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
            "cat_y": pd.Categorical(rng.choice(["p", "q"], size=n)),
        }
    )


def test_onehot_augmente_bien_le_nombre_de_colonnes(X_mixte):
    """La demonstration centrale : sans support categoriel, la matrice explose."""
    prep = build_onehot_preprocessor(X_mixte, scale=False)
    transforme = prep.fit_transform(X_mixte)

    # 2 numeriques + (3 + 2) modalites one-hot = 7 colonnes, contre 4 en entree.
    assert transforme.shape[1] == 7
    assert transforme.shape[1] > X_mixte.shape[1]


def test_onehot_ne_laisse_aucune_colonne_categorielle(X_mixte):
    """Un MLP ou une foret sklearn n'accepte que du numerique."""
    prep = build_onehot_preprocessor(X_mixte, scale=False)
    transforme = prep.fit_transform(X_mixte)
    assert np.issubdtype(transforme.dtype, np.number)


def test_normalisation_appliquee_seulement_si_demandee(X_mixte):
    """Le MLP a besoin de features centrees-reduites, pas les arbres."""
    sans = build_onehot_preprocessor(X_mixte, scale=False).fit_transform(X_mixte)
    avec = build_onehot_preprocessor(X_mixte, scale=True).fit_transform(X_mixte)

    # Les 2 premieres colonnes sont les numeriques (ordre du ColumnTransformer).
    assert avec[:, :2].mean() == pytest.approx(0.0, abs=1e-6)
    assert avec[:, :2].std() == pytest.approx(1.0, abs=1e-2)
    # Sans normalisation, num_b (entiers 0-20) garde son echelle d'origine.
    assert sans[:, 1].max() > 1.5


def test_les_cinq_modeles_sont_construits(X_mixte):
    modeles = build_models(X_mixte)
    assert set(modeles) == {
        "naive_median",
        "lightgbm",
        "xgboost",
        "random_forest",
        "mlp",
    }


def test_seuls_les_modeles_de_boosting_gerent_le_categoriel_nativement():
    """Justifie l'etape de one-hot imposee aux deux autres modeles."""
    assert CATEGORICAL_NATIVE == {"lightgbm", "xgboost"}


@pytest.mark.parametrize("nom", ["random_forest", "mlp"])
def test_les_modeles_sans_support_categoriel_ont_un_preprocesseur(X_mixte, nom):
    """Sans cette etape, sklearn leverait une erreur sur les dtypes 'category'."""
    modele = build_models(X_mixte)[nom]
    assert "prep" in modele.named_steps


@pytest.mark.parametrize("nom", ["lightgbm", "xgboost"])
def test_les_modeles_natifs_nont_pas_de_preprocesseur(X_mixte, nom):
    """Ils ingerent le DataFrame tel quel : c'est l'argument du choix."""
    modele = build_models(X_mixte)[nom]
    assert not hasattr(modele, "named_steps")
