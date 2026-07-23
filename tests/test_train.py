"""Tests de la baseline : metriques, validation croisee, garde-fou de qualite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.data.preprocess import TARGET
from src.models.train import compute_metrics, cross_validate, split_xy


@pytest.fixture
def df_apprenable() -> pd.DataFrame:
    """Jeu synthetique ou la cible depend REELLEMENT des features.

    Indispensable : sur du bruit pur, un modele ne peut pas battre la mediane
    et le test de garde-fou n'aurait aucun sens.
    """
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 5, size=n)
    bruit = rng.normal(scale=0.5, size=n)
    # Relation lineaire + bruit, ramenee dans [1, 14] comme la vraie cible.
    y = np.clip((3 + 2 * x1 + 0.5 * x2 + bruit).round(), 1, 14)
    return pd.DataFrame({TARGET: y, "x1": x1, "x2": x2})


def test_split_xy_separe_la_cible(df_apprenable):
    X, y = split_xy(df_apprenable)
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y) == len(df_apprenable)


def test_compute_metrics_sur_prediction_parfaite():
    """Une prediction exacte doit donner MAE = RMSE = 0 et R2 = 1."""
    y = pd.Series([1.0, 4.0, 7.0, 14.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_valeurs_connues():
    """Erreurs de 1 et 3 jours : MAE = 2, RMSE = sqrt(5) ~ 2,236."""
    y_true = pd.Series([2.0, 6.0])
    y_pred = pd.Series([3.0, 3.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5))


def test_compute_metrics_expose_les_trois_metriques_de_la_spec():
    y = pd.Series([1.0, 2.0, 3.0])
    assert set(compute_metrics(y, y)) == {"mae", "rmse", "r2"}


def test_cross_validate_renvoie_moyenne_et_ecart_type(df_apprenable):
    X, y = split_xy(df_apprenable)
    scores = cross_validate(DummyRegressor(strategy="median"), X, y)

    assert set(scores) == {"mae", "rmse", "r2"}
    for valeurs in scores.values():
        assert set(valeurs) == {"mean", "std"}
        assert valeurs["std"] >= 0


def test_lightgbm_bat_le_modele_naif(df_apprenable):
    """Garde-fou de qualite : si LightGBM ne bat pas la mediane, quelque chose
    est casse (features melangees, cible mal alignee, mauvais objectif)."""
    from lightgbm import LGBMRegressor

    X, y = split_xy(df_apprenable)
    # Parametres reduits : ce test verifie une relation d'ordre, pas la
    # performance absolue. Inutile de payer 400 arbres ici.
    lgbm = cross_validate(
        LGBMRegressor(objective="regression", n_estimators=30, verbose=-1, n_jobs=1),
        X,
        y,
    )
    naif = cross_validate(DummyRegressor(strategy="median"), X, y)

    assert lgbm["mae"]["mean"] < naif["mae"]["mean"]
    assert lgbm["r2"]["mean"] > naif["r2"]["mean"]


def test_cross_validate_est_deterministe(df_apprenable):
    """Meme seed, memes scores : sans cela les metriques ne sont pas comparables."""
    X, y = split_xy(df_apprenable)
    a = cross_validate(DummyRegressor(strategy="median"), X, y)
    b = cross_validate(DummyRegressor(strategy="median"), X, y)
    assert a == b
