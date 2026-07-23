"""Tests du post-traitement des intervalles conformes.

Importe src.models.conformal et NON src.models.calibrate : ce dernier importe
mapie, absent de requirements.txt (dependance d'entrainement). La CI n'installe
que les dependances de production, donc ces tests doivent tourner sans mapie.

Les proprietes verifiees ici sont celles exigees par la SPEC :
lower <= point <= upper, largeur > 0, bornes dans les limites physiologiques.
"""

import numpy as np
import pytest

from src.models.conformal import TARGET_MAX, TARGET_MIN, postprocess


def test_bornes_ramenees_dans_les_limites_physiologiques():
    """Un sejour dure de 1 a 14 jours : aucune borne ne doit sortir de la."""
    point = np.array([0.5, 7.0, 20.0])
    low = np.array([-3.0, 2.0, 13.0])
    high = np.array([0.9, 9.0, 18.0])

    point, low, high = postprocess(point, low, high)

    assert (low >= TARGET_MIN).all()
    assert (high <= TARGET_MAX).all()


def test_point_toujours_dans_son_intervalle():
    """LA propriete de la SPEC : annoncer un point hors de ses bornes serait
    incoherent pour l'utilisateur."""
    # Cas reels observes : la mediane tombait sous la borne basse.
    point = np.array([0.86, 0.97, 12.0, 3.0])
    low = np.array([0.99, 0.98, 2.0, 2.0])
    high = np.array([5.70, 5.29, 8.0, 9.0])

    point, low, high = postprocess(point, low, high)

    assert (point >= low).all()
    assert (point <= high).all()


def test_largeur_toujours_positive_ou_nulle():
    point = np.array([3.0, 5.0])
    low = np.array([2.0, 4.0])
    high = np.array([6.0, 9.0])

    _, low, high = postprocess(point, low, high)
    assert (high - low >= 0).all()


def test_bornes_croisees_sont_reordonnees():
    """Les trois quantiles etant appris independamment, rien ne garantit leur
    ordre : on retablit plutot que de renvoyer un intervalle vide."""
    point = np.array([5.0])
    low = np.array([9.0])  # plus haut que high
    high = np.array([3.0])

    _, low, high = postprocess(point, low, high)

    assert low <= high
    assert high - low >= 0


def test_troncature_ne_fait_jamais_baisser_la_couverture():
    """Propriete cle : tronquer a [1, 14] preserve la garantie conforme.

    La vraie valeur etant toujours dans [1, 14], si elle appartenait a
    l'intervalle d'origine elle appartient encore a l'intervalle tronque.
    La couverture ne peut donc que rester egale ou augmenter -- jamais baisser,
    ce qui invaliderait la garantie a 90 %.
    """
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(TARGET_MIN, TARGET_MAX + 1, size=n).astype(float)

    # Intervalles bruites debordant volontairement des bornes physiologiques,
    # certains tombant meme entierement en dehors.
    centre = y + rng.normal(scale=2.0, size=n)
    low = centre - rng.uniform(1, 6, size=n)
    high = centre + rng.uniform(1, 6, size=n)
    point = centre.copy()

    couverture_avant = ((y >= low) & (y <= high)).mean()
    _, low_t, high_t = postprocess(point, low, high)
    couverture_apres = ((y >= low_t) & (y <= high_t)).mean()

    assert couverture_apres >= couverture_avant


def test_troncature_neutre_quand_les_intervalles_restent_dans_les_bornes():
    """Cas du modele reel : aucun intervalle entierement hors de [1, 14],
    donc la couverture est rigoureusement identique avant et apres."""
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.integers(TARGET_MIN, TARGET_MAX + 1, size=n).astype(float)

    # Intervalles debordant des bornes mais chevauchant toujours [1, 14].
    low = y - rng.uniform(1, 4, size=n)
    high = y + rng.uniform(1, 4, size=n)
    point = y.astype(float)

    couverture_avant = ((y >= low) & (y <= high)).mean()
    _, low_t, high_t = postprocess(point, low, high)
    couverture_apres = ((y >= low_t) & (y <= high_t)).mean()

    assert couverture_avant == pytest.approx(couverture_apres)


def test_intervalle_deja_valide_est_inchange():
    """Le post-traitement ne doit rien modifier quand tout est deja coherent."""
    point = np.array([4.0, 7.0])
    low = np.array([2.0, 5.0])
    high = np.array([6.0, 10.0])

    p, lo, hi = postprocess(point.copy(), low.copy(), high.copy())

    assert np.array_equal(p, point)
    assert np.array_equal(lo, low)
    assert np.array_equal(hi, high)
