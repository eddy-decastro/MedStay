"""Validation statistique des intervalles conformes -- le coeur du projet.

Repond a UNE question : la promesse de couverture a 90 % est-elle tenue sur des
donnees jamais vues ? Un intervalle sans couverture verifiee n'est qu'une
decoration.

C'est ici, et seulement ici, que le TEST SET est ouvert. Il n'a servi ni a
l'entrainement, ni a la calibration, ni au choix des hyperparametres.

Cinq analyses (SPEC section 4) :
  1. Couverture empirique globale + largeur des intervalles
  2. Couverture par SOUS-GROUPE (age, type d'admission, assurance)
  3. Courbe couverture vs alpha, comparee a la diagonale ideale
  4. Comparaison quantiles CALIBRES vs NON calibres (l'apport du conformal)
  5. Largeur en fonction de la duree reelle (adaptativite)

Usage : python -m src.models.evaluate
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sans interface : indispensable hors notebook
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (  # noqa: E402
    ALPHA,
    EVALUATION_REPORT_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    MODEL_PATH,
)
from src.data.split import load_split  # noqa: E402
from src.models.train import split_xy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Chemin defini dans config.py (voir la note qui l'accompagne).

# Palette unique pour toutes les figures.
BLEU, ROUGE, VERT, GRIS = "#2563eb", "#dc2626", "#059669", "#6b7280"


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Proportion de vraies valeurs tombant dans leur intervalle."""
    return float(((y >= low) & (y <= high)).mean())


def _save(fig, nom: str) -> str:
    """Sauvegarde une figure et renvoie son chemin relatif."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chemin = FIGURES_DIR / f"{nom}.png"
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("   figure : %s", chemin.name)
    return f"reports/figures/{nom}.png"


# --- 1. Couverture globale --------------------------------------------------


def plot_coverage_and_width(y, low, high, alpha: float) -> dict:
    """Couverture empirique et distribution des largeurs."""
    couv = coverage(y, low, high)
    largeurs = high - low
    cible = 1 - alpha

    # Erreur type d'une proportion : sqrt(p(1-p)/n). Elle donne la marge de
    # fluctuation attendue par simple hasard d'echantillonnage, sans quoi on ne
    # peut pas dire si 0,898 "vaut" 0,900.
    n = len(y)
    erreur_type = np.sqrt(cible * (1 - cible) / n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(["Couverture\nmesuree", "Cible"], [couv, cible], color=[BLEU, GRIS])
    axes[0].axhline(cible, color=ROUGE, ls="--", lw=1)
    # Bande a +/- 2 erreurs types : ~95 % des tirages devraient y tomber.
    axes[0].axhspan(
        cible - 2 * erreur_type,
        cible + 2 * erreur_type,
        color=ROUGE,
        alpha=0.12,
        label="+/- 2 erreurs types",
    )
    for i, v in enumerate([couv, cible]):
        axes[0].text(i, v + 0.008, f"{v:.4f}", ha="center", fontweight="bold")
    axes[0].set(
        ylim=(0.7, 1.0),
        ylabel="Couverture",
        title=f"Couverture empirique sur le test (n = {n:,})",
    )
    axes[0].legend(fontsize=8)

    axes[1].hist(largeurs, bins=40, color=BLEU, edgecolor="white", lw=0.4)
    axes[1].axvline(
        largeurs.mean(),
        color=ROUGE,
        ls="--",
        label=f"moyenne = {largeurs.mean():.2f} j",
    )
    axes[1].set(
        xlabel="Largeur de l'intervalle (jours)",
        ylabel="Nombre de patients",
        title="Distribution des largeurs",
    )
    axes[1].legend()

    plt.tight_layout()
    figure = _save(fig, "01_couverture_et_largeur")

    return {
        "n_test": n,
        "target_coverage": round(cible, 4),
        "empirical_coverage": round(couv, 4),
        "standard_error": round(float(erreur_type), 5),
        "ecarts_types_a_la_cible": round(float((couv - cible) / erreur_type), 2),
        "mean_width": round(float(largeurs.mean()), 3),
        "median_width": round(float(np.median(largeurs)), 3),
        "min_width": round(float(largeurs.min()), 3),
        "max_width": round(float(largeurs.max()), 3),
        "figure": figure,
    }


# --- 2. Couverture par sous-groupe -------------------------------------------


def plot_subgroup_coverage(df, y, low, high, alpha: float) -> dict:
    """Couverture conditionnelle : la garantie tient-elle dans chaque population ?

    Point THEORIQUE essentiel : le conformal split ne garantit que la couverture
    MARGINALE, c'est-a-dire moyennee sur toute la population. Rien n'assure
    qu'elle soit atteinte dans chaque sous-groupe. Un modele peut tres bien
    couvrir 90 % globalement tout en ne couvrant que 80 % des patients ages.
    C'est precisement ce qu'on va mesurer.
    """
    cible = 1 - alpha
    colonnes = ["age", "admission_type_id", "payer_code"]
    resultats = {}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, col in zip(axes, colonnes):
        lignes = []
        for modalite in df[col].astype(str).unique():
            masque = (df[col].astype(str) == modalite).to_numpy()
            # Sous 100 individus, une couverture n'est pas estimable de facon
            # fiable : l'erreur type depasserait 3 points.
            if masque.sum() < 100:
                continue
            lignes.append(
                {
                    "modalite": modalite,
                    "n": int(masque.sum()),
                    "couverture": coverage(y[masque], low[masque], high[masque]),
                    "largeur": float(np.mean(high[masque] - low[masque])),
                }
            )

        lignes.sort(key=lambda r: r["couverture"])
        resultats[col] = [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ligne.items()}
            for ligne in lignes
        ]

        couvertures = [ligne["couverture"] for ligne in lignes]
        etiquettes = [f"{ligne['modalite']} (n={ligne['n']})" for ligne in lignes]
        couleurs = [ROUGE if c < cible - 0.03 else BLEU for c in couvertures]

        ax.barh(etiquettes, couvertures, color=couleurs)
        ax.axvline(cible, color=ROUGE, ls="--", lw=1.2, label=f"cible {cible:.0%}")
        ax.set(xlim=(0.7, 1.0), xlabel="Couverture", title=f"Par {col}")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    figure = _save(fig, "02_couverture_par_sous_groupe")

    toutes = [ligne["couverture"] for lignes in resultats.values() for ligne in lignes]
    return {
        "par_variable": resultats,
        "couverture_min": round(min(toutes), 4),
        "couverture_max": round(max(toutes), 4),
        "ecart_max": round(max(toutes) - min(toutes), 4),
        "figure": figure,
    }


# --- 3. Courbe couverture vs alpha -------------------------------------------


def plot_coverage_vs_alpha(predictor, X, y) -> dict:
    """La couverture suit-elle la cible a TOUS les niveaux de confiance ?

    Test le plus exigeant : si la calibration etait un coup de chance a 90 %,
    la courbe s'ecarterait de la diagonale aux autres niveaux.
    """
    alphas = np.array(
        [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    )
    mesures = []

    for a in alphas:
        _, low, high = predictor.predict(X, alpha=float(a))
        mesures.append(
            {
                "alpha": round(float(a), 3),
                "cible": round(float(1 - a), 3),
                "mesuree": round(coverage(y, low, high), 4),
                "largeur": round(float(np.mean(high - low)), 3),
            }
        )

    cibles = [m["cible"] for m in mesures]
    mesurees = [m["mesuree"] for m in mesures]
    largeurs = [m["largeur"] for m in mesures]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot([0.5, 1.0], [0.5, 1.0], ls="--", color=GRIS, label="ideal (diagonale)")
    axes[0].plot(cibles, mesurees, "o-", color=BLEU, label="mesure sur le test")
    axes[0].set(
        xlabel="Couverture visee (1 - alpha)",
        ylabel="Couverture obtenue",
        title="Calibration a tous les niveaux",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(cibles, largeurs, "o-", color=VERT)
    axes[1].set(
        xlabel="Couverture visee (1 - alpha)",
        ylabel="Largeur moyenne (jours)",
        title="Prix de la confiance",
    )
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    figure = _save(fig, "03_couverture_vs_alpha")

    ecarts = [abs(m["mesuree"] - m["cible"]) for m in mesures]
    return {
        "points": mesures,
        "ecart_absolu_moyen": round(float(np.mean(ecarts)), 4),
        "ecart_absolu_max": round(float(np.max(ecarts)), 4),
        "figure": figure,
    }


# --- 4. Calibre vs non calibre ------------------------------------------------


def plot_calibrated_vs_raw(predictor, X, y, alpha: float) -> dict:
    """A quoi sert la calibration conforme ? Preuve chiffree.

    On compare les quantiles BRUTS des deux LightGBM (sans correction) aux
    memes quantiles APRES correction conforme. C'est la demonstration que le
    conformal apporte quelque chose de mesurable, pas une couche cosmetique.
    """
    # Quantiles bruts, tels que sortis des modeles, sans marge conforme.
    brut_low = np.asarray(predictor.model_low.predict(X), dtype=float)
    brut_high = np.asarray(predictor.model_high.predict(X), dtype=float)
    brut_low, brut_high = np.clip(brut_low, 1, 14), np.clip(brut_high, 1, 14)

    _, cal_low, cal_high = predictor.predict(X, alpha=alpha)

    donnees = {
        "non calibre\n(quantiles bruts)": (brut_low, brut_high),
        "calibre\n(CQR)": (cal_low, cal_high),
    }
    couvertures = {k: coverage(y, lo, hi) for k, (lo, hi) in donnees.items()}
    largeurs = {k: float(np.mean(hi - lo)) for k, (lo, hi) in donnees.items()}
    cible = 1 - alpha

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    couleurs = [ROUGE if c < cible - 0.01 else VERT for c in couvertures.values()]
    axes[0].bar(list(couvertures), list(couvertures.values()), color=couleurs)
    axes[0].axhline(cible, color=ROUGE, ls="--", label=f"cible {cible:.0%}")
    for i, v in enumerate(couvertures.values()):
        axes[0].text(i, v + 0.006, f"{v:.4f}", ha="center", fontweight="bold")
    axes[0].set(ylim=(0.7, 1.0), ylabel="Couverture", title="Couverture atteinte")
    axes[0].legend(fontsize=8)

    axes[1].bar(list(largeurs), list(largeurs.values()), color=BLEU)
    for i, v in enumerate(largeurs.values()):
        axes[1].text(i, v + 0.06, f"{v:.2f} j", ha="center", fontweight="bold")
    axes[1].set(ylabel="Largeur moyenne (jours)", title="Cout en largeur")

    plt.tight_layout()
    figure = _save(fig, "04_calibre_vs_non_calibre")

    return {
        "non_calibre": {
            "couverture": round(couvertures["non calibre\n(quantiles bruts)"], 4),
            "largeur": round(largeurs["non calibre\n(quantiles bruts)"], 3),
        },
        "calibre": {
            "couverture": round(couvertures["calibre\n(CQR)"], 4),
            "largeur": round(largeurs["calibre\n(CQR)"], 3),
        },
        "figure": figure,
    }


# --- 5. Adaptativite ----------------------------------------------------------


def plot_width_vs_truth(y, low, high) -> dict:
    """L'intervalle s'elargit-il quand le cas est difficile ?

    Un intervalle conforme de largeur constante serait valide mais inutile.
    L'interet de la CQR est justement d'etre ADAPTATIVE.
    """
    largeurs = high - low
    df = pd.DataFrame(
        {"y": y, "largeur": largeurs, "couvert": (y >= low) & (y <= high)}
    )
    par_duree = df.groupby("y").agg(
        largeur=("largeur", "mean"), couverture=("couvert", "mean"), n=("y", "size")
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(par_duree.index, par_duree["largeur"], color=BLEU)
    axes[0].set(
        xlabel="Duree reelle du sejour (jours)",
        ylabel="Largeur moyenne (jours)",
        title="Adaptativite : largeur selon la difficulte du cas",
    )

    axes[1].bar(par_duree.index, par_duree["couverture"], color=BLEU)
    axes[1].axhline(0.9, color=ROUGE, ls="--", label="cible 90 %")
    axes[1].set(
        xlabel="Duree reelle du sejour (jours)",
        ylabel="Couverture",
        ylim=(0, 1.05),
        title="Couverture selon la duree reelle",
    )
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    figure = _save(fig, "05_adaptativite")

    return {
        "par_duree_reelle": [
            {
                "duree": int(duree),
                "n": int(ligne["n"]),
                "largeur": round(float(ligne["largeur"]), 3),
                "couverture": round(float(ligne["couverture"]), 4),
            }
            for duree, ligne in par_duree.iterrows()
        ],
        "figure": figure,
    }


def evaluate() -> dict:
    """Execute les cinq analyses sur le test set."""
    predictor = joblib.load(MODEL_PATH)
    test = load_split()["test"]
    X_test, y_series = split_xy(test)
    y = y_series.to_numpy(dtype=float)

    logger.info("Evaluation finale sur le TEST : %d patients", len(y))
    logger.info("(jamais utilise a l'entrainement ni a la calibration)")

    _, low, high = predictor.predict(X_test, alpha=ALPHA)

    logger.info("1/5 couverture globale et largeur...")
    global_ = plot_coverage_and_width(y, low, high, ALPHA)

    logger.info("2/5 couverture par sous-groupe...")
    sousgroupes = plot_subgroup_coverage(test, y, low, high, ALPHA)

    logger.info("3/5 couverture vs alpha...")
    vs_alpha = plot_coverage_vs_alpha(predictor, X_test, y)

    logger.info("4/5 calibre vs non calibre...")
    comparaison = plot_calibrated_vs_raw(predictor, X_test, y, ALPHA)

    logger.info("5/5 adaptativite...")
    adaptativite = plot_width_vs_truth(y, low, high)

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alpha": ALPHA,
        "global": global_,
        "sous_groupes": sousgroupes,
        "couverture_vs_alpha": vs_alpha,
        "calibre_vs_non_calibre": comparaison,
        "adaptativite": adaptativite,
    }


def main() -> None:
    rapport = evaluate()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_PATH.write_text(json.dumps(rapport, indent=2), encoding="utf-8")

    g = rapport["global"]
    logger.info("")
    logger.info("=== RESULTAT PRINCIPAL ===")
    logger.info("Couverture visee   : %.4f", g["target_coverage"])
    logger.info(
        "Couverture mesuree : %.4f  (%+.2f erreurs types)",
        g["empirical_coverage"],
        g["ecarts_types_a_la_cible"],
    )
    logger.info("Largeur moyenne    : %.2f jours", g["mean_width"])
    logger.info("Rapport : %s", EVALUATION_REPORT_PATH)


if __name__ == "__main__":
    main()
