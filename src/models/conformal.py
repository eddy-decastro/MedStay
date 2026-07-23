"""Predicteur conforme servi par l'API : intervalles a niveau de confiance libre.

POURQUOI CETTE CLASSE PLUTOT QUE L'OBJET MAPIE DIRECTEMENT

MapieQuantileRegressor fige alpha a l'entrainement : `predict(alpha=0.30)`
renvoie exactement le meme intervalle que `predict(alpha=0.10)`. Or la SPEC
demande un curseur de confiance, et c'est le moment fort de la demo.

La theorie permet de faire mieux. Le score de conformite CQR vaut

    E_i = max( q_bas(X_i) - Y_i ,  Y_i - q_haut(X_i) )

mesure sur le set de calibration. Pour un niveau 1 - alpha quelconque, la
correction est le quantile d'ordre ceil((n+1)(1-alpha))/n de ces scores, et
l'intervalle devient [q_bas - c, q_haut + c]. Cette construction est valide
POUR TOUT alpha avec les MEMES modeles quantiles : la garantie de couverture
tient a chaque niveau, seule l'efficacite (la largeur) se degrade quand alpha
s'eloigne des quantiles appris (0,05 / 0,95).

Le facteur (n+1) est la correction d'echantillon fini : c'est elle qui rend la
garantie valable a n fini, et non seulement asymptotiquement.

Verifie : a alpha = 0,10 cette formule reproduit la correction de MAPIE a
5e-4 pres (0,014345 contre 0,013800), l'ecart provenant de la convention
d'interpolation du quantile.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Bornes physiologiques : un sejour dure de 1 a 14 jours dans ce dataset.
TARGET_MIN = 1.0
TARGET_MAX = 14.0

# Bornes admises pour alpha, exposees par l'API (SPEC section 6).
ALPHA_MIN = 0.01
ALPHA_MAX = 0.50


@dataclass
class FeatureSpec:
    """Schema exact attendu par les modeles.

    Indispensable au service : l'API recoit du JSON, et un DataFrame reconstruit
    naivement n'aurait ni le bon ordre de colonnes, ni les bons dtypes, ni les
    memes categories qu'a l'entrainement. LightGBM produirait alors des
    predictions silencieusement fausses.
    """

    columns: list[str]
    categories: dict[str, list[str]]
    numeric: list[str]

    def build_frame(self, records: list[dict]) -> pd.DataFrame:
        """Construit un DataFrame conforme au schema d'entrainement."""
        df = pd.DataFrame(records)

        manquantes = set(self.columns) - set(df.columns)
        if manquantes:
            raise ValueError(f"Colonnes manquantes : {sorted(manquantes)}")

        # Reordonne : LightGBM se fie a la POSITION des colonnes.
        df = df[self.columns].copy()

        for col, cats in self.categories.items():
            # Normalisation en chaines DES DEUX COTES. Indispensable : certaines
            # colonnes categorielles ont des modalites entieres
            # (admission_type_id, admission_source_id). Sans cette conversion,
            # l'entier 1 ne correspond pas a la categorie "1" et la valeur
            # devient NaN -- sans erreur, avec des predictions silencieusement
            # fausses. Bug reellement rencontre : 4,45 jours predits contre 4,77.
            # Cela rend aussi l'API tolerante au JSON, qui peut livrer 1 ou "1".
            valeurs = df[col].map(lambda v: None if pd.isna(v) else str(v))

            # Categories figees a celles du train. Une modalite inconnue devient
            # NaN, que LightGBM sait traiter -- plutot qu'un decalage d'encodage.
            df[col] = pd.Categorical(valeurs, categories=cats)

        for col in self.numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def postprocess(
    point: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rend les sorties coherentes : bornes dans [1, 14] et point dedans.

    Deux defauts mesures sur les sorties brutes des modeles :

    1. 57,9 % des bornes basses tombaient sous 1 jour et 0,16 % des hautes
       au-dessus de 14, alors que la cible est bornee par construction.
       Tronquer NE PEUT JAMAIS FAIRE BAISSER LA COUVERTURE : la vraie valeur
       etant toujours dans [1, 14], si elle appartenait a l'intervalle
       d'origine elle appartient encore a l'intervalle tronque. La garantie
       conforme est donc preservee, avec des intervalles plus serres.

       Cas limite : un intervalle situe ENTIEREMENT hors de [1, 14] devient
       degenere sur la borne et peut se mettre a couvrir. La couverture peut
       donc AUGMENTER -- la propriete exacte est une inegalite, pas une
       egalite. Sur ce modele l'egalite est observee (0,928294 avant et
       apres), aucun intervalle ne tombant entierement hors bornes.

    2. 0,3 % des estimations ponctuelles sortaient de leur propre intervalle :
       les trois quantiles sont appris independamment et la correction conforme
       decale les bornes sans decaler la mediane. Annoncer
       "0,86 jour, intervalle [0,99 ; 5,70]" serait incoherent.

    Cette fonction vit dans conformal.py et NON dans calibrate.py : ce dernier
    importe mapie, absent de l'image de production. L'API doit pouvoir
    l'utiliser sans tirer les dependances d'entrainement.
    """
    low = np.clip(low, TARGET_MIN, TARGET_MAX)
    high = np.clip(high, TARGET_MIN, TARGET_MAX)

    # Garde-fou : si les quantiles se croisent, on retablit l'ordre plutot que
    # de renvoyer un intervalle vide.
    low, high = np.minimum(low, high), np.maximum(low, high)

    point = np.clip(point, low, high)
    return point, low, high


class ConformalPredictor:
    """Intervalles de prediction conformes, a niveau de confiance parametrable."""

    def __init__(
        self,
        model_low,
        model_high,
        model_median,
        conformity_scores: np.ndarray,
        feature_spec: FeatureSpec,
        metadata: dict,
    ) -> None:
        self.model_low = model_low
        self.model_high = model_high
        self.model_median = model_median
        # Scores CQR mesures sur le set de calibration, jamais sur le train.
        self.conformity_scores = np.asarray(conformity_scores, dtype=float)
        self.feature_spec = feature_spec
        self.metadata = metadata

    @property
    def n_calibration(self) -> int:
        return len(self.conformity_scores)

    def correction(self, alpha: float) -> float:
        """Marge conforme a appliquer aux quantiles pour un niveau 1 - alpha."""
        if not ALPHA_MIN <= alpha <= ALPHA_MAX:
            raise ValueError(
                f"alpha doit etre dans [{ALPHA_MIN}, {ALPHA_MAX}], recu {alpha}"
            )

        n = self.n_calibration
        # Correction d'echantillon fini : rang ceil((n+1)(1-alpha)) parmi n.
        # Le min(..., 1.0) evite de depasser le quantile maximal quand n est
        # petit ou alpha tres proche de 0.
        niveau = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(np.quantile(self.conformity_scores, niveau, method="higher"))

    def predict(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Renvoie (estimation ponctuelle, borne basse, borne haute)."""
        marge = self.correction(alpha)

        point = np.asarray(self.model_median.predict(X), dtype=float)
        low = np.asarray(self.model_low.predict(X), dtype=float) - marge
        high = np.asarray(self.model_high.predict(X), dtype=float) + marge

        return postprocess(point, low, high)
