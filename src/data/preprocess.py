"""Nettoyage et feature engineering du dataset UCI Diabetes 130-US Hospitals.

Deux blocs :
  1. `group_icd9`  : ~800 codes ICD-9 -> 9 categories cliniques larges
  2. `preprocess`  : nettoyage, exclusions, features derivees, encodage LightGBM

Toutes les regles appliquees ici sont DETERMINISTES et ne dependent d'aucune
statistique apprise sur les donnees (pas de moyenne, pas de frequence). Elles
peuvent donc etre appliquees avant le split sans fuite : voir la note en bas de
fichier pour ce qui devra, lui, etre fitte sur le train uniquement.

Usage : python -m src.data.preprocess
"""

import logging

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.load import DATA_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "time_in_hospital"

# --- Colonnes ecartees ------------------------------------------------------

# Identifiants : aucune valeur predictive, et les garder ferait fuiter l'identite
# du patient dans le modele.
ID_COLS = ["encounter_id", "patient_nbr"]

# 96,9 % de manquants : rien a en tirer (constate en EDA).
MOSTLY_MISSING = ["weight"]

# Constantes sur tout le dataset (toujours "No") : variance nulle, zero information.
CONSTANT_COLS = ["examide", "citoglipton"]

# --- REGLE TEMPORELLE : ne garder que ce qui est connu A L'ADMISSION ---------
#
# Le modele doit servir a anticiper l'occupation des lits AU MOMENT ou le
# patient arrive. Toute variable renseignee plus tard est inutilisable en
# pratique, meme si elle ameliore les metriques : au moment de la prediction
# elle n'existe pas encore.

# Connues seulement APRES la sortie. Fuite non discutable.
#   - discharge_disposition_id : mode de sortie du patient
#   - readmitted : ce qui s'est passe APRES ce sejour (cible d'origine du
#     dataset UCI, sans rapport avec notre tache)
POST_DISCHARGE_COLS = ["discharge_disposition_id", "readmitted"]

# Mesurees PENDANT le sejour. Ce sont les variables les plus correlees a la
# cible (num_medications r=+0,47 ; num_lab_procedures r=+0,33), mais la
# correlation est en grande partie inverse : un sejour long genere davantage de
# prescriptions et d'analyses. Elles refletent la duree plus qu'elles ne la
# predisent. Les garder donnerait de meilleures metriques et un modele
# inutilisable a l'admission.
DURING_STAY_COLS = [
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    # Posologie ajustee pendant le sejour ("Up"/"Down") : decision therapeutique
    # prise apres l'admission. Nuance a connaitre : le fait qu'un patient soit
    # DEJA sous traitement est, lui, connu a l'admission (conciliation
    # medicamenteuse). Mais l'encodage du dataset confond traitement anterieur
    # et instauration pendant le sejour -- indemelable, donc ecarte en bloc.
    "change",
    "diabetesMed",
]

# --- Exclusions de lignes ---------------------------------------------------

# Codes discharge_disposition_id correspondant a un deces ou a des soins
# palliatifs (IDS_mapping) : 11=Expired, 13/14=Hospice, 19/20/21=Expired.
# Pour ces sejours la duree n'est pas determinee par la guerison mais par le
# deces : la cible ne mesure pas le meme phenomene. On les exclut (2,4 % des
# lignes) plutot que d'apprendre a predire une duree censuree par la mort.
DEATH_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

# --- Traitement des manquants informatifs -----------------------------------

# Un test non effectue n'est pas une donnee manquante : c'est la decision du
# medecin de ne pas doser, ce qui est en soi un signal clinique. On la nomme.
NOT_MEASURED_COLS = ["max_glu_serum", "A1Cresult"]

# Manquants remplaces par une modalite explicite plutot qu'imputes : l'absence
# de valeur est elle-meme porteuse d'information (cf. medical_specialty).
UNKNOWN_COLS = ["medical_specialty", "payer_code", "race"]

# Les 23 colonnes de medicaments, toutes a valeurs {No, Steady, Up, Down}.
DRUG_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Identifiants administratifs codes en entiers : ce sont des CATEGORIES, pas des
# quantites. Sans conversion explicite, LightGBM les traiterait comme numeriques
# et pourrait inferer que "type 3 > type 1", ce qui n'a aucun sens.
# discharge_disposition_id n'y figure pas : il sert au filtrage des deces
# (etape 2) puis est supprime, car inconnu a l'admission.
ID_CATEGORICAL_COLS = [
    "admission_type_id",
    "admission_source_id",
]


def group_icd9(code: object) -> str:
    """Regroupe un code ICD-9 en categorie clinique large."""
    # Manquant : categorie dediee, jamais imputee.
    if pd.isna(code):
        return "Missing"

    code = str(code).strip()

    # PIEGE : 1 645 codes commencent par V (facteurs influencant l'etat de sante)
    # ou E (causes externes de traumatisme). Ils ne sont PAS numeriques : tout
    # float(code) plante ici. Ils forment leur propre categorie.
    if code.startswith(("V", "E")):
        return "Other"

    try:
        # float() et non int() : les codes portent des sous-divisions decimales
        # (250.83 = diabete avec complications). On tronque a la racine entiere.
        value = float(code)
    except ValueError:
        return "Other"

    # Le diabete (250.x) est teste EN PREMIER : il tombe sinon dans la plage
    # endocrinienne 240-279 et se noierait dans "Other". C'est la pathologie
    # centrale de cette cohorte, elle merite sa propre categorie.
    if 250 <= value < 251:
        return "Diabetes"

    # Plages ICD-9 standard, regroupement usuel dans la litterature sur ce dataset.
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 140 <= value <= 239:
        return "Neoplasms"

    return "Other"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le nettoyage complet et renvoie le DataFrame pret a modeliser."""
    n_start = len(df)

    # --- 1. Manquants encodes en "?" --------------------------------------
    # Le CSV brut n'utilise pas de champ vide mais un point d'interrogation.
    df = df.replace("?", pd.NA)

    # --- 2. Exclusions de lignes ------------------------------------------
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_CODES)]
    logger.info("Deces / soins palliatifs exclus : %d lignes", n_start - len(df))

    # 3 lignes seulement, mais une modalite a 3 individus est du bruit pur.
    n = len(df)
    df = df[df["gender"] != "Unknown/Invalid"]
    logger.info("Genre invalide exclu : %d lignes", n - len(df))

    # --- 3. Une seule admission par patient -------------------------------
    # LE point methodologique du projet. La prediction conforme suppose des
    # observations echangeables, donc independantes. Un meme patient present a
    # la fois dans le train et dans le set de calibration rendrait la couverture
    # mesuree optimiste : le modele l'aurait deja vu. On garde la PREMIERE
    # admission (les suivantes sont conditionnees par les precedentes).
    # Cout : ~30 % des lignes. Non negociable.
    n = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    logger.info("Re-admissions ecartees : %d lignes", n - len(df))

    # --- 4. Features derivees ---------------------------------------------
    # Seules des features construites a partir d'information disponible A
    # L'ADMISSION sont autorisees (cf. regle temporelle en haut de fichier).

    # Historique de recours aux soins sur l'annee precedente, toutes voies
    # confondues. Anterieur au sejour, donc pleinement utilisable.
    df["n_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # Le patient a-t-il deja ete hospitalise dans l'annee ? Une hospitalisation
    # anterieure est un marqueur de fragilite plus net que le simple comptage.
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

    # Nombre de diagnostics rapporte au recours anterieur : distingue un patient
    # polypathologique connu du systeme d'un patient polypathologique decouvert.
    df["diagnoses_per_prior_visit"] = df["number_diagnoses"] / (
        df["n_prior_visits"] + 1
    )

    # --- 5. Regroupement des diagnostics ICD-9 ----------------------------
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(group_icd9)

    # --- 6. Manquants -> modalites explicites -----------------------------
    for col in NOT_MEASURED_COLS:
        # "None" dans ce dataset signifie deja "test non effectue".
        df[col] = df[col].fillna("not_measured").replace("None", "not_measured")

    for col in UNKNOWN_COLS:
        df[col] = df[col].fillna("Unknown")

    # --- 7. Colonnes ecartees ---------------------------------------------
    # DRUG_COLS part en bloc : ces 21 colonnes encodent des ajustements de
    # posologie decides pendant le sejour (cf. commentaire de DURING_STAY_COLS).
    df = df.drop(
        columns=ID_COLS
        + MOSTLY_MISSING
        + CONSTANT_COLS
        + POST_DISCHARGE_COLS
        + DURING_STAY_COLS
        + DRUG_COLS
    )

    # --- 8. Encodage categoriel natif LightGBM ----------------------------
    # dtype "category" plutot qu'un one-hot : LightGBM gere nativement les
    # categorielles par partitionnement optimal des modalites. Un one-hot sur
    # medical_specialty (72 modalites) creerait 72 colonnes creuses et
    # degraderait la qualite des splits.
    for col in ID_CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    logger.info(
        "Sortie : %d lignes x %d colonnes (%.1f %% du brut)",
        len(df),
        df.shape[1],
        len(df) / n_start * 100,
    )
    return df.reset_index(drop=True)


def main() -> None:
    """Charge le CSV brut, applique le preprocessing et sauvegarde le resultat."""
    raw = pd.read_csv(RAW_DATA_DIR / DATA_FILE, low_memory=False)
    df = preprocess(raw)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / "clean.csv"
    df.to_csv(out, index=False)
    logger.info("Ecrit : %s", out)


if __name__ == "__main__":
    main()


# NOTE SUR LA FUITE DE DONNEES
# ---------------------------
# Tout ce qui precede est deterministe : chaque regle (seuils ICD-9, listes de
# colonnes, remplissage par une constante) est fixee a l'avance et ne lit aucune
# statistique du jeu de donnees. L'appliquer avant le split ne fuit donc rien.
#
# En revanche, toute transformation APPRISE sur les donnees -- imputation par la
# moyenne, standardisation, target encoding, selection de features par
# correlation -- devra etre fittee sur le TRAIN seul, puis appliquee telle quelle
# aux sets de calibration et de test (contrainte 3 de CLAUDE.md).
