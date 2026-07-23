"""Interface Streamlit de MedStay-CI.

REGLE ABSOLUE (contrainte 4 de CLAUDE.md) : ce fichier ne charge JAMAIS le
modele. Il ne parle a l'API qu'en HTTP. Le decouplage garantit qu'on pourrait
remplacer le front par une application mobile, ou l'API par un autre modele,
sans toucher a l'autre moitie.
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# En local comme dans le conteneur Render, l'API tourne sur le port 8000 de la
# meme machine : start.sh lance les deux processus cote a cote.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

st.set_page_config(page_title="MedStay-CI", page_icon="🏥", layout="wide")


# --- Acces a l'API ----------------------------------------------------------


@st.cache_data(ttl=300)
def get_categories() -> dict:
    """Modalites acceptees par le modele, pour peupler les menus deroulants.

    Mise en cache 5 minutes : ces listes ne changent qu'au redeploiement, il
    serait absurde de les redemander a chaque interaction.
    """
    return httpx.get(f"{API_URL}/categories", timeout=30).json()


@st.cache_data(ttl=300)
def get_model_info() -> dict:
    return httpx.get(f"{API_URL}/model-info", timeout=30).json()


def predict(patient: dict, alpha: float) -> dict:
    reponse = httpx.post(
        f"{API_URL}/predict", json=patient, params={"alpha": alpha}, timeout=30
    )
    reponse.raise_for_status()
    return reponse.json()


# --- Graphique de l'intervalle ----------------------------------------------


def afficher_intervalle(resultat: dict) -> None:
    """Represente l'intervalle sur un axe de 1 a 14 jours.

    Un graphique matplotlib serait plus riche, mais matplotlib n'est pas dans
    l'image de production : on dessine en HTML/CSS, ce qui est aussi plus leger
    et plus net a l'affichage.
    """
    bas, haut = resultat["lower_bound"], resultat["upper_bound"]
    point = resultat["point_estimate"]

    # Conversion en pourcentages de largeur sur une echelle de 1 a 14 jours.
    en_pct = lambda jours: (jours - 1) / 13 * 100  # noqa: E731

    st.markdown(
        f"""
        <div style="position:relative;height:90px;margin:24px 0 8px 0;">
          <div style="position:absolute;top:38px;left:0;right:0;height:6px;
                      background:#e5e7eb;border-radius:3px;"></div>
          <div style="position:absolute;top:38px;left:{en_pct(bas)}%;
                      width:{en_pct(haut) - en_pct(bas)}%;height:6px;
                      background:#2563eb;border-radius:3px;opacity:.35;"></div>
          <div style="position:absolute;top:30px;left:calc({en_pct(point)}% - 7px);
                      width:14px;height:22px;background:#2563eb;
                      border-radius:3px;"></div>
          <div style="position:absolute;top:58px;left:{en_pct(bas)}%;
                      font-size:12px;color:#6b7280;">{bas:.1f} j</div>
          <div style="position:absolute;top:58px;left:{en_pct(haut)}%;
                      font-size:12px;color:#6b7280;">{haut:.1f} j</div>
          <div style="position:absolute;top:6px;left:calc({en_pct(point)}% - 20px);
                      font-size:13px;font-weight:600;color:#2563eb;">
                      {point:.1f} j</div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:11px;color:#9ca3af;">
          <span>1 jour</span><span>7 jours</span><span>14 jours</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Onglet 1 : prediction ---------------------------------------------------


def onglet_prediction() -> None:
    st.subheader("Prédire la durée de séjour d'un patient")

    try:
        categories = get_categories()
    except Exception as erreur:  # noqa: BLE001 - on affiche l'erreur a l'ecran
        st.error(f"API injoignable : {erreur}")
        st.info("Le service gratuit s'endort après 15 min. Rechargez dans ~40 s.")
        return

    colonne_formulaire, colonne_resultat = st.columns([1, 1])

    with colonne_formulaire:
        st.markdown("**Profil du patient**")
        c1, c2 = st.columns(2)
        with c1:
            age = st.selectbox("Âge", categories["age"], index=7)
            gender = st.selectbox("Sexe", categories["gender"])
            race = st.selectbox("Origine déclarée", categories["race"])
        with c2:
            admission_type = st.selectbox(
                "Type d'admission", categories["admission_type_id"]
            )
            admission_source = st.selectbox(
                "Origine de l'admission", categories["admission_source_id"], index=10
            )
            payer_code = st.selectbox("Assurance", categories["payer_code"])

        medical_specialty = st.selectbox(
            "Spécialité du médecin admetteur",
            categories["medical_specialty"],
            index=categories["medical_specialty"].index("InternalMedicine")
            if "InternalMedicine" in categories["medical_specialty"]
            else 0,
        )

        st.markdown("**Diagnostics** (catégories cliniques)")
        d1, d2, d3 = st.columns(3)
        diag_1 = d1.selectbox("Principal", categories["diag_1"])
        diag_2 = d2.selectbox("Secondaire", categories["diag_2"], index=1)
        diag_3 = d3.selectbox("Tertiaire", categories["diag_3"])

        st.markdown("**Biologie**")
        b1, b2 = st.columns(2)
        max_glu = b1.selectbox("Glycémie sérique", categories["max_glu_serum"], index=3)
        a1c = b2.selectbox("HbA1c", categories["A1Cresult"], index=3)

        st.markdown("**Antécédents sur l'année écoulée**")
        h1, h2, h3, h4 = st.columns(4)
        n_out = h1.number_input("Consultations", 0, 50, 0)
        n_emg = h2.number_input("Urgences", 0, 50, 1)
        n_inp = h3.number_input("Hospitalisations", 0, 25, 2)
        n_diag = h4.number_input("Diagnostics", 1, 16, 9)

    with colonne_resultat:
        st.markdown("**Niveau de confiance**")
        # LE moment fort de la demo : deplacer ce curseur elargit l'intervalle
        # en direct, ce qui rend tangible le compromis certitude / precision.
        confiance = st.slider(
            "Couverture visée",
            50,
            99,
            90,
            step=1,
            format="%d %%",
            help="Plus la couverture exigée est élevée, plus l'intervalle s'élargit.",
        )
        alpha = round(1 - confiance / 100, 4)

        patient = {
            "race": race,
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type,
            "admission_source_id": admission_source,
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "max_glu_serum": max_glu,
            "A1Cresult": a1c,
            "number_outpatient": n_out,
            "number_emergency": n_emg,
            "number_inpatient": n_inp,
            "number_diagnoses": n_diag,
        }

        try:
            resultat = predict(patient, alpha)
        except Exception as erreur:  # noqa: BLE001
            st.error(f"Erreur de prédiction : {erreur}")
            return

        st.markdown("### Résultat")
        m1, m2, m3 = st.columns(3)
        m1.metric("Durée probable", f"{resultat['point_estimate']:.1f} j")
        m2.metric(
            "Intervalle",
            f"{resultat['lower_bound']:.1f} – {resultat['upper_bound']:.1f} j",
        )
        m3.metric("Largeur", f"{resultat['interval_width']:.1f} j")

        afficher_intervalle(resultat)

        st.info(
            f"Sur 100 patients de ce profil, environ **{confiance}** "
            f"séjourneront entre **{resultat['lower_bound']:.1f}** et "
            f"**{resultat['upper_bound']:.1f} jours**."
        )
        st.caption(
            "Garantie issue de la prédiction conforme : elle ne repose sur "
            "aucune hypothèse de loi de probabilité, seulement sur "
            "l'échangeabilité des observations."
        )


# --- Onglet 2 : performance --------------------------------------------------


def onglet_performance() -> None:
    st.subheader("Validation statistique")

    try:
        info = get_model_info()
    except Exception as erreur:  # noqa: BLE001
        st.error(f"API injoignable : {erreur}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Couverture visée", f"{info['target_coverage']:.1%}")
    mesuree = info.get("measured_test_coverage")
    c2.metric(
        "Couverture mesurée",
        f"{mesuree:.2%}" if mesuree else "n/a",
        delta=f"{(mesuree - info['target_coverage']) * 100:+.2f} pts"
        if mesuree
        else None,
    )
    c3.metric("Largeur moyenne", f"{info.get('mean_interval_width', 0):.2f} j")
    c4.metric("Patients de test", f"{info['n_calibration']:,}".replace(",", " "))

    st.caption(
        f"Entraîné sur {info['n_train']:,} patients, calibré sur "
        f"{info['n_calibration']:,}, évalué sur un jeu de test jamais vu. "
        f"{info['n_features']} variables, toutes connues à l'admission.".replace(
            ",", " "
        )
    )

    figures = [
        (
            "01_couverture_et_largeur.png",
            "Couverture empirique et distribution des largeurs",
            "La couverture mesurée tombe dans la bande de fluctuation attendue "
            "(± 2 erreurs types) : la promesse de 90 % est tenue.",
        ),
        (
            "04_calibre_vs_non_calibre.png",
            "Ce qu'apporte la calibration conforme",
            "Sans calibration, les quantiles bruts ne couvrent que 87,4 %. La "
            "correction conforme ramène à 90,2 % pour une largeur quasi "
            "identique : la garantie ne coûte presque rien.",
        ),
        (
            "03_couverture_vs_alpha.png",
            "Calibration à tous les niveaux de confiance",
            "La courbe suit la diagonale de 50 % à 99 %. La calibration n'est "
            "donc pas un ajustement chanceux au seul niveau de 90 %.",
        ),
        (
            "02_couverture_par_sous_groupe.png",
            "Couverture par sous-groupe",
            "Limite connue : la prédiction conforme ne garantit que la couverture "
            "MARGINALE. Certains sous-groupes descendent à 86,4 %.",
        ),
        (
            "05_adaptativite.png",
            "Adaptativité des intervalles",
            "L'intervalle s'élargit là où le modèle est le moins sûr, au lieu "
            "d'appliquer une largeur constante.",
        ),
    ]

    for nom, titre, commentaire in figures:
        chemin = FIGURES_DIR / nom
        if chemin.exists():
            st.markdown(f"#### {titre}")
            st.image(str(chemin), use_container_width=True)
            st.caption(commentaire)
            st.divider()


# --- Onglet 3 : a propos ------------------------------------------------------


def onglet_a_propos() -> None:
    st.subheader("Comment ça marche")

    st.markdown(
        """
### La prédiction conforme en cinq lignes

1. Deux modèles apprennent des **bornes** de durée, pas une valeur unique.
2. Ces bornes sont systématiquement **trop optimistes** : elles sont ajustées
   sur les données d'entraînement.
3. On mesure leur erreur réelle sur un jeu **jamais vu à l'entraînement**.
4. On les élargit exactement de ce qu'il faut pour atteindre 90 % de couverture.
5. La garantie qui en résulte ne suppose **aucune loi de probabilité** —
   seulement que les patients futurs ressemblent aux patients passés.

### Pourquoi un intervalle plutôt qu'un chiffre

Annoncer « 4,2 jours » suggère une précision que le modèle n'a pas. Deux
patients identiques sur le papier peuvent séjourner 2 ou 9 jours selon des
événements imprévisibles à l'admission. L'intervalle rend cette incertitude
visible — et, surtout, il s'élargit tout seul quand le cas est difficile.

### Limites connues

- **Données de 1999 à 2008**, hôpitaux américains. Les pratiques ont changé.
- **Couverture marginale seulement.** Les 90 % valent en moyenne sur toute la
  population, pas dans chaque sous-groupe : certains descendent à 86,4 %.
- **Aucune validation clinique.** Projet pédagogique, jamais éprouvé en
  conditions réelles.
- **Échangeabilité discutable** entre 130 hôpitaux aux pratiques hétérogènes.
- **Variables limitées à l'admission**, par choix. Les variables mesurées
  pendant le séjour amélioreraient les métriques, mais donneraient un modèle
  inutilisable au moment où la décision se prend.
- **Service gratuit** : mise en veille après 15 min, réveil en 30 à 50 s.
"""
    )

    try:
        info = get_model_info()
        with st.expander("Détails techniques"):
            st.json(
                {
                    "méthode": info["method"],
                    "entraîné le": info["trained_at"],
                    "couverture visée": info["target_coverage"],
                    "couverture mesurée sur le test": info["measured_test_coverage"],
                    "versions": info["versions"],
                }
            )
            st.markdown("**Variables utilisées**")
            st.dataframe(
                pd.DataFrame({"variable": info["features"]}),
                use_container_width=True,
                hide_index=True,
            )
    except Exception:  # noqa: BLE001 - onglet informatif, l'API peut dormir
        st.caption("Détails techniques indisponibles (API endormie).")


# --- Assemblage ---------------------------------------------------------------

st.title("🏥 MedStay-CI")
st.caption(
    "Durée de séjour hospitalier avec intervalle garanti à 90 % — "
    "prédiction conforme (CQR)"
)

onglet1, onglet2, onglet3 = st.tabs(["Prédiction", "Performance", "À propos"])
with onglet1:
    onglet_prediction()
with onglet2:
    onglet_performance()
with onglet3:
    onglet_a_propos()
