import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diamants • Naive Bayes",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diamonds.csv")
MODELS_DIR = BASE_DIR

COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]

PALETTE = {
    "Fair": "#E4572E",
    "Good": "#F3A712",
    "Very Good": "#A8C256",
    "Premium": "#5B9BD5",
    "Ideal": "#7C4DFF",
}

# ----------------------------------------------------------------------------
# STYLE CSS PERSONNALISE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
    }

    .hero {
        background: linear-gradient(120deg, #1a1440 0%, #3d2b6b 45%, #7c4dff 100%);
        padding: 2.6rem 2.2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 34px rgba(90, 50, 180, 0.35);
    }
    .hero h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: white;
    }
    .hero p {
        font-size: 1.02rem;
        color: #E4DBFF;
        max-width: 780px;
        line-height: 1.55;
    }
    .badge {
        display: inline-block;
        padding: 0.28rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.35);
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 18px rgba(30, 20, 70, 0.08);
        border: 1px solid #EFE9FF;
        text-align: left;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #6b6382;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #241C4F;
        font-family: 'Poppins', sans-serif;
    }
    .metric-card .sub {
        font-size: 0.85rem;
        color: #8b839f;
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #241C4F;
        border-left: 5px solid #7C4DFF;
        padding-left: 0.7rem;
        margin: 1.4rem 0 0.9rem 0;
    }

    .pill-best {
        background: linear-gradient(90deg, #7C4DFF, #5B9BD5);
        color: white;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .result-box {
        background: linear-gradient(135deg, #241C4F, #4b2e8f);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(60, 20, 120, 0.25);
    }
    .result-box .cut-name {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }

    section[data-testid="stSidebar"] {
        background: #14102E;
    }
    section[data-testid="stSidebar"] * {
        color: #EDE7FF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNEES ET DES MODELES (produits par les notebooks)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)].drop_duplicates()
    df["color_enc"] = df["color"].map({c: i for i, c in enumerate(COLOR_ORDER)})
    df["clarity_enc"] = df["clarity"].map({c: i for i, c in enumerate(CLARITY_ORDER)})
    df["cut_enc"] = df["cut"].map({c: i for i, c in enumerate(CUT_ORDER)})
    return df


@st.cache_resource
def load_models():
    artefacts = {}
    for key in ["underfit", "overfit"]:
        model_path = os.path.join(MODELS_DIR, f"nb_{key}.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{key}.pkl")
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{key}.json")
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            return None
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        artefacts[key] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
            "metrics": metrics,
        }
    # Modele optimise (issu de la recherche d'hyperparametres GridSearchCV), s'il a ete genere
    opt_model_path = os.path.join(MODELS_DIR, "nb_optimise.pkl")
    opt_scaler_path = os.path.join(MODELS_DIR, "scaler_optimise.pkl")
    opt_metrics_path = os.path.join(MODELS_DIR, "metrics_optimise.json")
    if os.path.exists(opt_model_path) and os.path.exists(opt_metrics_path):
        with open(opt_metrics_path, "r", encoding="utf-8") as f:
            opt_metrics = json.load(f)
        artefacts["optimise"] = {
            "model": joblib.load(opt_model_path),
            "scaler": joblib.load(opt_scaler_path),
            "metrics": opt_metrics,
        }

    selection_path = os.path.join(MODELS_DIR, "selection_modele.json")
    if os.path.exists(selection_path):
        with open(selection_path, "r", encoding="utf-8") as f:
            artefacts["selection"] = json.load(f)
    else:
        best = "overfit" if artefacts["overfit"]["metrics"]["accuracy_test"] >= artefacts["underfit"]["metrics"]["accuracy_test"] else "underfit"
        artefacts["selection"] = {"meilleur_modele": best}
    return artefacts


df = load_data()
artefacts = load_models()

if artefacts is None:
    st.error(
        "⚠️ Les modèles n'ont pas été trouvés dans le dossier `models/`. "
        "Merci d'exécuter d'abord les notebooks `01_naive_bayes_underfit.ipynb` "
        "et `02_naive_bayes_overfit.ipynb` (ils génèrent les fichiers .pkl et .json nécessaires)."
    )
    st.stop()

BEST_KEY = artefacts["selection"]["meilleur_modele"]        # meilleur entre underfit / overfit (demo pedagogique)
BEST = artefacts[BEST_KEY]
OTHER_KEY = "underfit" if BEST_KEY == "overfit" else "overfit"
OTHER = artefacts[OTHER_KEY]

# Modele reellement utilise pour la prediction : l'optimise (GridSearchCV) s'il existe, sinon le "meilleur" brut
HAS_OPTIMISE = "optimise" in artefacts
PROD_KEY = artefacts["selection"].get("modele_production", BEST_KEY) if HAS_OPTIMISE else BEST_KEY
PROD = artefacts[PROD_KEY]


# ----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES METIER
# ----------------------------------------------------------------------------
def build_engineered_row(carat, color_enc, clarity_enc, depth, table, price, x, y, z):
    """Construit toutes les variables (brutes + ingenierie) a partir d'une saisie utilisateur."""
    row = {
        "carat": carat,
        "depth": depth,
        "table": table,
        "price": price,
        "x": x,
        "y": y,
        "z": z,
        "color_enc": color_enc,
        "clarity_enc": clarity_enc,
    }
    row["volume"] = x * y * z
    row["price_per_carat"] = price / carat if carat else 0.0
    row["carat2"] = carat ** 2
    row["depth_table"] = depth * table
    row["carat_clarity"] = carat * clarity_enc
    row["carat_color"] = carat * color_enc
    return row


def predict_with(artefact, row_dict):
    features = artefact["metrics"]["features"]
    X = pd.DataFrame([{f: row_dict[f] for f in features}])
    X_sc = artefact["scaler"].transform(X)
    proba = artefact["model"].predict_proba(X_sc)[0]
    pred_idx = int(np.argmax(proba))
    return pred_idx, proba, X_sc, features


def nb_feature_log_contributions(model, x_scaled_row, class_idx):
    means = model.theta_[class_idx]
    variances = model.var_[class_idx]
    log_p = -0.5 * np.log(2 * np.pi * variances) - 0.5 * ((x_scaled_row - means) ** 2) / variances
    return log_p


# ----------------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 💎 Navigation")
page = st.sidebar.radio(
    "",
    [
        "🏠 Accueil",
        "📊 Exploration des données",
        "⚖️ Comparaison des modèles",
        "🔮 Prédiction",
        "🧠 Explicabilité",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Modèle retenu pour la prédiction**
    <span class="pill-best">{PROD_KEY.upper()}</span>

    Accuracy test : **{PROD['metrics']['accuracy_test']:.1%}**
    """,
    unsafe_allow_html=True,
)
if HAS_OPTIMISE:
    st.sidebar.caption(f"Optimisé par GridSearchCV à partir du modèle {BEST_KEY} (var_smoothing = {PROD['metrics']['var_smoothing']:.2e})")
st.sidebar.markdown("---")


# ----------------------------------------------------------------------------
# PAGE 1 — ACCUEIL
# ----------------------------------------------------------------------------
if page == "🏠 Accueil":
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">🎓 Master 1 — Statistique & Analyse de Données</span>
            <span class="badge">💎 Dataset Diamonds</span>
            <span class="badge">🧮 Algorithme imposé : Naïve Bayes</span>
            <h1>Prédiction de la qualité de taille des diamants (cut)</h1>
            <p>
            Cette application compare deux versions volontairement construites d'un même modèle
            <b>Naïve Bayes Gaussien</b> — une version <b>sous-ajustée (underfit)</b> et une version
            <b>sur-ajustée (overfit)</b> — puis utilise la meilleure des deux (selon l'accuracy sur
            le jeu de test) pour prédire la qualité de taille (<i>cut</i>) d'un diamant à partir
            de ses caractéristiques physiques.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Diamants (dataset nettoyé)", f"{len(df):,}".replace(",", " "), "après suppression des valeurs aberrantes"),
        (c2, "Modèle en production", PROD_KEY.upper(), f"basé sur {BEST_KEY}" if HAS_OPTIMISE else "meilleure accuracy test"),
        (c3, "Accuracy test — modèle en production", f"{PROD['metrics']['accuracy_test']:.1%}", f"vs {BEST['metrics']['accuracy_test']:.1%} avant optimisation" if HAS_OPTIMISE else f"vs {OTHER['metrics']['accuracy_test']:.1%} pour l'autre"),
        (c4, "Classes à prédire", "5", ", ".join(CUT_ORDER)),
    ]
    for col, label, value, sub in cards:
        with col:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value">{value}</div>
                        <div class="sub">{sub}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title"><h3>Comment utiliser cette application ?</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **📊 Exploration des données** : visualiser la structure du dataset diamonds et comprendre les relations entre variables et la qualité de taille.
        - **⚖️ Comparaison des modèles** : comprendre comment ont été construits les modèles underfit et overfit, et pourquoi l'un a été retenu.
        - **🔮 Prédiction** : entrer les caractéristiques d'un diamant et obtenir une prédiction instantanée de sa qualité de taille.
        - **🧠 Explicabilité** : comprendre, variable par variable, pourquoi le modèle Naïve Bayes a choisi telle classe plutôt qu'une autre.
        """
    )

    st.markdown('<div class="section-title"><h3>Aperçu du jeu de données</h3></div>', unsafe_allow_html=True)
    st.dataframe(df.drop(columns=["color_enc", "clarity_enc", "cut_enc"]).sample(8, random_state=1), use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 2 — EXPLORATION DES DONNEES (EDA)
# ----------------------------------------------------------------------------
elif page == "📊 Exploration des données":
    st.markdown('<div class="section-title"><h2>📊 Exploration des données</h2></div>', unsafe_allow_html=True)

    colf1, colf2 = st.columns(2)
    with colf1:
        cuts_sel = st.multiselect("Filtrer par qualité de taille (cut)", CUT_ORDER, default=CUT_ORDER)
    with colf2:
        carat_range = st.slider(
            "Filtrer par carat", float(df["carat"].min()), float(df["carat"].max()),
            (float(df["carat"].min()), float(df["carat"].max()))
        )

    dff = df[df["cut"].isin(cuts_sel) & df["carat"].between(*carat_range)]
    st.caption(f"{len(dff):,} diamants affichés après filtrage.".replace(",", " "))

    c1, c2 = st.columns(2)
    with c1:
        counts = dff["cut"].value_counts().reindex(CUT_ORDER).fillna(0)
        fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            color_discrete_map=PALETTE, labels={"x": "Cut", "y": "Nombre de diamants"},
            title="Répartition des diamants par qualité de taille",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(
            dff, x="cut", y="price", color="cut", category_orders={"cut": CUT_ORDER},
            color_discrete_map=PALETTE, title="Distribution du prix par qualité de taille",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sample = dff.sample(min(4000, len(dff)), random_state=1) if len(dff) > 0 else dff
        fig3 = px.scatter(
            sample, x="carat", y="price", color="cut", category_orders={"cut": CUT_ORDER},
            color_discrete_map=PALETTE, opacity=0.55,
            title="Carat vs Prix (coloré par cut)",
        )
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.violin(
            dff, x="cut", y="depth", color="cut", category_orders={"cut": CUT_ORDER},
            color_discrete_map=PALETTE, box=True,
            title="Profondeur (depth %) par qualité de taille",
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title"><h3>Corrélations entre variables numériques</h3></div>', unsafe_allow_html=True)
    num_cols = ["carat", "depth", "table", "price", "x", "y", "z", "cut_enc"]
    corr = dff[num_cols].corr().round(2)
    fig5 = px.imshow(
        corr, text_auto=True, color_continuous_scale="Purples", aspect="auto",
        title="Matrice de corrélation",
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.info(
        "💡 **Observation clé** : `depth` et `table` sont les variables les plus liées à `cut` "
        "(la qualité de taille est historiquement définie à partir des proportions du diamant), "
        "tandis que `color` et `clarity` y sont très faiblement liées. C'est cette observation, "
        "mesurée avec l'information mutuelle dans le notebook, qui a guidé la construction du "
        "modèle **sous-ajusté** (basé uniquement sur `color`/`clarity`)."
    )


# ----------------------------------------------------------------------------
# PAGE 3 — COMPARAISON DES MODELES
# ----------------------------------------------------------------------------
elif page == "⚖️ Comparaison des modèles":
    st.markdown('<div class="section-title"><h2>⚖️ Comparaison Underfit vs Overfit vs Optimisé</h2></div>', unsafe_allow_html=True)

    st.markdown(
        """
        Conformément à la consigne du professeur, **deux versions volontairement dégradées** du même algorithme
        Naïve Bayes ont été construites (underfit et overfit), comparées sur un **jeu de test identique**, puis le
        meilleur des deux a été **réellement optimisé par recherche d'hyperparamètres (`GridSearchCV`, validation
        croisée à 5 plis)** afin d'obtenir le modèle final utilisé pour la prédiction.
        """
    )

    cols = st.columns(3) if HAS_OPTIMISE else st.columns(2)
    keys_icons = [("underfit", "📉"), ("overfit", "📈")] + ([("optimise", "🏁")] if HAS_OPTIMISE else [])
    for col, (key, icon) in zip(cols, keys_icons):
        art = artefacts[key]
        m = art["metrics"]
        is_prod = key == PROD_KEY
        with col:
            border = "border: 2px solid #7C4DFF;" if is_prod else ""
            best_tag = '<span class="pill-best">✓ UTILISÉ EN PRODUCTION</span>' if is_prod else ""
            st.markdown(
                f"""<div class="metric-card" style="{border}">
                    <div class="label">{icon} {m['nom']}</div>
                    <div class="value">{m['accuracy_test']:.1%}</div>
                    <div class="sub">Accuracy test  •  Accuracy train : {m['accuracy_train']:.1%}  •  Écart : {m['ecart_train_test']:+.1%}</div>
                    <br>{best_tag}
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(f"**Variables ({len(m['features'])})** : `{', '.join(m['features'])}`")
            st.markdown(f"**var_smoothing** : `{m['var_smoothing']:.2e}` &nbsp;&nbsp; **Taille d'entraînement** : `{m['n_train']}` obs.")
            if key == "optimise":
                st.markdown(f"**Score CV (GridSearchCV)** : `{m['cv_best_score']:.4f}`")

    st.markdown("---")

    bar_rows = [
        {"Modèle": "Underfit", "Type": "Train", "Accuracy": artefacts["underfit"]["metrics"]["accuracy_train"]},
        {"Modèle": "Underfit", "Type": "Test", "Accuracy": artefacts["underfit"]["metrics"]["accuracy_test"]},
        {"Modèle": "Overfit", "Type": "Train", "Accuracy": artefacts["overfit"]["metrics"]["accuracy_train"]},
        {"Modèle": "Overfit", "Type": "Test", "Accuracy": artefacts["overfit"]["metrics"]["accuracy_test"]},
    ]
    if HAS_OPTIMISE:
        bar_rows += [
            {"Modèle": "Optimisé", "Type": "Train", "Accuracy": artefacts["optimise"]["metrics"]["accuracy_train"]},
            {"Modèle": "Optimisé", "Type": "Test", "Accuracy": artefacts["optimise"]["metrics"]["accuracy_test"]},
        ]
    bar_df = pd.DataFrame(bar_rows)
    fig = px.bar(
        bar_df, x="Modèle", y="Accuracy", color="Type", barmode="group",
        category_orders={"Modèle": ["Underfit", "Overfit", "Optimisé"]},
        color_discrete_map={"Train": "#241C4F", "Test": "#7C4DFF"},
        title="Accuracy train vs test — l'optimisation réduit l'écart tout en augmentant la performance test",
        text_auto=".1%",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title"><h3>Matrices de confusion (jeu de test)</h3></div>', unsafe_allow_html=True)
    cm_keys = [("underfit", "Blues"), ("overfit", "Oranges")] + ([("optimise", "Greens")] if HAS_OPTIMISE else [])
    cm_cols = st.columns(len(cm_keys))
    for col, (key, palette) in zip(cm_cols, cm_keys):
        m = artefacts[key]["metrics"]
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True, x=CUT_ORDER, y=CUT_ORDER,
            color_continuous_scale=palette,
            labels={"x": "Prédit", "y": "Réel"},
            title=f"{m['nom']}",
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title"><h3>Rapport de classification détaillé</h3></div>', unsafe_allow_html=True)
    tab_labels = ["Underfit", "Overfit"] + (["Optimisé"] if HAS_OPTIMISE else [])
    tab_keys = ["underfit", "overfit"] + (["optimise"] if HAS_OPTIMISE else [])
    tabs = st.tabs(tab_labels)
    for tab, key in zip(tabs, tab_keys):
        with tab:
            report = artefacts[key]["metrics"]["classification_report"]
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

    if HAS_OPTIMISE:
        st.success(
            f"""
            **🏁 Modèle utilisé en production pour la prédiction : OPTIMISÉ** (accuracy test = {PROD['metrics']['accuracy_test']:.1%}),
            obtenu par `GridSearchCV` (var_smoothing = {PROD['metrics']['var_smoothing']:.2e}, score CV = {PROD['metrics']['cv_best_score']:.1%})
            à partir du jeu de variables du modèle **{BEST_KEY}** (celui qui avait la meilleure accuracy test brute, {BEST['metrics']['accuracy_test']:.1%}).

            ⚠️ **Nuance pédagogique** : le modèle {BEST_KEY} brut gagnait déjà sur le critère strict d'accuracy test face à l'autre
            modèle dégradé, mais affichait un écart train/test important (signe de sur-apprentissage). L'optimisation par validation
            croisée sur l'ensemble du jeu d'entraînement permet d'obtenir un modèle **à la fois plus précis et plus stable**
            (écart train/test réduit à {PROD['metrics']['ecart_train_test']:+.1%}).
            """
        )
    else:
        st.success(
            f"""
            **🏆 Modèle retenu pour la prédiction : {BEST_KEY.upper()}** (accuracy test = {BEST['metrics']['accuracy_test']:.1%}).
            """
        )


# ----------------------------------------------------------------------------
# PAGE 4 — PREDICTION
# ----------------------------------------------------------------------------
elif page == "🔮 Prédiction":
    st.markdown('<div class="section-title"><h2>🔮 Prédire la qualité de taille d\'un diamant</h2></div>', unsafe_allow_html=True)
    st.caption(f"Prédiction réalisée avec le modèle **{PROD_KEY.upper()}** (accuracy test : {PROD['metrics']['accuracy_test']:.1%})")

    with st.form("prediction_form"):
        st.markdown("#### Caractéristiques du diamant")
        c1, c2, c3 = st.columns(3)
        with c1:
            carat = st.slider("Carat", 0.2, 3.0, 0.7, 0.01)
            color = st.selectbox("Couleur (color)", COLOR_ORDER, index=COLOR_ORDER.index("E"))
            clarity = st.selectbox("Pureté (clarity)", CLARITY_ORDER, index=CLARITY_ORDER.index("SI1"))
        with c2:
            depth = st.slider("Profondeur — depth (%)", 55.0, 70.0, 61.8, 0.1)
            table = st.slider("Table (%)", 50.0, 70.0, 57.0, 0.5)
            price = st.number_input("Prix (USD)", min_value=300, max_value=20000, value=2400, step=50)
        with c3:
            x = st.slider("Longueur x (mm)", 3.5, 9.5, 5.7, 0.01)
            y = st.slider("Largeur y (mm)", 3.5, 9.5, 5.7, 0.01)
            z = st.slider("Hauteur z (mm)", 2.0, 6.0, 3.5, 0.01)

        submitted = st.form_submit_button("✨ Prédire la qualité de taille", use_container_width=True)

    if submitted:
        color_enc = COLOR_ORDER.index(color)
        clarity_enc = CLARITY_ORDER.index(clarity)
        row = build_engineered_row(carat, color_enc, clarity_enc, depth, table, price, x, y, z)

        pred_idx, proba, x_scaled, feats = predict_with(PROD, row)
        pred_cut = CUT_ORDER[pred_idx]

        st.session_state["last_prediction"] = {
            "row": row, "pred_idx": pred_idx, "proba": proba.tolist(),
            "x_scaled": x_scaled.tolist(), "features": feats,
        }

        colr1, colr2 = st.columns([1, 1.4])
        with colr1:
            st.markdown(
                f"""<div class="result-box">
                    <div style="font-size:0.95rem; opacity:0.85;">Qualité de taille prédite</div>
                    <div class="cut-name">{pred_cut}</div>
                    <div style="font-size:1rem;">Confiance : {proba[pred_idx]:.1%}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with colr2:
            proba_df = pd.DataFrame({"Cut": CUT_ORDER, "Probabilité": proba})
            fig = px.bar(
                proba_df, x="Cut", y="Probabilité", color="Cut",
                category_orders={"Cut": CUT_ORDER}, color_discrete_map=PALETTE,
                title="Probabilités prédites par classe", text_auto=".1%",
            )
            fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.info("Rendez-vous dans l'onglet Explicabilité** pour comprendre en détail pourquoi ce diamant a reçu cette prédiction.")
    else:
        st.warning("Renseignez les caractéristiques du diamant ci-dessus puis cliquez sur **Prédire**.")


# ----------------------------------------------------------------------------
# PAGE 5 — EXPLICABILITE (specifique Naive Bayes)
# ----------------------------------------------------------------------------
elif page == "🧠 Explicabilité":
    st.markdown('<div class="section-title"><h2>🧠 Explicabilité du modèle Naïve Bayes</h2></div>', unsafe_allow_html=True)

    st.markdown(
        """
        Contrairement aux modèles à base d'arbres (où l'on utilise souvent **SHAP**), le modèle **Naïve Bayes Gaussien**
        est nativement interprétable : sa prédiction est simplement une somme de **log-vraisemblances par variable**.
        Pour une observation donnée, la contribution de chaque variable à chaque classe est calculée comme :

        `log P(variable = valeur | classe) = -0.5·log(2π·variance) - 0.5·(valeur - moyenne)² / variance`

        La classe retenue est celle dont la somme de ces contributions (+ probabilité a priori de la classe) est la plus élevée.
        """
    )

    if "last_prediction" not in st.session_state:
        st.warning("Merci de faire d'abord une prédiction dans l'onglet **🔮 Prédiction** pour voir son explication ici.")
    else:
        pred = st.session_state["last_prediction"]
        proba = np.array(pred["proba"])
        x_scaled = np.array(pred["x_scaled"])[0]
        feats = pred["features"]
        pred_idx = pred["pred_idx"]

        order = np.argsort(-proba)
        runner_up_idx = order[1]

        model = PROD["model"]
        contrib_pred = nb_feature_log_contributions(model, x_scaled, pred_idx)
        contrib_runner = nb_feature_log_contributions(model, x_scaled, runner_up_idx)
        diff = contrib_pred - contrib_runner

        st.markdown(
            f"**Classe prédite : `{CUT_ORDER[pred_idx]}`** (probabilité {proba[pred_idx]:.1%}) "
            f"vs. **deuxième meilleure classe : `{CUT_ORDER[runner_up_idx]}`** (probabilité {proba[runner_up_idx]:.1%})"
        )

        explain_df = pd.DataFrame({
            "Variable": feats,
            "Contribution": diff,
        }).sort_values("Contribution")

        fig = go.Figure(go.Bar(
            x=explain_df["Contribution"], y=explain_df["Variable"], orientation="h",
            marker_color=np.where(explain_df["Contribution"] >= 0, "#7C4DFF", "#E4572E"),
        ))
        fig.update_layout(
            title=f"Contribution de chaque variable : {CUT_ORDER[pred_idx]} (violet) vs {CUT_ORDER[runner_up_idx]} (orange)",
            xaxis_title=f"Écart de log-vraisemblance (favorise {CUT_ORDER[pred_idx]} si > 0)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
            **Lecture du graphique :**
            - Une barre **violette** (positive) signifie que cette variable, pour la valeur saisie, penche en faveur de `{CUT_ORDER[pred_idx]}` plutôt que `{CUT_ORDER[runner_up_idx]}`.
            - Une barre **orange** (négative) signifie l'inverse : la variable aurait plutôt orienté vers `{CUT_ORDER[runner_up_idx]}`.
            - Plus la barre est longue, plus la variable pèse dans la décision finale du modèle.
            """
        )

        with st.expander("Voir le détail des probabilités par classe"):
            st.dataframe(
                pd.DataFrame({"Cut": CUT_ORDER, "Probabilité": proba}).sort_values("Probabilité", ascending=False),
                use_container_width=True,
            )
