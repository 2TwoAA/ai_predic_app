import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration générale de la page
st.set_page_config(
    page_title="Maintenance Prédictive - Abderrahim Aghzal",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Supprimer le footer "Made with Streamlit"
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Charger modèle et encodeur
model, label_encoder = joblib.load("model.pkl")

# Titre principal + intro
st.title("🛠️ Maintenance Prédictive")
st.markdown("""
Optimisez la fiabilité de vos machines grâce à la détection automatique des défauts des pièces.  
Le système analyse les vibrations pour anticiper les pannes.  
Abderrahim Aghzal AI_Predic  
""")

# Sidebar infos/contact
with st.sidebar:
    st.header("ℹ️ À propos")
    st.write("""
    Ce projet détecte les défauts internes et externes sur des pièces industrielles via l'analyse des vibrations.
    """)
    st.write("👨‍💻 Par : Abderrahim Aghzal")
    st.write("📧 Contact : abderrahimaghzal1@gmail.com")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/abderrahim-aghzal-43b6a521b/)")

# Charger dataset
data = pd.read_csv("vibrations_pieces.csv")
features = ["RMS", "Kurtosis", "Skewness", "Peak2Peak"]

if not all(col in data.columns for col in features):
    st.error(f"⚠️ Le dataset doit contenir les colonnes : {features}")
else:
    # Prédictions
    preds_encoded = model.predict(data[features])
    preds = label_encoder.inverse_transform(preds_encoded)
    data["Prédiction"] = preds

    # Résumé global
    total = len(data)
    counts = data["Prédiction"].value_counts()
    nb_defauts = counts.get("Inner_Fault", 0) + counts.get("Outer_Fault", 0)

    st.markdown(f"### Résumé global")
    st.write(f"Sur **{total}** pièces analysées, **{nb_defauts}** présentent un défaut.")

    # Graphiques
    palette = {"Normal": "#2ecc71", "Inner_Fault": "#e67e22", "Outer_Fault": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x="Prédiction", data=data, palette=palette, ax=ax)
    ax.set_title("Nombre de pièces par état")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    st.pyplot(fig)

    # Camembert
    st.subheader("Répartition des états")
    pie_data = data["Prédiction"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=[palette[k] for k in pie_data.index])
    st.pyplot(fig2)

    # Alerte et détails
    inner_faults = data[data["Prédiction"] == "Inner_Fault"]
    outer_faults = data[data["Prédiction"] == "Outer_Fault"]
    total_defauts = len(inner_faults) + len(outer_faults)

    st.subheader("⚠️ Alerte maintenance")
    if total_defauts > 0:
        st.warning(f"🚨 {total_defauts} pièce(s) présentent un défaut.")
        st.write(f"- {len(inner_faults)} pièces avec défaut interne (Inner_Fault)")
        st.write(f"- {len(outer_faults)} pièces avec défaut externe (Outer_Fault)")

        with st.expander(f"Afficher les détails des pièces défectueuses ({total_defauts})"):
            st.table(pd.concat([inner_faults, outer_faults])[["RMS", "Kurtosis", "Skewness", "Peak2Peak", "Prédiction"]])
    else:
        st.success("✅ Toutes les pièces sont en bon état. Aucun défaut détecté.")

# Footer personnalisé
st.markdown("---")
st.markdown("© 2025 Abderrahim Aghzal AI_Predic")
