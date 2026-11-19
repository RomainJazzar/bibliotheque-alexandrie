"""
PAGE STATISTIQUES de l'interface Streamlit (surveillance de l'application, palier 3).

Streamlit transforme automatiquement chaque fichier du dossier pages/ en une page
accessible depuis le menu de gauche.

Cette page demande les statistiques à l'API (GET /stats) puis affiche :
  - des indicateurs (nombre de questions, temps moyen, taux de « je ne sais pas »...) ;
  - un graphique du temps de réponse ;
  - le tableau complet, téléchargeable en CSV.
"""

import os

import pandas as pd  # pour manipuler le tableau de statistiques
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Statistiques", page_icon="📊")
st.title("📊 Statistiques d'utilisation")

st.write(
    "**Ce qui est enregistré pour chaque question :** la date, le temps de réponse, le nombre de "
    "passages envoyés au modèle, si l'assistant a répondu « je ne sais pas », le fournisseur du "
    "modèle et la présence d'une erreur. **Le texte des questions n'est jamais enregistré.**"
)

try:
    rows = requests.get(f"{API_URL}/stats", timeout=5).json()
except requests.exceptions.RequestException:
    st.error("L'API ne répond pas. Lancez-la avec : `uvicorn api:app`")
    st.stop()

if not rows:
    st.info("Aucune question n'a encore été posée.")
    st.stop()

stats = pd.DataFrame(rows)
answered = stats[stats["erreur"] == 0]  # on exclut les erreurs des moyennes

# Indicateurs principaux, sur 4 colonnes
column1, column2, column3, column4 = st.columns(4)
column1.metric("Questions posées", len(stats))
column2.metric("Temps moyen", f"{answered['temps_reponse'].mean():.1f} s" if len(answered) else "-")
column3.metric("« Je ne sais pas »", f"{answered['je_ne_sais_pas'].mean() * 100:.0f} %" if len(answered) else "-")
column4.metric("Erreurs", int(stats["erreur"].sum()))

st.subheader("Temps de réponse par question (secondes)")
st.line_chart(answered, x="id", y="temps_reponse")

st.subheader("Nombre de passages utilisés")
st.bar_chart(answered["nb_passages"].value_counts().sort_index())

st.subheader("Détail")
st.dataframe(stats, hide_index=True)

# Export CSV : le bouton télécharge le tableau ci-dessus
st.download_button(
    "Télécharger les statistiques (CSV)",
    data=stats.to_csv(index=False).encode("utf-8"),
    file_name="statistiques.csv",
    mime="text/csv",
)
