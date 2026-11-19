"""
INTERFACE WEB (palier 3), construite avec Streamlit.

L'étudiant écrit sa question, clique sur « Envoyer » et voit :
  - la réponse de l'assistant ;
  - les sources cliquables ;
  - un message clair quand l'assistant ne sait pas ;
  - (option) les passages de documentation retrouvés.

Important : cette interface ne fait AUCUN calcul d'IA elle-même.
Elle envoie la question à l'API FastAPI (api.py) par une requête HTTP et affiche le résultat.
La page des statistiques se trouve dans pages/Statistiques.py.

Lancement (l'API doit tourner en parallèle) :  streamlit run app.py
"""

import os

import requests  # pour appeler l'API FastAPI
import streamlit as st  # pour créer l'interface web en Python
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Questions d'exemple proposées sous forme de boutons
EXAMPLE_QUESTIONS = [
    "Comment annuler un commit ?",
    "Comment créer une branche ?",
    "À quoi sert git stash ?",
]


def get_api_info():
    """Demande à l'API quel LLM est utilisé. Retourne un dictionnaire, ou None si l'API ne répond pas."""
    try:
        return requests.get(f"{API_URL}/info", timeout=5).json()
    except requests.exceptions.RequestException:
        return None


def ask_api(question):
    """
    Envoie la question à l'API.
    Retourne un couple (résultat, message_erreur) : l'un des deux vaut None.
    """
    try:
        response = requests.post(f"{API_URL}/question", json={"question": question}, timeout=300)
    except requests.exceptions.ConnectionError:
        return None, "L'API ne répond pas. Vérifiez qu'elle est lancée avec : uvicorn api:app"
    except requests.exceptions.Timeout:
        return None, "L'assistant a mis trop de temps à répondre. Réessayez dans un instant."

    if response.status_code != 200:
        # L'API renvoie un message d'erreur en français dans le champ « detail »
        try:
            return None, response.json()["detail"]
        except (ValueError, KeyError):
            return None, f"Erreur inattendue de l'API (code {response.status_code})."
    return response.json(), None


# ---------------------------------------------------------------------------
# Mise en page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bibliothèque d'Alexandrie", page_icon="📚")
st.title("📚 La Bibliothèque d'Alexandrie")
st.write(
    "Posez une question sur **Git** : l'assistant cherche la réponse dans la documentation "
    "officielle (livre *Pro Git* en français, sur git-scm.com) et cite ses sources. "
    "S'il ne trouve pas la réponse, il vous le dit."
)

# Bandeau de confidentialité : on dit clairement ce qui part vers l'extérieur
info = get_api_info()
if info is None:
    st.error("L'API ne répond pas. Lancez-la d'abord dans un autre terminal avec : `uvicorn api:app`")
    st.stop()  # inutile d'afficher la suite
if info["sends_data_to_openrouter"]:
    st.info(
        f"🔒 **Confidentialité** : votre question et les 3 passages de documentation trouvés sont "
        f"envoyés au service externe **OpenRouter** (modèle `{info['model']}`) pour rédiger la réponse. "
        "N'écrivez pas d'informations personnelles. Votre question n'est **pas** enregistrée : "
        "seuls le temps de réponse, le nombre de passages et l'indicateur « je ne sais pas » sont conservés."
    )
else:
    st.info(
        f"🔒 **Confidentialité** : votre question est traitée **entièrement sur ce serveur** "
        f"(modèle local `{info['model']}`), rien n'est envoyé à l'extérieur. Votre question n'est "
        "**pas** enregistrée : seuls le temps de réponse, le nombre de passages et l'indicateur "
        "« je ne sais pas » sont conservés."
    )

# Boutons d'exemples : un clic remplit la question et l'envoie
st.write("**Exemples de questions :**")
clicked_example = None
for column, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
    if column.button(example, width="stretch"):  # le bouton prend toute la largeur de sa colonne
        clicked_example = example
if clicked_example:
    st.session_state["question"] = clicked_example  # remplit le champ de saisie ci-dessous

question = st.text_input("Votre question :", key="question", max_chars=500,
                         placeholder="Ex. : Comment voir l'historique des commits ?")
show_passages = st.checkbox("Afficher les passages de documentation retrouvés")
send_clicked = st.button("Envoyer", type="primary")

# ---------------------------------------------------------------------------
# Envoi de la question et affichage de la réponse
# ---------------------------------------------------------------------------

if send_clicked or clicked_example:
    if len(question.strip()) < 3:
        st.warning("Écrivez d'abord une question (au moins 3 caractères).")
    else:
        with st.spinner("Recherche dans la documentation et rédaction de la réponse..."):
            result, error_message = ask_api(question)

        if error_message:
            st.error(f"⚠️ {error_message}")
        else:
            if result["i_dont_know"]:
                st.warning(
                    "🤷 **Je ne sais pas.** La documentation ne contient pas la réponse à cette "
                    "question (ou elle ne concerne pas Git). Essayez de la reformuler."
                )
            else:
                st.subheader("Réponse")
                st.markdown(result["answer"])
                st.subheader("Sources")
                for source in result["sources"]:
                    st.markdown(f"- [{source['title']} › {source['section']}]({source['url']})")

            st.caption(
                f"⏱️ {result['response_time']} s · {result['passages_used']} passage(s) envoyé(s) "
                f"au modèle · modèle : {result['provider']}"
            )

            if show_passages:
                st.subheader("Passages retrouvés")
                for passage in result["passages"]:
                    with st.expander(f"{passage['title']} › {passage['section']} "
                                     f"(similarité {passage['similarity']})"):
                        st.text(passage["text"])
                        st.markdown(f"[Ouvrir la page]({passage['url']})")
