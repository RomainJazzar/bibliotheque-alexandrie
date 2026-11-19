"""
API REST de l'assistant (palier 3), construite avec FastAPI.

Endpoints :
  - GET  /info      : quel LLM est utilisé (pour informer l'utilisateur dans l'interface)
  - POST /question  : pose une question → réponse, sources, temps de réponse
  - GET  /stats     : statistiques d'utilisation (pour la page Statistiques)

Statistiques (base SQLite data/stats.db) — pour chaque question, on enregistre UNIQUEMENT :
  date, temps de réponse, nombre de passages utilisés, « je ne sais pas » (oui/non),
  fournisseur du LLM et présence d'une erreur.
  → La question elle-même n'est JAMAIS enregistrée (respect du RGPD : minimisation des données).

Lien avec les autres fichiers :
  - appelle rag.answer_question() pour produire les réponses ;
  - app.py et pages/Statistiques.py (Streamlit) appellent cette API en HTTP.

Lancement :  uvicorn api:app --reload
Documentation interactive automatique : http://127.0.0.1:8000/docs
"""

import os  # pour créer le dossier data/ si besoin
import sqlite3  # base de données intégrée à Python (aucun serveur à installer)
from datetime import datetime  # pour dater chaque question

from fastapi import FastAPI, HTTPException  # le framework d'API
from pydantic import BaseModel  # pour décrire le format des données reçues

import rag

DATABASE_FILE = "data/stats.db"
MAX_QUESTION_LENGTH = 500  # au-delà, la question est refusée (évite les abus)

app = FastAPI(
    title="La Bibliothèque d'Alexandrie",
    description="Assistant qui répond aux questions sur Git à partir de la documentation officielle.",
)


class QuestionInput(BaseModel):
    """Format attendu dans le corps de la requête POST /question : {"question": "..."}"""
    question: str


# ---------------------------------------------------------------------------
# Statistiques (SQLite)
# ---------------------------------------------------------------------------

def create_stats_table():
    """Crée la table des statistiques si elle n'existe pas encore."""
    os.makedirs("data", exist_ok=True)  # le dossier doit exister avant de créer le fichier
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                temps_reponse REAL,
                nb_passages INTEGER,
                je_ne_sais_pas INTEGER,
                fournisseur TEXT,
                erreur INTEGER
            )
        """)


def save_stats(response_time, passages_used, i_dont_know, error):
    """
    Enregistre une ligne de statistiques (sans la question !).
    Paramètres : response_time (float, secondes), passages_used (int),
                 i_dont_know (bool), error (bool).
    """
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            "INSERT INTO stats (date, temps_reponse, nb_passages, je_ne_sais_pas, fournisseur, erreur) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            # Les « ? » sont remplacés par ces valeurs de façon sûre (pas d'injection SQL)
            (datetime.now().isoformat(timespec="seconds"), response_time, passages_used,
             int(i_dont_know), rag.LLM_PROVIDER, int(error)),
        )


# La table est créée au démarrage de l'API
create_stats_table()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/info")
def get_info():
    """Indique quel LLM est utilisé et si les données partent vers un service externe."""
    sends_to_openrouter = rag.LLM_PROVIDER == "openrouter"
    return {
        "provider": rag.LLM_PROVIDER,
        "model": rag.OPENROUTER_MODEL if sends_to_openrouter else rag.LOCAL_MODEL_NAME,
        "sends_data_to_openrouter": sends_to_openrouter,
    }


@app.post("/question")
def ask_question(data: QuestionInput):
    """
    Reçoit une question et renvoie la réponse de l'assistant.
    Entrée  : {"question": "Comment annuler mon dernier commit ?"}
    Sortie  : {"answer", "sources", "response_time", "passages_used", "i_dont_know", "passages", "provider"}
    Erreurs : 400 si la question est vide ou trop longue, 503 si le LLM ou la base est indisponible.
    """
    question = data.question.strip()

    # Validation simple de l'entrée
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="La question est vide ou trop courte.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"La question est trop longue (maximum {MAX_QUESTION_LENGTH} caractères).",
        )

    try:
        result = rag.answer_question(question)
    except RuntimeError as error:
        # Erreur prévue (clé absente, limite OpenRouter, base introuvable...) :
        # on l'enregistre dans les statistiques puis on renvoie un message clair.
        save_stats(response_time=0, passages_used=0, i_dont_know=False, error=True)
        raise HTTPException(status_code=503, detail=str(error))

    save_stats(result["response_time"], result["passages_used"], result["i_dont_know"], error=False)
    return result


@app.get("/stats")
def get_stats():
    """Renvoie toutes les lignes de statistiques (liste de dictionnaires)."""
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row  # pour obtenir des lignes utilisables comme dictionnaires
        rows = connection.execute("SELECT * FROM stats ORDER BY id").fetchall()
    return [dict(row) for row in rows]
