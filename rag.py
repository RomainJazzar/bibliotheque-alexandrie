"""
CŒUR DU PROJET : la chaîne RAG (Retrieval-Augmented Generation).

Question → Embedding → Recherche dans ChromaDB → Prompt → LLM → Réponse avec sources

Ce fichier contient :
  - search()          : la recherche sémantique (testable SANS le LLM) ;
  - build_prompt()    : la construction du prompt à partir d'un fichier de prompts/ ;
  - call_openrouter() : l'appel au LLM distant (avec cache et gestion du 429) ;
  - call_local_model(): l'appel au petit LLM local Hugging Face ;
  - answer_question() : la chaîne complète, utilisée par l'API.

Lien avec les autres fichiers :
  - build_index.py utilise get_embedding_model() et CHROMA_FOLDER pour remplir la base ;
  - api.py appelle answer_question() ;
  - les scripts de evaluation/ appellent search(), generate_answer(), etc.

Tous les réglages se trouvent dans le fichier .env (voir .env.example).
"""

import hashlib  # pour transformer un prompt en « empreinte » (clé de cache)
import json  # pour lire / écrire le cache sur le disque
import os  # pour lire les variables d'environnement
import time  # pour mesurer le temps de réponse et attendre entre deux appels

import chromadb  # la base vectorielle locale
import requests  # pour appeler l'API OpenRouter
from dotenv import load_dotenv  # pour lire le fichier .env
from sentence_transformers import SentenceTransformer  # pour calculer les embeddings

# Lit le fichier .env et place son contenu dans les variables d'environnement
load_dotenv()

# ---------------------------------------------------------------------------
# RÉGLAGES (modifiables dans le fichier .env, sans toucher au code)
# ---------------------------------------------------------------------------

# Modèle d'embeddings open source, multilingue et petit (118 millions de paramètres)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Dossier où ChromaDB enregistre la base vectorielle
CHROMA_FOLDER = "data/chroma"

# Taille des passages (en mots) utilisée par l'application (voir la comparaison dans le README)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))

# Nombre de passages envoyés au LLM
NUMBER_OF_PASSAGES = 3

# Si le meilleur passage a une similarité plus faible que ce seuil,
# on répond « Je ne sais pas » sans même interroger le LLM.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.85"))

# Quel LLM utiliser : "openrouter" (distant) ou "local" (Hugging Face sur ce PC)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Version du prompt utilisée : prompts/v1.txt ou prompts/v2.txt
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")

# Fichier de cache des réponses d'OpenRouter (évite de dépenser le quota gratuit)
CACHE_FILE = "data/cache_llm.json"

# Les modèles gratuits d'OpenRouter acceptent 20 requêtes par minute,
# donc on attend au moins 3,1 secondes entre deux appels.
SECONDS_BETWEEN_CALLS = 3.1

# Phrase exacte renvoyée quand l'assistant ne peut pas répondre
I_DONT_KNOW = "Je ne sais pas."

# Ces variables gardent en mémoire les modèles déjà chargés
# (les charger prend plusieurs secondes : on ne le fait qu'une fois).
_embedding_model = None
_local_pipeline = None
_last_openrouter_call = 0.0


# ---------------------------------------------------------------------------
# 1. RECHERCHE SÉMANTIQUE
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Charge le modèle d'embeddings (une seule fois) et le renvoie.
    Retourne : un objet SentenceTransformer capable de transformer un texte en vecteur.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection(chunk_size=CHUNK_SIZE):
    """
    Ouvre la collection ChromaDB correspondant à une taille de passage.
    Paramètre : chunk_size (int) = taille des passages en mots (ex. 100 ou 300).
    Retourne : la collection ChromaDB « git_docs_<taille> ».
    """
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    try:
        return client.get_collection(f"git_docs_{chunk_size}")
    except Exception:
        raise RuntimeError(
            "La base de connaissances est introuvable. "
            "Lancez d'abord : python download_docs.py puis python build_index.py"
        )


def search(question, number_of_results=NUMBER_OF_PASSAGES, chunk_size=CHUNK_SIZE):
    """
    Cherche les passages les plus proches d'une question.
    Paramètres :
      - question (str) : la question de l'utilisateur ;
      - number_of_results (int) : combien de passages renvoyer (3 par défaut) ;
      - chunk_size (int) : quelle collection interroger.
    Retourne : une liste de dictionnaires, du plus proche au moins proche :
      {"text": ..., "url": ..., "title": ..., "section": ..., "similarity": 0.87}
    """
    # Le modèle e5 attend le préfixe « query: » devant une question.
    question_vector = get_embedding_model().encode(
        "query: " + question, normalize_embeddings=True
    ).tolist()

    results = get_collection(chunk_size).query(
        query_embeddings=[question_vector], n_results=number_of_results
    )

    # ChromaDB renvoie des listes de listes (une par question) : on prend la première.
    passages = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        passages.append({
            "text": results["documents"][0][i],
            "url": metadata["url"],
            "title": metadata["title"],
            "section": metadata["section"],
            # ChromaDB renvoie une distance cosinus (0 = identique) ; on la convertit
            # en similarité (1 = identique), plus facile à comprendre.
            "similarity": round(1 - results["distances"][0][i], 3),
        })
    return passages


# ---------------------------------------------------------------------------
# 2. CONSTRUCTION DU PROMPT
# ---------------------------------------------------------------------------

def build_prompt(question, passages, prompt_version=PROMPT_VERSION):
    """
    Remplit le modèle de prompt (prompts/v1.txt ou v2.txt) avec les passages et la question.
    Paramètres :
      - question (str) ; passages (list[dict]) renvoyés par search() ;
      - prompt_version (str) : "v1" ou "v2".
    Retourne : le prompt complet (str) à envoyer au LLM.
    """
    with open(f"prompts/{prompt_version}.txt", encoding="utf-8") as file:
        template = file.read()

    passages_text = ""
    for number, passage in enumerate(passages, start=1):
        passages_text += (
            f"[Passage {number}]\n"
            f"Source : {passage['url']}\n"
            f"Section : {passage['section']}\n"
            f"{passage['text']}\n\n"
        )

    # On utilise replace() plutôt que format() : le texte de la doc contient
    # parfois des accolades { } qui feraient planter format().
    return template.replace("{passages}", passages_text).replace("{question}", question)


# ---------------------------------------------------------------------------
# 3. APPEL DES MODÈLES DE LANGAGE (LLM)
# ---------------------------------------------------------------------------

def load_cache():
    """Lit le cache sur le disque. Retourne un dictionnaire {empreinte du prompt: réponse}."""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache):
    """Écrit le cache (dictionnaire) sur le disque."""
    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=1)


def call_openrouter(prompt, use_cache=True):
    """
    Envoie le prompt au modèle OpenRouter et renvoie sa réponse (str).
    - Cache : si le même prompt a déjà été envoyé, on renvoie la réponse enregistrée.
      La clé du cache est une empreinte SHA-256 : la question n'y est pas stockée en clair.
      use_cache=False force un vrai appel (utile pour mesurer le temps de réponse).
    - Limites : on espace les appels de 3,1 s et on réessaie 3 fois en cas d'erreur 429.
    - En cas de problème, on lève une RuntimeError avec un message compréhensible.
    """
    global _last_openrouter_call

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Aucune clé OpenRouter n'est configurée. Ajoutez OPENROUTER_API_KEY dans le "
            "fichier .env, ou mettez LLM_PROVIDER=local pour utiliser le modèle local."
        )

    cache = load_cache()
    cache_key = hashlib.sha256((OPENROUTER_MODEL + prompt).encode("utf-8")).hexdigest()
    if use_cache and cache_key in cache:
        return cache[cache_key]

    for attempt in range(1, 4):  # 3 tentatives au maximum
        # Respecter la limite de 20 requêtes par minute
        seconds_since_last_call = time.time() - _last_openrouter_call
        if seconds_since_last_call < SECONDS_BETWEEN_CALLS:
            time.sleep(SECONDS_BETWEEN_CALLS - seconds_since_last_call)
        _last_openrouter_call = time.time()

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,  # réponses stables et peu « créatives »
                },
                timeout=90,
            )
        except requests.exceptions.RequestException:
            raise RuntimeError("Impossible de joindre OpenRouter. Vérifiez votre connexion internet.")

        if response.status_code == 429:
            # Trop de requêtes : on attend un peu avant de réessayer
            print(f"OpenRouter : limite atteinte (429), tentative {attempt}/3, attente 10 s...")
            time.sleep(10)
            continue
        if response.status_code == 401:
            raise RuntimeError("La clé OpenRouter est invalide. Vérifiez OPENROUTER_API_KEY dans .env.")
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter a renvoyé l'erreur {response.status_code} : {response.text[:200]}")

        try:
            answer = response.json()["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            # Le modèle distant a renvoyé une réponse vide ou mal formée
            raise RuntimeError("OpenRouter a renvoyé une réponse inattendue. Réessayez dans un instant.")
        cache[cache_key] = answer
        save_cache(cache)
        return answer

    raise RuntimeError(
        "Limite de requêtes OpenRouter atteinte (modèles gratuits : 20 par minute, "
        "50 par jour sans crédit). Réessayez plus tard ou utilisez LLM_PROVIDER=local."
    )


def call_local_model(prompt):
    """
    Envoie le prompt au petit modèle Hugging Face exécuté sur ce PC et renvoie sa réponse (str).
    Le modèle est téléchargé automatiquement la première fois (~3 Go), puis gardé en mémoire.
    """
    global _local_pipeline
    if _local_pipeline is None:
        # Import ici (et pas en haut du fichier) : transformers est long à charger,
        # inutile de le faire si on n'utilise que OpenRouter.
        from transformers import pipeline
        _local_pipeline = pipeline("text-generation", model=LOCAL_MODEL_NAME)

    messages = [{"role": "user", "content": prompt}]
    # do_sample=False : le modèle choisit toujours le mot le plus probable (réponses stables)
    result = _local_pipeline(messages, max_new_tokens=300, do_sample=False)
    # Le pipeline renvoie toute la conversation : le dernier message est la réponse du modèle.
    return result[0]["generated_text"][-1]["content"].strip()


def ask_llm(prompt, provider=LLM_PROVIDER):
    """Envoie le prompt au bon LLM selon provider ("openrouter" ou "local") et renvoie la réponse."""
    if provider == "local":
        return call_local_model(prompt)
    return call_openrouter(prompt)


# ---------------------------------------------------------------------------
# 4. LA CHAÎNE COMPLÈTE
# ---------------------------------------------------------------------------

def generate_answer(question, passages, provider=LLM_PROVIDER, prompt_version=PROMPT_VERSION):
    """
    À partir d'une question et de passages déjà trouvés, produit la réponse finale.
    Retourne un dictionnaire :
      {"answer": str, "sources": [ {url, title, section} ], "passages_used": int, "i_dont_know": bool}
    """
    # Règle 1 : si aucun passage n'est assez proche de la question, on ne dérange pas le LLM.
    # Avantages : pas d'invention, plus rapide, et la question n'est pas envoyée à l'extérieur.
    if not passages or passages[0]["similarity"] < MIN_SIMILARITY:
        return {"answer": I_DONT_KNOW, "sources": [], "passages_used": 0, "i_dont_know": True}

    prompt = build_prompt(question, passages, prompt_version)
    answer = ask_llm(prompt, provider)

    # Règle 2 : si le LLM dit qu'il ne sait pas, on le signale et on n'affiche pas de sources.
    i_dont_know = "je ne sais pas" in answer.lower()

    # Les sources sont ajoutées par le code (et pas seulement par le LLM) :
    # même si on demande au modèle de « ne pas citer ses sources », elles restent affichées.
    sources = []
    if not i_dont_know:
        for passage in passages:
            source = {"url": passage["url"], "title": passage["title"], "section": passage["section"]}
            if source not in sources:
                sources.append(source)

    return {"answer": answer, "sources": sources, "passages_used": len(passages), "i_dont_know": i_dont_know}


def answer_question(question, provider=LLM_PROVIDER, prompt_version=PROMPT_VERSION):
    """
    Chaîne complète : question → recherche → prompt → LLM → réponse avec sources.
    Paramètres : question (str), provider ("openrouter"/"local"), prompt_version ("v1"/"v2").
    Retourne le dictionnaire de generate_answer() complété par :
      "passages" (les passages trouvés), "response_time" (secondes) et "provider".
    """
    start = time.perf_counter()
    passages = search(question)
    result = generate_answer(question, passages, provider, prompt_version)
    result["passages"] = passages
    result["response_time"] = round(time.perf_counter() - start, 2)
    result["provider"] = provider
    return result


# Petit test manuel : python rag.py "Comment annuler mon dernier commit ?"
if __name__ == "__main__":
    import sys
    user_question = sys.argv[1] if len(sys.argv) > 1 else "Comment annuler mon dernier commit ?"
    print("=== Recherche seule (sans LLM) ===")
    for found in search(user_question):
        print(f"{found['similarity']}  {found['title']} > {found['section']}\n   {found['url']}")
    print("\n=== Réponse complète ===")
    print(answer_question(user_question)["answer"])
