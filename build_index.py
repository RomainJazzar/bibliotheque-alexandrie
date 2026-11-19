"""
ÉTAPE 2 DU PROJET : construire la base vectorielle (la « base de connaissances »).

Ce script :
  1. lit data/pages.json (produit par download_docs.py) ;
  2. découpe chaque section en passages (chunks) d'environ N mots ;
  3. transforme chaque passage en vecteur (embedding) ;
  4. enregistre vecteurs + texte + source (URL, titre, section) dans ChromaDB.

On construit DEUX collections, une par taille de passage à comparer :
  - git_docs_100 : petits passages (~100 mots)
  - git_docs_300 : grands passages (~300 mots)
La comparaison est faite par evaluation/eval_search.py.

Lien avec les autres fichiers :
  - utilise get_embedding_model() et CHROMA_FOLDER définis dans rag.py ;
  - rag.py interroge ensuite ces collections avec search().

Lancement :  python build_index.py   (quelques minutes la première fois)
"""

import json  # pour lire data/pages.json

import chromadb  # la base vectorielle

from rag import CHROMA_FOLDER, get_embedding_model

# Les deux tailles de passage (en nombre de mots) que l'on compare
CHUNK_SIZES_TO_TEST = [100, 300]

PAGES_FILE = "data/pages.json"


def split_into_chunks(text, chunk_size):
    """
    Découpe un texte en passages d'environ chunk_size mots.
    On coupe uniquement entre deux lignes (dans cette documentation, une ligne = une phrase
    ou une ligne de code) pour ne jamais couper une phrase au milieu.
    Paramètres : text (str), chunk_size (int).
    Retourne : une liste de passages (list[str]).
    """
    chunks = []
    current_lines = []
    current_word_count = 0

    for line in text.split("\n"):
        current_lines.append(line)
        current_word_count += len(line.split())
        if current_word_count >= chunk_size:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_word_count = 0

    # Le reste du texte : s'il est très court, on le colle au passage précédent
    if current_lines:
        rest = "\n".join(current_lines)
        if chunks and current_word_count < chunk_size / 4:
            chunks[-1] += "\n" + rest
        else:
            chunks.append(rest)
    return chunks


def build_collection(pages, chunk_size):
    """
    Crée (ou recrée) la collection ChromaDB « git_docs_<chunk_size> ».
    Paramètres : pages (list[dict]) lues dans pages.json, chunk_size (int).
    Retourne : le nombre de passages enregistrés (int).
    """
    # 1. Découpage : on prépare 3 listes parallèles (même position = même passage)
    texts = []      # le texte de chaque passage
    metadatas = []  # sa source : URL, titre de la page, section
    for page in pages:
        for section in page["sections"]:
            for chunk in split_into_chunks(section["text"], chunk_size):
                # On ajoute le titre et la section en tête du passage : cela aide la
                # recherche (le passage « sait » de quoi il parle) et l'affichage.
                texts.append(f"{page['title']} > {section['section']}\n{chunk}")
                metadatas.append({
                    "url": page["url"],
                    "title": page["title"],
                    "section": section["section"],
                })

    # 2. Embeddings : le modèle e5 attend le préfixe « passage: » devant les documents.
    print(f"Taille {chunk_size} mots : calcul de {len(texts)} embeddings...")
    vectors = get_embedding_model().encode(
        ["passage: " + text for text in texts],
        normalize_embeddings=True,  # vecteurs de longueur 1 → similarité cosinus
        batch_size=32,
        show_progress_bar=True,
    )

    # 3. Stockage dans ChromaDB (on supprime l'ancienne version si elle existe)
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    collection_name = f"git_docs_{chunk_size}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # la collection n'existait pas encore : rien à supprimer
    # "hnsw:space": "cosine" → ChromaDB mesure la distance avec le cosinus
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    # ChromaDB limite la taille d'un envoi : on ajoute les passages par paquets de 1000
    for start in range(0, len(texts), 1000):
        end = start + 1000
        collection.add(
            ids=[f"{chunk_size}-{i}" for i in range(start, min(end, len(texts)))],
            embeddings=vectors[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    return len(texts)


def main():
    with open(PAGES_FILE, encoding="utf-8") as file:
        pages = json.load(file)
    print(f"{len(pages)} pages chargées depuis {PAGES_FILE}")

    for chunk_size in CHUNK_SIZES_TO_TEST:
        count = build_collection(pages, chunk_size)
        print(f"Collection git_docs_{chunk_size} : {count} passages enregistrés.\n")


if __name__ == "__main__":
    main()
