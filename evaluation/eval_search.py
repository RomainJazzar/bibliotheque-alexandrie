"""
ÉVALUATION DE LA RECHERCHE (palier 1) — n'utilise PAS le LLM.

Pour chacune des 20 questions du corpus, on regarde si la page qui contient la réponse
(colonne url_attendue de questions.csv) apparaît dans les 3 premiers résultats : c'est le Hit@3.

On le fait pour les deux tailles de passage (100 et 300 mots) afin de les comparer.
On note aussi la similarité du meilleur passage, y compris pour les questions hors corpus :
cela sert à choisir le seuil MIN_SIMILARITY (en dessous → « Je ne sais pas »).

Lien avec les autres fichiers : utilise search() de rag.py et lit evaluation/questions.csv.
Résultat : evaluation/resultats_recherche.csv + un résumé affiché dans le terminal.

Lancement (depuis la racine du projet) :  python evaluation/eval_search.py
"""

import csv  # pour lire les questions et écrire les résultats
import os
import sys

# Permet d'importer rag.py qui se trouve dans le dossier parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import search  # noqa: E402

CHUNK_SIZES = [100, 300]
QUESTIONS_FILE = "evaluation/questions.csv"
RESULTS_FILE = "evaluation/resultats_recherche.csv"


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        questions = [row for row in csv.DictReader(file) if row["type"] in ["corpus", "hors_corpus"]]

    rows = []  # une ligne de résultat par question
    hits = {size: 0 for size in CHUNK_SIZES}  # questions dont la bonne page est dans le top 3
    first_place = {size: 0 for size in CHUNK_SIZES}  # questions dont la bonne page est 1re (Hit@1)
    best_scores = {size: {"corpus": [], "hors_corpus": []} for size in CHUNK_SIZES}

    for question in questions:
        row = {"id": question["id"], "type": question["type"], "question": question["question"]}
        for size in CHUNK_SIZES:
            passages = search(question["question"], number_of_results=3, chunk_size=size)
            urls_found = [passage["url"] for passage in passages]
            best_scores[size][question["type"]].append(passages[0]["similarity"])
            row[f"meilleure_similarite_{size}"] = passages[0]["similarity"]

            if question["type"] == "corpus":
                is_hit = question["url_attendue"] in urls_found
                # Rang = position de la bonne page dans les résultats (1, 2 ou 3), vide si absente
                rank = urls_found.index(question["url_attendue"]) + 1 if is_hit else ""
                hits[size] += 1 if is_hit else 0
                first_place[size] += 1 if rank == 1 else 0
                row[f"hit3_{size}"] = 1 if is_hit else 0
                row[f"rang_{size}"] = rank
        rows.append(row)
        print(f"Question {question['id']} traitée")

    # Écriture du fichier CSV de résultats
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Résumé affiché
    number_of_corpus_questions = len([q for q in questions if q["type"] == "corpus"])
    print("\n===== RÉSULTATS DE LA RECHERCHE =====")
    for size in CHUNK_SIZES:
        corpus_scores = best_scores[size]["corpus"]
        outside_scores = best_scores[size]["hors_corpus"]
        print(f"Passages de {size} mots : Hit@3 = {hits[size]}/{number_of_corpus_questions}, "
              f"bonne page en 1re position (Hit@1) = {first_place[size]}/{number_of_corpus_questions}")
        print(f"   meilleure similarité, questions du corpus : min {min(corpus_scores)}, "
              f"moyenne {sum(corpus_scores) / len(corpus_scores):.3f}")
        print(f"   meilleure similarité, questions hors corpus : max {max(outside_scores)}, "
              f"moyenne {sum(outside_scores) / len(outside_scores):.3f}")
    print(f"Détail enregistré dans {RESULTS_FILE}")


if __name__ == "__main__":
    main()
