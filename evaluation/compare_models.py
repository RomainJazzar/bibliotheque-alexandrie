"""
COMPARAISON DES DEUX MODÈLES DE LANGAGE (palier 2).

On pose les mêmes questions, avec exactement le même prompt (v2) et les mêmes passages, à :
  - un petit modèle Hugging Face exécuté en local (LOCAL_MODEL dans .env) ;
  - un modèle OpenRouter (OPENROUTER_MODEL dans .env).
On mesure le temps de réponse et on enregistre les réponses pour juger la qualité du français.
Le cache est désactivé ici pour mesurer de vrais temps de réponse.

Lien : utilise search(), build_prompt(), call_local_model() et call_openrouter() de rag.py.
Résultat : evaluation/comparaison_modeles.csv

Lancement (depuis la racine du projet) :  python evaluation/compare_models.py
"""

import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import (LOCAL_MODEL_NAME, OPENROUTER_MODEL, build_prompt,  # noqa: E402
                 call_local_model, call_openrouter, search)

# 5 questions communes, choisies dans questions.csv
QUESTIONS = [
    "Comment configurer mon nom et mon adresse e-mail dans Git ?",
    "J'ai oublié d'ajouter un fichier dans mon dernier commit : comment le corriger ?",
    "Que faire quand Git m'indique un conflit pendant une fusion ?",
    "Comment mettre de côté mes modifications en cours sans faire de commit ?",
    "Quelle est la différence entre git reset --soft et git reset --hard ?",
]


def call_openrouter_without_cache(prompt):
    """Appelle OpenRouter en ignorant le cache, pour mesurer un vrai temps de réponse."""
    return call_openrouter(prompt, use_cache=False)


def main():
    # Dictionnaire {nom affiché du modèle: fonction à appeler avec le prompt}
    models = {
        f"local : {LOCAL_MODEL_NAME}": call_local_model,
        f"openrouter : {OPENROUTER_MODEL}": call_openrouter_without_cache,
    }

    rows = []
    for question in QUESTIONS:
        prompt = build_prompt(question, search(question), "v2")
        for model_name, call_model in models.items():
            start = time.perf_counter()
            try:
                answer = call_model(prompt)
            except RuntimeError as error:
                answer = f"ERREUR : {error}"
            duration = round(time.perf_counter() - start, 2)
            rows.append({"modele": model_name, "question": question, "temps": duration, "reponse": answer})
            print(f"\n--- {model_name} ({duration} s)\nQ : {question}\n{answer}")

    with open("evaluation/comparaison_modeles.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["modele", "question", "temps", "reponse"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== TEMPS MOYEN =====")
    for model_name in models:
        times = [row["temps"] for row in rows if row["modele"] == model_name]
        print(f"{model_name} : {sum(times) / len(times):.1f} s")


if __name__ == "__main__":
    main()
