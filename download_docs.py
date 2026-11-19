"""
ÉTAPE 1 DU PROJET : télécharger la documentation officielle de Git.

Ce script :
  1. lit la table des matières du livre officiel « Pro Git » en français
     (hébergé sur git-scm.com, le site officiel de Git) ;
  2. télécharge chaque page ;
  3. nettoie le HTML pour ne garder que le texte, découpé par section ;
  4. enregistre le tout dans data/pages.json.

Lien avec les autres fichiers :
  - build_index.py lit data/pages.json pour construire la base vectorielle.
  - Pour changer de corpus (par exemple la documentation de l'école), il suffit
    de produire un autre data/pages.json au même format : le reste du projet
    ne change pas.

Format de data/pages.json (une liste de pages) :
  [
    {
      "url": "https://git-scm.com/book/fr/v2/...",
      "title": "2.4 Les bases de Git - Annuler des actions",
      "sections": [ {"section": "Annuler des actions", "text": "..."}, ... ]
    },
    ...
  ]

Lancement :  python download_docs.py
"""

import json  # pour écrire le fichier pages.json
import os  # pour créer le dossier data/
import time  # pour faire une petite pause entre deux téléchargements

import requests  # pour télécharger les pages web
from bs4 import BeautifulSoup  # pour lire le HTML et en extraire le texte

# Adresse du site et de la table des matières du livre en français
SITE_URL = "https://git-scm.com"
TABLE_OF_CONTENTS_URL = "https://git-scm.com/book/fr/v2"

# Fichier de sortie
OUTPUT_FILE = "data/pages.json"


def download_html(url):
    """
    Télécharge une page web.
    Paramètre : url (str) = adresse de la page.
    Retourne : le code HTML de la page (str).
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # déclenche une erreur si le site répond 404, 500...
    response.encoding = "utf-8"  # le site est en UTF-8 (accents français)
    return response.text


def get_page_links():
    """
    Récupère la liste des URL des pages du livre depuis la table des matières.
    On ignore les pages « Résumé » qui ne font que répéter le chapitre.
    Retourne : une liste d'URL complètes (list[str]), sans doublon.
    """
    html = download_html(TABLE_OF_CONTENTS_URL)
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for link in soup.find_all("a"):
        href = link.get("href", "")
        is_book_page = href.startswith("/book/fr/v2/")
        is_summary = link.get_text().strip() == "Résumé"
        full_url = SITE_URL + href
        if is_book_page and not is_summary and full_url not in links:
            links.append(full_url)
    return links


def extract_sections(html):
    """
    Nettoie le HTML d'une page et découpe son texte par section.
    Paramètre : html (str) = code HTML de la page.
    Retourne : (titre de la page, liste de sections)
               chaque section est un dictionnaire {"section": ..., "text": ...}.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Le contenu utile de la page est dans <div id="main"> (pas les menus).
    main_content = soup.find("div", id="main")
    page_title = main_content.find("h1").get_text(" ", strip=True)

    sections = []
    current_section = page_title  # section par défaut si aucun titre n'est trouvé
    current_lines = []

    # On parcourt les titres (h2, h3, h4), les paragraphes (p) et le code (pre)
    # dans l'ordre où ils apparaissent dans la page.
    for element in main_content.find_all(["h2", "h3", "h4", "p", "pre"]):
        if element.name in ["h2", "h3", "h4"]:
            # Nouveau titre : on enregistre la section précédente et on en commence une autre.
            if current_lines:
                sections.append({"section": current_section, "text": "\n".join(current_lines)})
            current_section = element.get_text(" ", strip=True)
            current_lines = []
        else:
            # Paragraphe : on colle les morceaux avec des espaces.
            # Bloc de code (pre) : on garde les retours à la ligne d'origine.
            text =element.get_text(" ", strip=True) if element.name == "p" else element.get_text()
            if text.strip():
                current_lines.append(text.strip())

    # Ne pas oublier la dernière section de la page
    if current_lines:
        sections.append({"section": current_section, "text": "\n".join(current_lines)})

    return page_title, sections


def main():
    """Télécharge toutes les pages et les enregistre dans data/pages.json."""
    os.makedirs("data", exist_ok=True)

    links = get_page_links()
    print(f"{len(links)} pages trouvées dans la table des matières.")

    pages = []
    for number, url in enumerate(links, start=1):
        try:
            html = download_html(url)
            title, sections = extract_sections(html)
            pages.append({"url": url, "title": title, "sections": sections})
            print(f"[{number}/{len(links)}] {title} ({len(sections)} sections)")
        except Exception as error:
            # Une page en erreur ne doit pas arrêter tout le téléchargement
            print(f"[{number}/{len(links)}] ERREUR pour {url} : {error}")
        time.sleep(0.3)  # politesse : on ne surcharge pas le site

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)
    print(f"\n{len(pages)} pages enregistrées dans {OUTPUT_FILE}")


# Ce bloc s'exécute seulement quand on lance « python download_docs.py »
# (et pas quand un autre fichier importe ce module).
if __name__ == "__main__":
    main()
