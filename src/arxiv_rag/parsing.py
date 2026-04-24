"""
Module : parsing.py
Auteur : Franklin
Date   : 2026-04

Description :
    Extraction et nettoyage du texte brut depuis les PDFs arXiv.

    Un PDF est un format d'AFFICHAGE, pas un format texte.
    Les caracteres sont positionnes par coordonnees XY sans notion
    de phrase ou paragraphe. L'extraction produit donc du texte
    imparfait qu'on doit nettoyer.

    Ce module constitue la 2eme etape du pipeline RAG :
        PDFs bruts -> texte nettoye -> (suite : chunking)

Limitations connues :
    - Les formules mathematiques sont souvent illisibles apres extraction
    - Les tableaux complexes deviennent du texte desordonne
    - Les papers en 2 colonnes peuvent avoir des melanges de colonnes
    Ces limitations sont acceptables pour notre cas d'usage RAG.

Utilisation :
    uv run python scripts/parse_papers.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# re : module standard Python pour les expressions regulieres
# On l'utilise pour nettoyer le texte (remplacer des patterns)
import re

# fitz : c'est le nom interne de PyMuPDF
# PyMuPDF est la meilleure lib pour extraire du texte de PDFs scientifiques
import fitz

# Path : gestion des chemins de fichiers cross-platform
# Path("data/raw") fonctionne sur Windows, Linux et macOS
# os.path.join() est l'ancienne facon, Path() est le standard moderne
from pathlib import Path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Seuil minimum de caracteres pour considerer une page comme utile
# En dessous de ce seuil, la page est probablement :
#   - Une page de garde (juste un titre et des auteurs)
#   - Une page blanche
#   - Une page purement graphique (schema, figure sans texte)
# On l'ignore pour ne pas polluer notre corpus avec du contenu vide
MIN_PAGE_CHARS = 100


# ==============================================================================
# FONCTIONS DE NETTOYAGE
# ==============================================================================

def clean_text(raw_text: str) -> str:
    """
    Nettoie le texte brut extrait d'un PDF.

    Probleme : PyMuPDF extrait le texte tel qu'il est dans le PDF,
    c'est-a-dire avec tous les artefacts visuels du format PDF.
    Ces artefacts degradent la qualite de notre RAG si on les garde.

    On corrige 4 types d'artefacts :

    1. Tirets de coupure de mot
       Avant  : "trans-\nformer"
       Apres  : "transformer"
       Cause  : dans les PDFs en 2 colonnes, les mots longs sont
                coupes avec un tiret en fin de colonne

    2. Sauts de ligne en milieu de paragraphe
       Avant  : "The model\nachieves good results"
       Apres  : "The model achieves good results"
       Cause  : chaque ligne visuelle du PDF devient une ligne de texte,
                meme si elle est au milieu d'une phrase

    3. Espaces multiples
       Avant  : "the    attention    mechanism"
       Apres  : "the attention mechanism"
       Cause  : l'espacement visuel du PDF cree des espaces multiples

    4. Paragraphes trop courts (bruit)
       Avant  : "\n\n3\n\n" (numero de page isole)
       Apres  : supprime
       Cause  : numeros de page, headers, footers, legendes de figures

    Args:
        raw_text : texte brut extrait par PyMuPDF, avec tous les artefacts

    Returns:
        Texte nettoye, plus propre pour l'indexation et la generation
    """

    # --- Correction 1 : tirets de coupure ---
    # re.sub(pattern, remplacement, texte)
    # r"-\n" : un tiret (-) suivi d'un saut de ligne (\n)
    # On remplace par "" (rien) pour recoller les deux morceaux du mot
    text = re.sub(r"-\n", "", raw_text)

    # --- Correction 2 : sauts de ligne en milieu de paragraphe ---
    # Probleme : si on remplace tous les \n par des espaces,
    # on perd aussi les separateurs de paragraphes (\n\n)
    # Solution en 3 etapes :
    #   a) on remplace les \n\n par un placeholder temporaire
    #   b) on remplace les \n restants par des espaces
    #   c) on restaure les vrais separateurs de paragraphes

    # Etape a : protection des separateurs de paragraphes
    text = re.sub(r"\n\n", "DOUBLE_NEWLINE", text)

    # Etape b : remplacement des sauts de ligne simples par des espaces
    text = re.sub(r"\n", " ", text)

    # Etape c : restauration des vrais separateurs de paragraphes
    text = re.sub(r"DOUBLE_NEWLINE", "\n\n", text)

    # --- Correction 3 : espaces multiples ---
    # r" {2,}" : 2 espaces ou plus consecutifs
    # On remplace par un seul espace
    text = re.sub(r" {2,}", " ", text)

    # --- Correction 4 : suppression des paragraphes trop courts ---
    # On decoupe le texte en paragraphes (separes par \n\n)
    # On filtre ceux de moins de 40 caracteres :
    #   - "3" (numero de page) → supprime
    #   - "Figure 2." → supprime
    #   - "The attention mechanism allows..." → garde
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 40]

    # On recolle les paragraphes gardes avec un double saut de ligne
    text = "\n\n".join(paragraphs)

    # strip() supprime les espaces/sauts de ligne en debut et fin de texte
    return text.strip()


# ==============================================================================
# PARSING D'UN SEUL PDF
# ==============================================================================

def parse_pdf(pdf_path: Path) -> str | None:
    """
    Extrait et nettoie le texte d'un fichier PDF.

    Strategie : extraction page par page
        On pourrait extraire tout le texte d'un coup avec PyMuPDF,
        mais traiter page par page nous donne :
            - Un separateur clair entre les pages [PAGE N]
            - La possibilite d'ignorer les pages vides individuellement
            - Une meilleure traceabilite (on sait de quelle page vient
              chaque bout de texte, utile pour le debugging)

    Args:
        pdf_path : chemin vers le fichier PDF a parser (objet Path)

    Returns:
        Texte nettoye du paper complet, ou None si :
            - Le PDF est corrompu et ne peut pas etre ouvert
            - Le PDF ne contient aucune page avec assez de texte
    """

    # On entoure tout dans un try/except
    # Pourquoi ? Certains PDFs sont corrompus, proteges par mot de passe,
    # ou dans un format que PyMuPDF ne supporte pas.
    # Sans try/except, un seul PDF problematique ferait planter
    # le traitement de tous les autres.
    try:
        # Ouverture du PDF avec un context manager (with ... as ...)
        # Le context manager garantit que le fichier sera ferme
        # meme si une erreur survient a l'interieur du bloc
        # fitz.open() = PyMuPDF, "fitz" est son nom historique interne
        with fitz.open(pdf_path) as doc:

            # Liste qui va accumuler le texte de chaque page valide
            pages_text = []

            # enumerate() donne a la fois l'index (page_num) et la valeur (page)
            # page_num commence a 0, donc on fait +1 pour afficher 1, 2, 3...
            for page_num, page in enumerate(doc):

                # Extraction du texte de cette page
                # get_text("text") = mode texte brut, le plus simple
                # D'autres modes existent : "html", "dict", "blocks"
                # mais "text" est suffisant pour notre usage RAG
                page_text = page.get_text("text")

                # On ignore les pages avec trop peu de texte
                # strip() supprime les espaces avant de compter
                if len(page_text.strip()) < MIN_PAGE_CHARS:
                    continue

                # On ajoute le texte de la page avec un marqueur
                # Ce marqueur [PAGE N] sera utile plus tard pour
                # savoir d'ou vient un chunk lors du debugging
                pages_text.append(f"[PAGE {page_num + 1}]\n{page_text}")

            # Si aucune page n'avait assez de texte, on retourne None
            # Ca peut arriver pour des PDFs de slides ou de figures
            if not pages_text:
                return None

            # On assemble toutes les pages en un seul grand texte
            # \n\n entre chaque page pour bien les separer
            full_text = "\n\n".join(pages_text)

            # On applique le nettoyage sur le texte complet
            cleaned_text = clean_text(full_text)

            return cleaned_text

    except Exception as e:
        # On affiche l'erreur mais on ne fait pas planter le script
        # \n au debut pour ne pas ecraser une eventuelle barre de progression
        print(f"\nErreur parsing {pdf_path.name} : {e}")
        return None


# ==============================================================================
# TRAITEMENT EN BATCH DE TOUS LES PDFS
# ==============================================================================

def parse_all_papers(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Parse tous les PDFs d'un dossier et sauvegarde les textes extraits.

    Organisation des fichiers :
        Entree  : data/raw/2401_12345v1.pdf
        Sortie  : data/processed/2401_12345v1.txt
        
        On garde le meme nom, on change l'extension .pdf -> .txt
        Ca permet de faire le lien facilement entre PDF et texte extrait.

    Cette fonction est idempotente : si on la relance, les fichiers
    deja traites sont skipes. On peut interrompre et reprendre.

    Args:
        input_dir  : dossier contenant les PDFs bruts
        output_dir : dossier ou sauvegarder les fichiers .txt

    Returns:
        Dictionnaire de statistiques :
        {"success": nb_succes, "skipped": nb_skipes, "failed": nb_echecs}
    """

    # Creation du dossier de sortie s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recuperation de tous les fichiers .pdf dans le dossier d'entree
    # glob("*.pdf") retourne un generator, on le convertit en liste
    # pour pouvoir afficher le nombre total avant de commencer
    pdf_files = list(input_dir.glob("*.pdf"))

    # Cas ou le dossier est vide (peut arriver si le download a rate)
    if not pdf_files:
        print(f"Aucun PDF trouve dans {input_dir}")
        return {"success": 0, "skipped": 0, "failed": 0}

    print(f"PDFs a traiter : {len(pdf_files)}")

    # Compteurs pour le resume final
    stats = {"success": 0, "skipped": 0, "failed": 0}

    for pdf_path in pdf_files:

        # Construction du chemin du fichier de sortie
        # .with_suffix(".txt") remplace l'extension du Path
        # .name extrait juste le nom du fichier (sans le dossier parent)
        # Exemple : Path("data/raw/2401_12345v1.pdf").with_suffix(".txt").name
        #         = "2401_12345v1.txt"
        output_path = output_dir / pdf_path.with_suffix(".txt").name

        # Idempotence : on skip les fichiers deja traites
        if output_path.exists():
            stats["skipped"] += 1
            continue

        # Appel de la fonction de parsing sur ce PDF
        text = parse_pdf(pdf_path)

        # Si le parsing a echoue ou que le PDF etait vide
        if text is None:
            stats["failed"] += 1
            continue

        # Sauvegarde du texte dans le fichier .txt
        # encoding="utf-8" : indispensable pour les caracteres speciaux
        # errors="replace" : remplace les caracteres non-encodables
        # par un ? plutot que de planter
        output_path.write_text(text, encoding="utf-8", errors="replace")
        stats["success"] += 1

    return stats