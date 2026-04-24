"""
Module : download_papers.py
Auteur : [Franklin]
Date   : 2026-04
Description :
    Télécharge les papers académiques les plus récents depuis arXiv
    dans les catégories NLP (cs.CL) et Machine Learning (cs.LG).

    Ce script constitue la première étape du pipeline RAG :
    on récupère les données brutes (PDFs) qui seront ensuite
    parsées, découpées, vectorisées et indexées.

Utilisation :
    uv run python scripts/download_papers.py

Notes :
    - Le script est idempotent : on peut le relancer sans créer de doublons.
    - On commence avec MAX_RESULTS=100 pour valider le pipeline.
      Augmenter à 500-1000 une fois que tout fonctionne.
    - arXiv limite le débit des téléchargements volontairement.
      Ne pas supprimer les délais implicites du client arxiv.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import arxiv                  # Client Python pour l'API arXiv
from pathlib import Path      # Gestion des chemins de fichiers (cross-platform)
from tqdm import tqdm         # Barre de progression dans le terminal


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dossier de destination des PDFs téléchargés
# On utilise Path() plutôt que os.path.join() car :
#   - c'est plus lisible
#   - ça fonctionne identiquement sur Windows, Linux et macOS
#   - l'opérateur / pour joindre les chemins est très pratique
OUTPUT_DIR = Path("data/raw")

# Catégories arXiv à interroger
# cs.CL = Computation and Language → tout ce qui est NLP, LLM, RAG, traduction...
# cs.LG = Machine Learning → deep learning, optimisation, généralisation...
# Référence complète des catégories : https://arxiv.org/category_taxonomy
CATEGORIES = ["cs.CL", "cs.LG"]

# Nombre maximum de papers à télécharger
# Valeur basse au départ pour valider que le pipeline fonctionne
# Sans erreurs avant de scaler
MAX_RESULTS = 100


# ==============================================================================
# FONCTIONS
# ==============================================================================

def build_arxiv_query(categories: list[str]) -> str:
    """
    Construit la chaîne de requête arXiv à partir d'une liste de catégories.

    arXiv utilise un langage de requête spécifique :
        cat:cs.CL OR cat:cs.LG
    signifie "papers appartenant à cs.CL OU à cs.LG".

    On sépare la construction de la requête dans sa propre fonction
    pour deux raisons :
        1. Réutilisabilité : on peut l'appeler depuis d'autres scripts
        2. Testabilité : on peut écrire un test unitaire dessus facilement

    Args:
        categories : liste de catégories arXiv, ex: ["cs.CL", "cs.LG"]

    Returns:
        Chaîne de requête formatée, ex: "cat:cs.CL OR cat:cs.LG"
    """
    return " OR ".join(f"cat:{cat}" for cat in categories)


def build_filename(entry_id: str) -> str:
    """
    Construit un nom de fichier propre à partir d'un ID arXiv.

    Problème : les IDs arXiv contiennent des points, ex: "2401.12345v1"
    Or les points séparent le nom de l'extension dans un fichier.
    "2401.12345v1.pdf" pourrait être mal interprété par certains outils.

    Solution : on remplace les points par des underscores.
        "http://arxiv.org/abs/2401.12345v1" → "2401_12345v1.pdf"

    Args:
        entry_id : URL complète du paper, ex: "http://arxiv.org/abs/2401.12345v1"

    Returns:
        Nom de fichier sécurisé, ex: "2401_12345v1.pdf"
    """
    # On prend la dernière partie de l'URL (après le dernier "/")
    # puis on remplace les points par des underscores
    clean_id = entry_id.split("/")[-1].replace(".", "_")
    return f"{clean_id}.pdf"


def download_papers() -> None:
    """
    Télécharge les papers arXiv les plus récents vers le dossier OUTPUT_DIR.

    Logique générale :
        1. Interroger l'API arXiv avec notre requête de catégories
        2. Pour chaque paper retourné :
            a. Construire le chemin de destination
            b. Vérifier s'il existe déjà (idempotence)
            c. Télécharger le PDF si nécessaire
        3. Afficher un résumé

    Idempotence :
        Si on relance ce script plusieurs fois, les PDFs déjà présents
        sur le disque sont ignorés. Aucun doublon, aucune erreur.
        C'est une propriété fondamentale des pipelines de données.

    Returns:
        None
    """

    # --- Préparation ---

    # Crée le dossier de destination s'il n'existe pas encore
    # parents=True  → crée aussi les dossiers parents si nécessaire (ex: data/)
    # exist_ok=True → ne lève pas d'erreur si le dossier existe déjà
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Requête arXiv ---

    # Instanciation du client arXiv
    # Ce client gère automatiquement le rate limiting (délai entre requêtes)
    # pour respecter les conditions d'utilisation d'arXiv
    client = arxiv.Client()

    # Construction de la requête à partir de nos catégories
    query = build_arxiv_query(CATEGORIES)

    # Définition de la recherche
    # sort_by=SubmittedDate + Descending → les papers les plus récents d'abord
    # C'est plus intéressant pour notre RAG : on veut la recherche actuelle
    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # On matérialise le generator en liste
    # Pourquoi ? Un generator ne peut être parcouru qu'une seule fois.
    # En faisant list(), on peut connaître le nombre total de résultats
    # avant de commencer (utile pour la barre de progression tqdm)
    results = list(client.results(search))
    print(f"Papers trouvés : {len(results)}")

    # --- Téléchargement ---

    # Compteurs pour le résumé final
    downloaded = 0   # PDFs effectivement téléchargés pendant cette exécution
    skipped    = 0   # PDFs déjà présents sur le disque (ignorés)
    failed     = 0   # PDFs dont le téléchargement a échoué

    # tqdm affiche une barre de progression dans le terminal
    # desc= : label affiché à gauche de la barre
    for paper in tqdm(results, desc="Téléchargement en cours"):

        # Construction du nom de fichier et du chemin complet
        filename = build_filename(paper.entry_id)
        pdf_path = OUTPUT_DIR / filename   # l'opérateur / de Path joint les chemins

        # --- Idempotence ---
        # Si le fichier existe déjà, on ne le retélécharge pas
        # Avantage : on peut relancer le script après une interruption
        # sans recommencer depuis le début
        if pdf_path.exists():
            skipped += 1
            continue

        # --- Téléchargement ---
        try:
            paper.download_pdf(
                dirpath=str(OUTPUT_DIR),   # dossier de destination
                filename=filename          # nom du fichier
            )
            downloaded += 1

        except Exception as e:
            # On log l'erreur SANS arrêter le script
            # Philosophie : un PDF manquant ne doit pas bloquer
            # le téléchargement des 99 autres
            # \n avant le message pour ne pas écraser la barre tqdm
            print(f"\nÉchec pour {filename} : {e}")
            failed += 1

    # --- Résumé final ---
    print(f"\n{'='*50}")
    print(f"Téléchargement terminé !")
    print(f"  ✅ Téléchargés  : {downloaded}")
    print(f"  ⏭️  Déjà présents : {skipped}")
    print(f"  ❌ Échecs        : {failed}")
    print(f"  📁 Dossier       : {OUTPUT_DIR.absolute()}")
    print(f"{'='*50}")


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

# Ce bloc s'exécute UNIQUEMENT si on lance ce fichier directement :
#     uv run python scripts/download_papers.py   ← exécute le bloc
#
# Si on importe ce module depuis un autre fichier :
#     from scripts.download_papers import build_filename  ← n'exécute PAS le bloc
#
# C'est une convention Python universelle. Toujours faire ça.
if __name__ == "__main__":
    download_papers()