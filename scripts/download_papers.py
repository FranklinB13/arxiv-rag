"""
Module : download_papers.py
Auteur : Franklin
Date   : 2026-04

Description :
    Telecharge des papers academiques depuis arXiv.

    Deux modes de telechargement :
        - Mode categories (v1) : tous les papers recents de cs.CL et cs.LG
        - Mode queries (v2, actuel) : papers cibles sur des sujets precis

    On utilise maintenant le mode queries car il donne un corpus
    beaucoup plus pertinent pour notre RAG. Un corpus de 200 papers
    tres pertinents vaut mieux que 1000 papers generiques.

    Idempotence : le script peut etre relance sans creer de doublons.
    Si un PDF existe deja sur le disque, il est skipe.

Utilisation :
    uv run python scripts/download_papers.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import arxiv                # Client Python pour l'API arXiv
from pathlib import Path    # Gestion des chemins cross-platform
from tqdm import tqdm       # Barre de progression dans le terminal


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dossier de destination des PDFs telecharges
OUTPUT_DIR = Path("data/raw")

# Queries thematiques ciblees sur nos sujets d'interet
# Chaque query va chercher les papers les plus pertinents sur ce sujet
# C'est beaucoup plus efficace que de prendre tous les papers d'une categorie
#
# Pourquoi ces sujets ?
#   - RAG : c'est notre projet, on veut que le corpus parle de RAG
#   - RLHF : technique fondamentale d'alignement des LLMs
#   - Hallucinations : probleme majeur des LLMs, sujet de recherche actif
#   - Fine-tuning vs RAG : question classique en entretien ML
#   - Chain of thought : technique de raisonnement tres citee
QUERIES = [
    "RAG retrieval augmented generation language model",
    "RLHF reinforcement learning human feedback LLM alignment",
    "hallucination large language models reduction",
    "fine-tuning vs RAG language models comparison",
    "chain of thought reasoning large language models",
]

# Nombre maximum de papers par query
# 50 x 5 queries = 250 papers maximum (moins avec les doublons)
# Les doublons sont skipes automatiquement grace a l'idempotence
MAX_PER_QUERY = 50


# ==============================================================================
# FONCTIONS
# ==============================================================================

def build_filename(entry_id: str) -> str:
    """
    Construit un nom de fichier propre depuis un ID arXiv.

    Les IDs arXiv contiennent des points qui peuvent etre confondus
    avec des extensions de fichier. On les remplace par des underscores.

    Exemple :
        "http://arxiv.org/abs/2401.12345v1" -> "2401_12345v1.pdf"

    Args:
        entry_id : URL complete du paper arXiv

    Returns:
        Nom de fichier securise avec extension .pdf
    """
    # On prend la derniere partie de l'URL apres le dernier "/"
    # puis on remplace les points par des underscores
    clean_id = entry_id.split("/")[-1].replace(".", "_")
    return f"{clean_id}.pdf"


def download_papers() -> None:
    """
    Telecharge des papers cibles sur nos sujets d'interet.

    Strategie :
        Pour chaque query thematique, on interroge arXiv avec
        sort_by=Relevance (pas par date comme avant).
        Cela retourne les papers les plus cites et pertinents
        sur ce sujet, pas juste les plus recents.

        Les doublons (un paper qui matche plusieurs queries)
        sont automatiquement skipes grace a l'idempotence :
        si le fichier existe deja, on passe au suivant.

    Returns:
        None (affiche les statistiques en fin d'execution)
    """

    # Creation du dossier de destination si necessaire
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Client arXiv : gere le rate limiting automatiquement
    client = arxiv.Client()

    # Compteurs globaux pour le resume final
    total_downloaded = 0
    total_skipped    = 0
    total_failed     = 0

    # On itere sur chaque query thematique
    for query in QUERIES:

        print(f"\nQuery : '{query}'")

        # Recherche arXiv par pertinence (pas par date)
        # sort_by=Relevance : les papers les plus pertinents pour ce sujet
        # C'est mieux que SubmittedDate pour construire un corpus de reference
        search = arxiv.Search(
            query      = query,
            max_results= MAX_PER_QUERY,
            sort_by    = arxiv.SortCriterion.Relevance,
        )

        # Materialisation des resultats
        results = list(client.results(search))
        print(f"  {len(results)} papers trouves")

        # Compteurs par query pour le suivi
        downloaded = 0
        skipped    = 0
        failed     = 0

        for paper in tqdm(results, desc="  Telechargement"):

            # Construction du nom de fichier
            filename = build_filename(paper.entry_id)
            pdf_path = OUTPUT_DIR / filename

            # Idempotence : skip si le fichier existe deja
            # Gere aussi les doublons entre queries
            if pdf_path.exists():
                skipped += 1
                continue

            try:
                paper.download_pdf(
                    dirpath = str(OUTPUT_DIR),
                    filename= filename,
                )
                downloaded += 1

            except Exception as e:
                # On log l'erreur sans arreter le script
                print(f"\n  Echec {filename} : {e}")
                failed += 1

        # Resume par query
        print(f"  Telecharges : {downloaded} | Skipes : {skipped} | Echecs : {failed}")

        # Mise a jour des compteurs globaux
        total_downloaded += downloaded
        total_skipped    += skipped
        total_failed     += failed

    # Resume global
    print(f"\n{'='*50}")
    print(f"Telechargement termine !")
    print(f"  Telecharges  : {total_downloaded}")
    print(f"  Deja presents: {total_skipped}")
    print(f"  Echecs       : {total_failed}")
    print(f"  Dossier      : {OUTPUT_DIR.absolute()}")
    print(f"{'='*50}")


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":
    download_papers()