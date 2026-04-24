"""
Script : chunk_papers.py
Description :
    Lance le decoupage de tous les textes extraits en chunks.
    C'est la 3eme etape du pipeline RAG.

    Entree  : data/processed/*.txt  (textes nettoyes)
    Sortie  : data/chunks/*.json    (chunks avec metadonnees)

Utilisation :
    uv run python scripts/chunk_papers.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path
from arxiv_rag.chunking import chunk_all_papers, CHUNK_SIZE_WORDS, OVERLAP_WORDS


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR  = Path("data/processed")   # textes nettoyes (etape 2)
OUTPUT_DIR = Path("data/chunks")      # chunks JSON (etape 3)


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Etape 3 : Chunking des textes")
    print("=" * 50)
    print(f"Taille des chunks  : {CHUNK_SIZE_WORDS} mots")
    print(f"Overlap            : {OVERLAP_WORDS} mots")
    print()

    # Lancement du chunking en batch
    stats = chunk_all_papers(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Affichage du resume
    print(f"\nResultats :")
    print(f"  📄 Papers traites  : {stats['papers']}")
    print(f"  🔪 Chunks produits : {stats['chunks']}")
    print(f"  ⏭️  Deja presents   : {stats['skipped']}")

    # Calcul de la moyenne de chunks par paper (evite la division par zero)
    if stats["papers"] > 0:
        avg = stats["chunks"] / stats["papers"]
        print(f"  📊 Moyenne         : {avg:.1f} chunks/paper")

    print(f"\n📁 Chunks sauvegardes dans : {OUTPUT_DIR.absolute()}")