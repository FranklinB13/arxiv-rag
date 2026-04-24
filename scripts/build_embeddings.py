"""
Script : build_embeddings.py
Description :
    Genere les embeddings de tous les chunks avec le modele BGE.
    C'est la 4eme etape du pipeline RAG.

    Entree  : data/chunks/*.json         (chunks avec metadonnees)
    Sortie  : data/embeddings/*.npy      (vecteurs numpy)
              data/embeddings/*.json     (metadonnees associees)

    Note : le modele BGE (~130MB) sera telecharge au premier lancement
    depuis HuggingFace et mis en cache localement.

Utilisation :
    uv run python scripts/build_embeddings.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path

# On importe nos fonctions depuis le module embeddings
from arxiv_rag.embeddings import load_model, embed_all_chunks


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CHUNKS_DIR = Path("data/chunks")        # chunks produits a l'etape 3
OUTPUT_DIR = Path("data/embeddings")    # embeddings a produire


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Etape 4 : Generation des embeddings")
    print("=" * 50)
    print()

    # Chargement du modele BGE
    # Au premier lancement : telecharge ~130MB depuis HuggingFace
    # Aux suivants : charge depuis le cache (~2 secondes)
    model = load_model()
    print()

    # Generation des embeddings pour tous les chunks
    stats = embed_all_chunks(
        chunks_dir=CHUNKS_DIR,
        output_dir=OUTPUT_DIR,
        model=model,
    )

    # Affichage du resume
    print(f"\nResultats :")
    print(f"  📄 Papers traites  : {stats['papers']}")
    print(f"  🔢 Chunks encodes  : {stats['chunks']}")
    print(f"  ⏭️  Deja presents   : {stats['skipped']}")
    print(f"\n📁 Embeddings sauvegardes dans : {OUTPUT_DIR.absolute()}")