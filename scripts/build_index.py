"""
Script : build_index.py
Description :
    Insere tous les embeddings dans Qdrant pour permettre la recherche.
    C'est la 5eme etape du pipeline RAG.

    Entree  : data/embeddings/*.npy + *.json
    Sortie  : qdrant_storage/ (base de donnees vectorielle locale)

Utilisation :
    uv run python scripts/build_index.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path
from arxiv_rag.vectorstore import (
    get_qdrant_client,
    create_collection,
    insert_embeddings,
    get_collection_info,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

EMBEDDINGS_DIR = Path("data/embeddings")


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Etape 5 : Construction de l'index Qdrant")
    print("=" * 50)
    print()

    # Connexion a Qdrant en mode local
    # Les donnees seront stockees dans ./qdrant_storage/
    print("Connexion a Qdrant (mode local)...")
    client = get_qdrant_client()

    # Creation de la collection
    # recreate=False : on ne recrée pas si elle existe deja (idempotence)
    create_collection(client, recreate=False)
    print()

    # Insertion de tous les embeddings
    stats = insert_embeddings(
        client=client,
        embeddings_dir=EMBEDDINGS_DIR,
    )

    # Affichage du resume
    print(f"\nResultats :")
    print(f"  📄 Papers indexes  : {stats['papers']}")
    print(f"  📍 Points inseres  : {stats['points']}")
    print(f"  ⏭️  Skipes          : {stats['skipped']}")

    # Verification finale : infos sur la collection
    get_collection_info(client)
    print(f"\n✅ Index pret pour le retrieval !")