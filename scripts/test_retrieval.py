"""
Script : test_retrieval.py
Description :
    Teste le pipeline de retrieval avec quelques questions exemples.
    Permet de verifier la qualite des resultats avant de brancher le LLM.

Utilisation :
    uv run python scripts/test_retrieval.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path
from arxiv_rag.vectorstore import get_qdrant_client
from arxiv_rag.retrieval import Retriever


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CHUNKS_DIR = Path("data/chunks")


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Test du pipeline de retrieval")
    print("=" * 50)
    print()

    # Initialisation : charge les modeles et construit l'index BM25
    # Premier lancement : telecharge le reranker (~80MB)
    client    = get_qdrant_client()
    retriever = Retriever(
        qdrant_client=client,
        chunks_dir=CHUNKS_DIR,
    )

    # Questions de test
    # On choisit des questions variees pour tester differents aspects :
    #   - Question large (themes generaux du corpus)
    #   - Question precise (termes techniques specifiques)
    #   - Question en langage naturel (pas de jargon)
    test_queries = [
        "What are the main techniques to reduce hallucinations in LLMs?",
        "How does RLHF work for aligning language models?",
        "What is the difference between RAG and fine-tuning?",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Question : {query}")
        print(f"{'='*50}")

        # Lancement du retrieval
        results = retriever.search(query, top_k=3)

        # Affichage des resultats
        for result in results:
            print(f"\n  Rank {result.rank} | Score : {result.score:.3f}")
            print(f"  Paper  : {result.paper_id}")
            print(f"  Page   : {result.page_number}")
            # On affiche les 200 premiers caracteres du chunk
            # pour voir rapidement si c'est pertinent
            preview = result.text[:200].replace("\n", " ")
            print(f"  Extrait: {preview}...")