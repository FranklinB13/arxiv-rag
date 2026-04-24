"""
Script : query.py
Description :
    Interface en ligne de commande pour interroger le RAG.
    Permet de poser des questions et d'obtenir des reponses
    avec citations depuis notre corpus de papers arXiv.

Utilisation :
    uv run python scripts/query.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from arxiv_rag.rag import RAGPipeline
import time

# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ArXiv RAG - Posez des questions sur la recherche NLP/ML")
    print("=" * 60)
    print()

    # Initialisation du pipeline
    # Charge tous les modeles une seule fois
    rag = RAGPipeline()

    # Questions de demonstration
    # On teste les memes questions que lors du test retrieval
    # pour voir la difference avec et sans generation
    demo_questions = [
        "What are the main techniques to reduce hallucinations in LLMs?",
        "How does RLHF work for aligning language models?",
        "What is the difference between RAG and fine-tuning?",
    ]

    for question in demo_questions:

        print(f"\n{'='*60}")
        print(f"Question : {question}")
        print(f"{'='*60}")

        # Appel du pipeline RAG complet
        # verbose=True pour voir quels chunks ont ete utilises
        response = rag.ask(question, verbose=True)

        # Affichage de la reponse
        print(f"\nReponse :\n")
        print(response.answer)

        # Affichage des sources completes en bas
        print(f"\nSources utilisees :")
        for chunk in response.sources:
            print(f"  [{chunk.rank}] Paper : {chunk.paper_id} | "
                  f"Page : {chunk.page_number} | "
                  f"Score : {chunk.score:.2f}")

        print()

        # Pause entre les questions pour ne pas surcharger l'API
        # L'API Mistral a des rate limits sur le tier gratuit
        input("Appuie sur Entree pour la question suivante...")
        # Pause pour respecter le rate limit Groq (6000 tokens/minute)
        # On attend 20 secondes entre chaque question
        time.sleep(20)