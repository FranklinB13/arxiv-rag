"""
Module : rag.py
Auteur : Franklin
Date   : 2026-04

Description :
    Pipeline RAG complet : de la question a la reponse avec citations.

    Ce module est le point d'entree principal du systeme.
    Il orchestre les deux grandes etapes :
        1. Retrieval  : trouver les chunks pertinents (retrieval.py)
        2. Generation : generer la reponse avec Mistral (generation.py)

    C'est le pattern "RAG = Retrieval + Augmented Generation" :
        Question
            ↓
        Retriever.search()    → top-5 chunks pertinents
            ↓
        generate_answer()     → reponse avec citations
            ↓
        RAGResponse

Utilisation :
    uv run python scripts/query.py
    ou depuis une interface (Gradio, FastAPI, etc.)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path

# Nos modules
from arxiv_rag.vectorstore import get_qdrant_client
from arxiv_rag.retrieval   import Retriever
from arxiv_rag.generation  import generate_answer, RAGResponse


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CHUNKS_DIR = Path("data/chunks")


# ==============================================================================
# CLASSE RAG PIPELINE
# ==============================================================================

class RAGPipeline:
    """
    Orchestre le pipeline complet : retrieval + generation.

    Pourquoi une classe ?
        Meme raison que pour Retriever : on veut charger les modeles
        une seule fois et les reutiliser pour plusieurs questions.
        En production, on instancie RAGPipeline au demarrage du serveur,
        et on appelle .ask() pour chaque requete utilisateur.

    Utilisation :
        rag = RAGPipeline()
        response = rag.ask("How does RLHF work?")
        print(response.answer)
    """

    def __init__(self, chunks_dir: Path = CHUNKS_DIR):
        """
        Initialise le pipeline en chargeant tous les composants.

        Composants charges :
            1. Client Qdrant (connexion a la vector DB locale)
            2. Retriever (BGE + reranker + BM25)

        Le client Mistral est instancie a chaque appel dans generate_answer()
        car il est leger (juste une connexion HTTP).

        Args:
            chunks_dir : dossier des chunks JSON (pour le BM25 du Retriever)
        """

        print("Initialisation du pipeline RAG...")

        # Connexion a Qdrant en mode local
        self.qdrant_client = get_qdrant_client()

        # Initialisation du retriever (charge BGE + reranker + BM25)
        self.retriever = Retriever(
            qdrant_client = self.qdrant_client,
            chunks_dir    = chunks_dir,
        )

        print("Pipeline RAG pret !\n")

    def ask(
        self,
        question : str,
        top_k    : int = 5,
        verbose  : bool = False,
    ) -> RAGResponse:
        """
        Pose une question et retourne une reponse avec citations.

        Etapes :
            1. Retrieval : hybrid search + reranking → top_k chunks
            2. Generation : prompt + Mistral → reponse avec [N] citations

        Args:
            question : question en langage naturel (anglais recommande)
            top_k    : nombre de chunks a utiliser comme contexte
            verbose  : si True, affiche les chunks retrouves avant la reponse

        Returns:
            RAGResponse avec answer, sources, question, model
        """

        # Etape 1 : retrieval des chunks pertinents
        chunks = self.retriever.search(question, top_k=top_k)

        # Affichage optionnel des sources pour le debugging
        if verbose:
            print(f"\nSources retriev es pour : '{question}'")
            for chunk in chunks:
                print(f"  [{chunk.rank}] {chunk.paper_id} "
                      f"(score: {chunk.score:.2f}, page: {chunk.page_number})")

        # Etape 2 : generation de la reponse avec Mistral
        response = generate_answer(
            question = question,
            chunks   = chunks,
        )

        return response