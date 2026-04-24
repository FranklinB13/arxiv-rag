"""
Module : retrieval.py
Auteur : Franklin
Date   : 2026-04

Description :
    Recherche des chunks les plus pertinents pour une question donnee.

    On implemente un retrieval hybride en deux etapes :

    Etape 1 - Recherche hybride (dense + sparse) :
        Dense  : recherche par similarite semantique via embeddings BGE
                 "Quel est le sens de cette question ?"
        Sparse : recherche par mots-cles via BM25
                 "Quels chunks contiennent exactement ces mots ?"
        Les deux se complementent : dense trouve le sens general,
        sparse trouve les correspondances exactes (noms, acronymes, chiffres)

    Etape 2 - Reranking :
        Un cross-encoder re-evalue chaque candidat en regardant
        la paire (question, chunk) ensemble.
        Plus precis que la similarite cosinus mais trop lent
        pour tout le corpus → on l'applique seulement sur les top-40.

    Ce pipeline est ce qui separe un RAG de tutoriel d'un RAG de production.

    Ce module constitue la 6eme etape du pipeline RAG :
        Question utilisateur → chunks pertinents → (suite : generation)

Utilisation :
    Importe depuis rag.py, pas appele directement.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# QdrantClient pour la recherche dense dans notre index
from qdrant_client import QdrantClient

# SentenceTransformer pour encoder la question en vecteur
# CrossEncoder pour le reranking : modele qui evalue la paire (question, chunk)
from sentence_transformers import SentenceTransformer, CrossEncoder

# BM25Okapi : implementation de BM25 (Best Match 25)
# C'est l'algorithme de recherche full-text le plus utilise au monde
# Il calcule un score de pertinence base sur la frequence des mots
from rank_bm25 import BM25Okapi

# numpy pour les calculs vectoriels
import numpy as np

# json et Path pour charger les donnees
import json
from pathlib import Path

# dataclass pour structurer les resultats de recherche
from dataclasses import dataclass
from typing import Optional


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Nom de la collection Qdrant (doit matcher vectorstore.py)
COLLECTION_NAME = "arxiv_papers"

# Nombre de resultats a recuperer pour chaque methode de recherche
# On prend intentionnellement plus que ce dont on a besoin (40 au lieu de 5)
# pour donner au reranker un bon ensemble de candidats a evaluer
TOP_K_DENSE  = 20   # top-20 par embeddings
TOP_K_SPARSE = 20   # top-20 par BM25
TOP_K_FINAL  = 5    # top-5 apres reranking : ce qu'on donne au LLM

# Modele de reranking
# "cross-encoder/ms-marco-MiniLM-L-6-v2" est un modele leger (~80MB)
# entraine specifiquement pour evaluer la pertinence question/passage
# ms-marco = dataset de 8 millions de paires question/passage de Microsoft
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ==============================================================================
# STRUCTURE DE DONNEES : SearchResult
# ==============================================================================

@dataclass
class SearchResult:
    """
    Represente un chunk retrouve lors d'une recherche.

    On structure le resultat plutot que d'utiliser un dictionnaire
    pour avoir l'autocompletion et la lisibilite dans le reste du code.

    Champs :
        chunk_id    : identifiant unique du chunk (ex: "2401_12345v1_chunk_3")
        paper_id    : identifiant du paper source (ex: "2401_12345v1")
        text        : texte du chunk (ce qui sera donne au LLM comme contexte)
        score       : score de pertinence (plus eleve = plus pertinent)
                      Apres reranking, c'est le score du cross-encoder
        page_number : numero de page dans le PDF original (pour les citations)
        rank        : position dans les resultats (1 = le plus pertinent)
    """
    chunk_id    : str
    paper_id    : str
    text        : str
    score       : float
    page_number : Optional[int] = None
    rank        : int = 0


# ==============================================================================
# RECHERCHE DENSE (EMBEDDINGS)
# ==============================================================================

def dense_search(
    query: str,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K_DENSE,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchResult]:
    """
    Recherche semantique par similarite d'embeddings dans Qdrant.

    Principe :
        1. On encode la question en vecteur avec BGE
        2. Qdrant trouve les top_k vecteurs les plus proches (HNSW)
        3. On retourne les chunks correspondants avec leurs scores

    La recherche dense est excellente pour :
        - Trouver des textes semantiquement similaires meme sans mots en commun
        - "Comment aligner les LLMs ?" → trouve des chunks sur RLHF et DPO
          meme s'ils ne contiennent pas le mot "aligner"

    Args:
        query           : question de l'utilisateur en langage naturel
        client          : client Qdrant connecte
        embed_model     : modele BGE pour encoder la question
        top_k           : nombre de resultats a retourner
        collection_name : nom de la collection Qdrant

    Returns:
        Liste de SearchResult tries par score decroissant
    """

    # Encodage de la question en vecteur
    # normalize_embeddings=True : coherence avec les vecteurs stockes dans Qdrant
    # convert_to_numpy=True puis .tolist() : Qdrant attend une liste Python
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # Recherche dans Qdrant avec l'API moderne query_points()
    # Depuis qdrant-client v1.7+, client.search() est depreciee
    # query_points() est le remplacement officiel, plus flexible
    # with_payload=True : on veut les metadonnees stockees avec chaque vecteur
    # (texte du chunk, paper_id, page_number, etc.)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    # response.points contient la liste des resultats
    # Chaque point a : id, score, payload (les metadonnees)
    hits = response.points

    # Conversion des resultats Qdrant en SearchResult
    results = []
    for hit in hits:
        results.append(SearchResult(
            chunk_id    = hit.payload.get("chunk_id", ""),
            paper_id    = hit.payload.get("paper_id", ""),
            text        = hit.payload.get("text", ""),
            score       = hit.score,
            page_number = hit.payload.get("page_number"),
        ))

    return results


# ==============================================================================
# RECHERCHE SPARSE (BM25)
# ==============================================================================

def build_bm25_index(chunks_dir: Path) -> tuple[BM25Okapi, list[dict]]:
    """
    Construit un index BM25 a partir de tous les chunks.

    BM25 (Best Match 25) :
        C'est l'algorithme de recherche full-text qui equipe la plupart
        des moteurs de recherche classiques (Elasticsearch, Lucene, etc.).
        Il calcule un score base sur :
            - TF (Term Frequency) : frequence du mot dans le chunk
            - IDF (Inverse Document Frequency) : rarete du mot dans le corpus
            - Normalisation par la longueur du document
        Un mot rare qui apparait souvent dans un chunk = tres pertinent.

    Difference avec les embeddings :
        BM25 est excellent pour les correspondances exactes :
        noms propres, acronymes, numeros de version, termes techniques.
        Les embeddings peuvent "diluer" ces details dans leur representation.
        D'ou l'interet de combiner les deux approches.

    Args:
        chunks_dir : dossier contenant les fichiers *_chunks.json

    Returns:
        Tuple (index BM25, liste de tous les chunks avec metadonnees)
        Les deux listes ont le meme ordre : all_chunks[i] correspond
        au texte tokenise[i] dans l'index BM25.
    """

    all_chunks = []   # metadonnees de tous les chunks
    all_texts  = []   # textes bruts (pour la tokenisation BM25)

    # Chargement de tous les chunks depuis les fichiers JSON
    # sorted() pour un ordre deterministe (important pour l'alignement index/metadata)
    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        chunks_data = json.loads(chunk_file.read_text(encoding="utf-8"))
        for chunk in chunks_data:
            all_chunks.append(chunk)
            all_texts.append(chunk["text"])

    # Tokenisation : BM25 travaille sur des listes de mots
    # lower() : insensible a la casse ("RLHF" et "rlhf" → meme token)
    # split() : decoupe sur les espaces (tokenisation simple mais efficace)
    tokenized_texts = [text.lower().split() for text in all_texts]

    # Construction de l'index BM25
    # BM25Okapi = variante la plus courante, issue du systeme Okapi de Londres
    bm25_index = BM25Okapi(tokenized_texts)

    return bm25_index, all_chunks


def sparse_search(
    query: str,
    bm25_index: BM25Okapi,
    all_chunks: list[dict],
    top_k: int = TOP_K_SPARSE,
) -> list[SearchResult]:
    """
    Recherche par mots-cles avec BM25.

    Args:
        query      : question de l'utilisateur
        bm25_index : index BM25 pre-construit par build_bm25_index()
        all_chunks : liste de tous les chunks (meme ordre que l'index)
        top_k      : nombre de resultats a retourner

    Returns:
        Liste de SearchResult tries par score BM25 decroissant
    """

    # Tokenisation de la query (meme traitement que l'indexation)
    tokenized_query = query.lower().split()

    # Calcul des scores BM25 pour TOUS les chunks
    # get_scores() retourne un array numpy de taille (nb_chunks,)
    # Chaque valeur = score BM25 du chunk correspondant pour cette query
    scores = bm25_index.get_scores(tokenized_query)

    # Recuperation des indices des top_k meilleurs scores
    # np.argsort() trie par ordre croissant (indices des plus petits aux plus grands)
    # [-top_k:] prend les top_k derniers (les plus grands)
    # [::-1] inverse l'ordre pour avoir decroissant
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Construction des resultats
    results = []
    for idx in top_indices:
        # On ignore les chunks avec score 0
        # (aucun mot de la query n'apparait dans le chunk)
        if scores[idx] > 0:
            chunk = all_chunks[idx]
            results.append(SearchResult(
                chunk_id    = chunk.get("chunk_id", ""),
                paper_id    = chunk.get("paper_id", ""),
                text        = chunk.get("text", ""),
                score       = float(scores[idx]),
                page_number = chunk.get("page_number"),
            ))

    return results


# ==============================================================================
# FUSION DES RESULTATS DENSE + SPARSE
# ==============================================================================

def merge_results(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
) -> list[SearchResult]:
    """
    Fusionne les resultats dense et sparse en eliminant les doublons.

    Strategie : union des deux listes, deduplication par chunk_id.
    Si un chunk apparait dans les deux listes, on garde le meilleur score.

    Pourquoi fusionner plutot que choisir l'un ou l'autre ?
        Dense seul : manque les correspondances exactes (noms, acronymes)
        Sparse seul : manque la comprehension semantique
        Fusion : les deux se completent, on donne plus de diversite au reranker

    Args:
        dense_results  : resultats de la recherche semantique (BGE + Qdrant)
        sparse_results : resultats de la recherche BM25

    Returns:
        Liste fusionnee sans doublons, non triee
        (le tri final est fait par le reranker)
    """

    # Dictionnaire pour tracker les chunks deja vus (cle = chunk_id)
    seen: dict[str, SearchResult] = {}

    # Ajout des resultats denses en premier
    for result in dense_results:
        seen[result.chunk_id] = result

    # Pour les resultats sparses, deduplication par chunk_id
    for result in sparse_results:
        if result.chunk_id not in seen:
            # Nouveau chunk : on l'ajoute
            seen[result.chunk_id] = result
        else:
            # Chunk deja vu : on garde le meilleur score
            if result.score > seen[result.chunk_id].score:
                seen[result.chunk_id] = result

    return list(seen.values())


# ==============================================================================
# RERANKING (CROSS-ENCODER)
# ==============================================================================

def rerank_results(
    query: str,
    candidates: list[SearchResult],
    reranker: CrossEncoder,
    top_k: int = TOP_K_FINAL,
) -> list[SearchResult]:
    """
    Reclasse les candidats avec un cross-encoder pour plus de precision.

    Difference fondamentale bi-encoder vs cross-encoder :

        Bi-encoder (BGE) :
            encode(question) → vecteur_q
            encode(chunk)    → vecteur_c
            score = cosine(vecteur_q, vecteur_c)
            ✅ Rapide : on pre-calcule les vecteurs des chunks
            ❌ Moins precis : question et chunk ne "se voient" pas

        Cross-encoder (reranker) :
            score = modele(question + chunk ensemble)
            Le modele voit la paire complete → comprend la relation
            ✅ Tres precis : "ce chunk repond-il a CETTE question ?"
            ❌ Lent : pas de pre-calcul possible, depend de la question
            → C'est pourquoi on l'applique seulement sur ~40 candidats

    Args:
        query      : question de l'utilisateur
        candidates : chunks candidats issus de la fusion dense+sparse
        reranker   : modele CrossEncoder charge
        top_k      : nombre de resultats finaux a retourner

    Returns:
        top_k meilleurs SearchResult apres reranking, par score decroissant
    """

    if not candidates:
        return []

    # Construction des paires (question, chunk) pour le cross-encoder
    # Le modele attend une liste de tuples [query, passage]
    pairs = [(query, result.text) for result in candidates]

    # Inference du cross-encoder sur toutes les paires
    # predict() retourne un score par paire
    # Plus le score est eleve, plus le chunk repond bien a la question
    scores = reranker.predict(pairs)

    # Mise a jour des scores dans les objets SearchResult
    for result, score in zip(candidates, scores):
        result.score = float(score)

    # Tri par score decroissant et selection des top_k meilleurs
    reranked    = sorted(candidates, key=lambda r: r.score, reverse=True)
    top_results = reranked[:top_k]

    # Attribution des rangs (1 = le plus pertinent)
    for i, result in enumerate(top_results):
        result.rank = i + 1

    return top_results


# ==============================================================================
# CLASSE RETRIEVER : PIPELINE COMPLET
# ==============================================================================

class Retriever:
    """
    Encapsule le pipeline de retrieval hybride complet.

    Pourquoi une classe plutot que des fonctions isolees ?
        Les modeles (BGE, reranker) et l'index BM25 sont couteux a charger.
        On veut les charger UNE seule fois au demarrage, puis les reutiliser
        pour toutes les requetes sans rechargement.
        Une classe garde ces ressources en memoire dans ses attributs (self.xxx).

    Utilisation typique :
        retriever = Retriever(client, chunks_dir)   # charge les modeles (1x)
        results   = retriever.search("How does RLHF work?")  # rapide
        results2  = retriever.search("What is RAG?")         # rapide aussi
    """

    def __init__(
        self,
        qdrant_client       : QdrantClient,
        chunks_dir          : Path,
        embed_model_name    : str = "BAAI/bge-small-en-v1.5",
        reranker_model_name : str = RERANKER_MODEL,
    ):
        """
        Charge tous les modeles et construit les index au demarrage.

        Chargements effectues une seule fois :
            1. Modele BGE   : depuis le cache HuggingFace (~2s)
            2. Cross-encoder: depuis le cache HuggingFace (~2s)
            3. Index BM25   : construit en memoire depuis les JSON (~5s)

        Args:
            qdrant_client       : client Qdrant avec l'index vectoriel
            chunks_dir          : dossier des chunks JSON (pour BM25)
            embed_model_name    : modele HuggingFace pour les embeddings
            reranker_model_name : modele HuggingFace pour le reranking
        """

        print("Initialisation du Retriever...")

        # Stockage du client Qdrant pour les recherches denses
        self.client = qdrant_client

        # Chargement du modele d'embeddings BGE
        # Deja telecharge a l'etape 4, charge depuis le cache
        print("  Chargement du modele d'embeddings BGE...")
        self.embed_model = SentenceTransformer(embed_model_name)

        # Chargement du modele de reranking cross-encoder
        # Premier lancement : telecharge ~80MB
        # Suivants : charge depuis le cache
        print("  Chargement du modele de reranking...")
        self.reranker = CrossEncoder(reranker_model_name)

        # Construction de l'index BM25 en memoire
        # Charge tous les chunks JSON et construit la structure BM25
        print("  Construction de l'index BM25...")
        self.bm25_index, self.all_chunks = build_bm25_index(chunks_dir)
        print(f"  Index BM25 : {len(self.all_chunks)} chunks indexes")

        print("Retriever pret !\n")

    def search(
        self,
        query : str,
        top_k : int = TOP_K_FINAL,
    ) -> list[SearchResult]:
        """
        Execute le pipeline complet de retrieval pour une question.

        Pipeline en 4 etapes :
            1. Dense search  → top-20 par embeddings (Qdrant)
            2. Sparse search → top-20 par mots-cles (BM25)
            3. Fusion        → union des deux (~40 candidats max)
            4. Reranking     → cross-encoder sur les ~40 → top-5 final

        Args:
            query : question en langage naturel
            top_k : nombre de chunks finaux a retourner

        Returns:
            Liste de top_k SearchResult tries par pertinence decroissante
            Le rank 1 = le chunk le plus pertinent pour la question
        """

        # Etape 1 : recherche semantique dense via Qdrant
        dense_results = dense_search(
            query       = query,
            client      = self.client,
            embed_model = self.embed_model,
        )

        # Etape 2 : recherche par mots-cles via BM25
        sparse_results = sparse_search(
            query      = query,
            bm25_index = self.bm25_index,
            all_chunks = self.all_chunks,
        )

        # Etape 3 : fusion des deux listes sans doublons
        candidates = merge_results(dense_results, sparse_results)

        # Etape 4 : reranking par cross-encoder pour la precision finale
        final_results = rerank_results(
            query      = query,
            candidates = candidates,
            reranker   = self.reranker,
            top_k      = top_k,
        )

        return final_results