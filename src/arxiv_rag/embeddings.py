"""
Module : embeddings.py
Auteur : Franklin
Date   : 2026-04

Description :
    Transformation des chunks de texte en vecteurs d'embeddings.

    Un embedding est une representation numerique dense du sens d'un texte.
    Les textes semantiquement proches ont des vecteurs proches dans
    l'espace mathematique (mesure par la similarite cosinus).

    On utilise le modele BGE (BAAI General Embedding) :
        - Modele : BAAI/bge-small-en-v1.5
        - Dimensions : 384 (chaque texte → vecteur de 384 nombres)
        - Licence : open source, gratuit, utilisable en production
        - Performance : top du benchmark MTEB pour sa taille

    Ce module constitue la 4eme etape du pipeline RAG :
        Chunks JSON → vecteurs d'embeddings → (suite : Qdrant)

    Pourquoi BGE et pas OpenAI text-embedding-3-small ?
        - Gratuit vs ~0.02$/1000 tokens
        - Local vs donnees envoyees sur internet
        - Comparable en qualite pour notre cas d'usage

Utilisation :
    uv run python scripts/build_embeddings.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# SentenceTransformer : classe principale de sentence-transformers
# Elle encapsule le modele de deep learning et expose une API simple :
#   model.encode(textes) → vecteurs numpy
from sentence_transformers import SentenceTransformer

# numpy : bibliotheque de calcul numerique
# Les embeddings sont des arrays numpy : efficaces en memoire et rapides
import numpy as np

# json : pour lire les chunks sauvegardes a l'etape precedente
import json

# Path : gestion des chemins de fichiers
from pathlib import Path

# tqdm : barre de progression (on va traiter 3377 chunks, ca prend du temps)
from tqdm import tqdm


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Nom du modele HuggingFace a utiliser
# "BAAI/bge-small-en-v1.5" = organisation/nom-du-modele
# La premiere fois, sentence-transformers le telecharge (~130MB)
# et le met en cache. Les fois suivantes, il est charge depuis le cache.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Taille du batch : nombre de chunks traites en parallele par le modele
# Plus le batch est grand, plus c'est rapide (jusqu'a la limite RAM/VRAM)
# 32 est un bon compromis pour un CPU sans GPU dedie
# Si tu as une erreur de memoire, diminue cette valeur
BATCH_SIZE = 32

# BGE recommande d'ajouter ce prefixe aux textes a indexer
# (pas aux requetes). Ca ameliore la qualite des embeddings.
# C'est specifique a BGE, pas a tous les modeles.
BGE_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "


# ==============================================================================
# CHARGEMENT DU MODELE
# ==============================================================================

def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Charge le modele d'embeddings depuis HuggingFace ou le cache local.

    Comportement :
        - 1ere execution : telecharge le modele (~130MB) depuis HuggingFace
          et le met en cache dans ~/.cache/huggingface/
        - Executions suivantes : charge depuis le cache (rapide, ~2 secondes)

    Pourquoi mettre le chargement dans une fonction separee ?
        - Separation des responsabilites : chaque fonction fait une chose
        - Testabilite : on peut mocker cette fonction dans les tests
        - Reutilisabilite : on charge le modele une fois, on le passe partout

    Args:
        model_name : identifiant HuggingFace du modele

    Returns:
        Modele charge, pret a encoder des textes
    """
    print(f"Chargement du modele : {model_name}")
    print("(Telechargement au premier lancement, ~130MB)")

    # SentenceTransformer gere automatiquement :
    #   - Le telechargement depuis HuggingFace si pas en cache
    #   - L'utilisation du GPU si disponible (sinon CPU)
    #   - La tokenisation du texte avant l'encodage
    model = SentenceTransformer(model_name)

    print(f"Modele charge. Dimensions : {model.get_sentence_embedding_dimension()}")
    return model


# ==============================================================================
# GENERATION D'EMBEDDINGS
# ==============================================================================

def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
    is_query: bool = False,
) -> np.ndarray:
    """
    Transforme une liste de textes en matrice d'embeddings.

    Traitement en batches :
        On ne passe pas tous les textes d'un coup au modele.
        On les traite par groupes de batch_size.
        Pourquoi ? Le modele de deep learning a besoin de charger
        les textes en memoire GPU/CPU. Si on envoie 3377 chunks d'un coup,
        on risque de saturer la memoire.
        En batches de 32, on reste dans des limites raisonnables.

    Prefixe BGE :
        BGE utilise deux modes :
            - "passage" : pour les textes a indexer (nos chunks)
              → on ajoute le prefixe BGE_PASSAGE_PREFIX
            - "query" : pour les questions de l'utilisateur
              → pas de prefixe
        Cette distinction ameliore la qualite du retrieval.
        C'est une specificite de BGE documentee sur HuggingFace.

    Args:
        texts      : liste de textes a encoder
        model      : modele SentenceTransformer charge
        batch_size : nombre de textes traites en parallele
        is_query   : True si c'est une question utilisateur,
                     False si c'est un chunk a indexer

    Returns:
        Matrice numpy de shape (len(texts), 384)
        Chaque ligne = embedding d'un texte
        embedding[i] correspond a texts[i]
    """

    # Si c'est un passage (chunk a indexer), on ajoute le prefixe BGE
    # Si c'est une query (question utilisateur), pas de prefixe
    if not is_query:
        texts = [BGE_PASSAGE_PREFIX + t for t in texts]

    # model.encode() fait tout le travail :
    #   1. Tokenisation : texte → tokens (sous-mots numeriques)
    #   2. Forward pass : tokens → vecteur de 384 dimensions via le reseau
    #   3. Normalisation : le vecteur est normalise (norme = 1)
    #      La normalisation permet d'utiliser le produit scalaire
    #      au lieu de la similarite cosinus (plus rapide, resultat identique)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,      # affiche une barre de progression
        normalize_embeddings=True,   # normalisation L2 (recommande pour BGE)
        convert_to_numpy=True,       # retourne un array numpy
    )

    return embeddings


# ==============================================================================
# TRAITEMENT EN BATCH DE TOUS LES CHUNKS
# ==============================================================================

def embed_all_chunks(
    chunks_dir: Path,
    output_dir: Path,
    model: SentenceTransformer,
) -> dict[str, int]:
    """
    Genere les embeddings de tous les chunks et les sauvegarde.

    Format de sauvegarde :
        Pour chaque paper, on sauvegarde deux fichiers :
            - "2401_12345v1_embeddings.npy" : matrice numpy des vecteurs
              Shape : (nb_chunks, 384)
            - "2401_12345v1_metadata.json"  : liste des metadonnees
              Chaque element correspond a la ligne de meme index dans .npy

        Pourquoi separer embeddings et metadonnees ?
            - Les embeddings sont des nombres flottants → format numpy (.npy)
              efficace en memoire et rapide a charger
            - Les metadonnees sont du texte → format JSON lisible
            - A l'etape suivante, on charge les deux et on les insere dans Qdrant

    Args:
        chunks_dir : dossier contenant les chunks JSON (data/chunks/)
        output_dir : dossier de destination (data/embeddings/)
        model      : modele d'embeddings charge

    Returns:
        Statistiques : {"papers": nb, "chunks": nb_total, "skipped": nb}
    """

    # Creation du dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Liste de tous les fichiers de chunks
    chunk_files = list(chunks_dir.glob("*_chunks.json"))

    if not chunk_files:
        print(f"Aucun fichier de chunks trouve dans {chunks_dir}")
        return {"papers": 0, "chunks": 0, "skipped": 0}

    print(f"Papers a traiter : {len(chunk_files)}")

    stats = {"papers": 0, "chunks": 0, "skipped": 0}

    # On itere sur chaque paper
    # tqdm affiche la progression au niveau des papers
    for chunk_file in tqdm(chunk_files, desc="Papers traites"):

        # Extraction de l'ID du paper depuis le nom de fichier
        # "2401_12345v1_chunks.json" → "2401_12345v1"
        paper_id = chunk_file.stem.replace("_chunks", "")

        # Fichiers de sortie pour ce paper
        embeddings_path = output_dir / f"{paper_id}_embeddings.npy"
        metadata_path   = output_dir / f"{paper_id}_metadata.json"

        # Idempotence : skip si deja traite
        if embeddings_path.exists() and metadata_path.exists():
            stats["skipped"] += 1
            continue

        # Chargement des chunks depuis le JSON
        chunks_data = json.loads(chunk_file.read_text(encoding="utf-8"))

        if not chunks_data:
            continue

        # Extraction des textes et des metadonnees
        # On separe les deux car embed_texts n'a besoin que des textes
        texts    = [chunk["text"] for chunk in chunks_data]
        metadata = chunks_data   # on garde tous les champs comme metadonnees

        # Generation des embeddings pour tous les chunks de ce paper
        # is_query=False car ce sont des passages a indexer (pas des questions)
        embeddings = embed_texts(texts, model, is_query=False)

        # Sauvegarde de la matrice d'embeddings au format numpy
        # np.save() est plus efficace que JSON pour des tableaux de flottants
        # Le fichier .npy contient la shape et les donnees brutes
        np.save(embeddings_path, embeddings)

        # Sauvegarde des metadonnees au format JSON
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        stats["papers"] += 1
        stats["chunks"] += len(chunks_data)

    return stats