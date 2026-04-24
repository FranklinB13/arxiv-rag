"""
Module : vectorstore.py
Auteur : Franklin
Date   : 2026-04

Description :
    Stockage et recherche des embeddings dans Qdrant.

    Qdrant est une base de donnees vectorielle : elle est optimisee
    pour stocker des vecteurs et trouver rapidement les plus proches
    voisins d'un vecteur de requete.

    Pourquoi une vector DB plutot qu'une recherche naive ?
        Recherche naive : pour trouver les 5 chunks les plus proches
        parmi 3377, on calcule 3377 distances. Sur 1 million de chunks,
        ca devient trop lent.
        Qdrant utilise l'algorithme HNSW (Hierarchical Navigable Small
        World) qui trouve les voisins les plus proches en temps
        quasi-logarithmique. C'est des ordres de grandeur plus rapide.

    On utilise Qdrant en mode local (pas de serveur Docker necessaire) :
        - Les donnees sont stockees dans un dossier local (qdrant_storage/)
        - Qdrant tourne dans le meme processus Python
        - Facile a switcher vers un vrai serveur en production

    Ce module constitue la 5eme etape du pipeline RAG :
        Embeddings .npy + metadata .json -> Qdrant -> pret pour le retrieval

Utilisation :
    uv run python scripts/build_index.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# QdrantClient : client Python pour interagir avec Qdrant
# En mode local, il cree une base de donnees dans un dossier
from qdrant_client import QdrantClient

# VectorParams : configuration des vecteurs (taille, metrique de distance)
# Distance : enumeration des metriques disponibles (COSINE, DOT, EUCLID)
# PointStruct : structure d'un point a inserer (id + vecteur + payload)
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

import numpy as np      # Pour charger les fichiers .npy d'embeddings
import json             # Pour charger les metadonnees JSON
from pathlib import Path
from tqdm import tqdm   # Barre de progression


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Nom de la "collection" dans Qdrant
# Une collection = un ensemble de vecteurs de meme dimension
# Equivalent a une table dans une base de donnees relationnelle
COLLECTION_NAME = "arxiv_papers"

# Dimension des vecteurs BGE small
# Doit correspondre exactement au modele utilise pour generer les embeddings
# BGE-small-en-v1.5 produit des vecteurs de 384 dimensions
VECTOR_SIZE = 384

# Dossier ou Qdrant va stocker ses donnees sur le disque
# Ce dossier est dans .gitignore (pas besoin de le committer)
QDRANT_PATH = "./qdrant_storage"

# Nombre de points inseres en une seule operation
# Inserer point par point serait tres lent (3377 appels reseau)
# En batch de 256, on fait ~14 appels au lieu de 3377
UPSERT_BATCH_SIZE = 256


# ==============================================================================
# CONNEXION A QDRANT
# ==============================================================================

def get_qdrant_client(path: str = QDRANT_PATH) -> QdrantClient:
    """
    Cree et retourne un client Qdrant en mode local.

    Mode local vs mode serveur :
        Mode local  (path=...) : Qdrant stocke les donnees dans un dossier
                                  sur le disque. Pas de serveur necessaire.
                                  Parfait pour le developpement.
        Mode serveur (url=...)  : Qdrant tourne comme un service separe
                                  (ex: Docker). Utilise en production.
        Changer de mode : remplacer QdrantClient(path=...) par
                          QdrantClient(url="http://localhost:6333")
                          Les methodes restent identiques.

    Args:
        path : dossier de stockage local pour Qdrant

    Returns:
        Client Qdrant connecte et pret a utiliser
    """
    # QdrantClient(path=...) = mode local
    # Qdrant cree le dossier s'il n'existe pas
    client = QdrantClient(path=path)
    return client


# ==============================================================================
# CREATION DE LA COLLECTION
# ==============================================================================

def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
    recreate: bool = False,
) -> None:
    """
    Cree une collection Qdrant pour stocker nos embeddings.

    Une collection Qdrant = un index vectoriel avec :
        - Une dimension fixe (384 pour BGE-small)
        - Une metrique de distance (COSINE pour nous)
        - Un stockage des metadonnees (payload) par point

    Metrique COSINE vs DOT vs EUCLID :
        COSINE  : mesure l'angle entre vecteurs, insensible a la norme
                  → standard pour les embeddings de texte
                  → nos embeddings sont normalises, donc COSINE = DOT
        DOT     : produit scalaire, tient compte de la norme
        EUCLID  : distance euclidienne, moins adapte aux embeddings texte

    Args:
        client          : client Qdrant connecte
        collection_name : nom de la collection a creer
        vector_size     : dimension des vecteurs (doit matcher le modele)
        recreate        : si True, supprime et recrée la collection
                          si False, skip si elle existe deja
    """

    # Verification si la collection existe deja
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            # On supprime la collection existante pour repartir de zero
            # Utile si on change de modele d'embeddings ou de configuration
            print(f"Suppression de la collection existante : {collection_name}")
            client.delete_collection(collection_name)
        else:
            # La collection existe et on ne veut pas la recreer : on skip
            print(f"Collection '{collection_name}' deja existante, skip.")
            return

    # Creation de la collection avec notre configuration
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,      # dimension des vecteurs
            distance=Distance.COSINE,  # metrique de distance
        ),
    )
    print(f"Collection '{collection_name}' creee avec succes.")


# ==============================================================================
# INSERTION DES POINTS
# ==============================================================================

def insert_embeddings(
    client: QdrantClient,
    embeddings_dir: Path,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """
    Insere tous les embeddings et leurs metadonnees dans Qdrant.

    Structure d'un point Qdrant :
        - id      : identifiant unique numerique (entier)
                    Qdrant requiert des IDs numeriques ou UUID
                    On va utiliser un compteur global
        - vector  : liste de 384 flottants (l'embedding du chunk)
        - payload : dictionnaire de metadonnees (paper_id, texte, etc.)
                    Stocke avec le vecteur, retourne lors du retrieval

    Insertion en batch :
        On groupe les points par UPSERT_BATCH_SIZE avant d'inserer.
        "Upsert" = insert + update : si un point avec cet ID existe deja,
        il est mis a jour. Sinon il est cree. Ca rend l'operation idempotente.

    Args:
        client          : client Qdrant connecte
        embeddings_dir  : dossier contenant les .npy et .json
        collection_name : nom de la collection cible

    Returns:
        Statistiques : {"papers": nb, "points": nb_total, "skipped": nb}
    """

    # Liste des fichiers d'embeddings a inserer
    embedding_files = list(embeddings_dir.glob("*_embeddings.npy"))

    if not embedding_files:
        print(f"Aucun fichier d'embeddings trouve dans {embeddings_dir}")
        return {"papers": 0, "points": 0, "skipped": 0}

    print(f"Papers a indexer : {len(embedding_files)}")

    stats        = {"papers": 0, "points": 0, "skipped": 0}
    global_id    = 0    # Compteur global pour les IDs des points Qdrant
    batch_points = []   # Buffer de points en attente d'insertion

    for emb_file in tqdm(embedding_files, desc="Indexation"):

        # Reconstruction du chemin vers les metadonnees correspondantes
        # "2401_12345v1_embeddings.npy" -> "2401_12345v1_metadata.json"
        paper_id      = emb_file.stem.replace("_embeddings", "")
        metadata_file = embeddings_dir / f"{paper_id}_metadata.json"

        # Si les metadonnees sont absentes, on ne peut pas inserer ce paper
        if not metadata_file.exists():
            print(f"\nMetadonnees manquantes pour {paper_id}, skip.")
            stats["skipped"] += 1
            continue

        # Chargement de la matrice d'embeddings
        # Shape attendue : (nb_chunks, 384)
        embeddings = np.load(emb_file)

        # Chargement des metadonnees
        metadata_list = json.loads(metadata_file.read_text(encoding="utf-8"))

        # Verification de coherence : autant d'embeddings que de metadonnees
        if len(embeddings) != len(metadata_list):
            print(f"\nIncoh, skip {paper_id}.")
            stats["skipped"] += 1
            continue

        # Construction des points Qdrant pour ce paper
        for embedding, metadata in zip(embeddings, metadata_list):

            # PointStruct = la structure qu'attend Qdrant pour un point
            point = PointStruct(
                id      = global_id,
                # .tolist() convertit le array numpy en liste Python
                # Qdrant n'accepte pas les arrays numpy directement
                vector  = embedding.tolist(),
                # Le payload contient toutes les metadonnees du chunk
                # On peut filtrer par n'importe quel champ du payload
                # lors d'une recherche (ex: filtrer par paper_id)
                payload = metadata,
            )
            batch_points.append(point)
            global_id += 1

            # Quand le batch est plein, on l'insere dans Qdrant
            if len(batch_points) >= UPSERT_BATCH_SIZE:
                client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                )
                # On vide le buffer apres insertion
                batch_points = []

        stats["papers"] += 1
        stats["points"] += len(embeddings)

    # Insertion du dernier batch (qui peut etre plus petit que UPSERT_BATCH_SIZE)
    # Sans ca, les derniers points ne seraient jamais inseres
    if batch_points:
        client.upsert(
            collection_name=collection_name,
            points=batch_points,
        )

    return stats


# ==============================================================================
# VERIFICATION DE L'INDEX
# ==============================================================================

def get_collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Affiche les informations sur la collection Qdrant.

    Utile pour verifier que l'indexation s'est bien passee :
        - Nombre de points inseres
        - Configuration des vecteurs
        - Statut de l'index

    Args:
        client          : client Qdrant connecte
        collection_name : nom de la collection a inspecter
    """
    info = client.get_collection(collection_name)

    print(f"\nCollection : {collection_name}")
    print(f"  Points indexes   : {info.points_count}")
    print(f"  Vecteurs config  : {info.config.params.vectors}")
    print(f"  Statut           : {info.status}")