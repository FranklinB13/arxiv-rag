"""
Module : chunking.py
Auteur : Franklin
Date   : 2026-04

Description :
    Decoupage des textes extraits des PDFs en chunks (morceaux)
    de taille optimale pour l'indexation et le retrieval RAG.

    Le chunking est l'etape la plus critique du pipeline RAG :
    - Des chunks trop petits = embeddings precis mais peu de contexte
    - Des chunks trop grands = beaucoup de contexte mais embeddings flous
    - Un bon overlap = les idees ne sont pas coupees en deux

    Ce module constitue la 3eme etape du pipeline RAG :
        Texte nettoye -> chunks -> (suite : embeddings)

    Strategie implementee : chunking par paragraphes avec overlap
        On respecte les separateurs naturels du texte (paragraphes)
        plutot que de couper tous les N caracteres sans reflechir.
        Quand un paragraphe est trop long, on le decoupe en sous-chunks.
        On ajoute un overlap entre chunks consecutifs pour eviter
        de couper des idees en deux.

Utilisation :
    uv run python scripts/chunk_papers.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# dataclass : permet de creer des classes de donnees proprement
# C'est le pattern moderne pour structurer des donnees en Python
# Equivalent a un dictionnaire mais avec typage et autocompletion
from dataclasses import dataclass, field

# List, Optional : annotations de types
# En Python moderne on peut ecrire list[str] directement,
# mais dataclass fonctionne mieux avec ces imports explicites
from typing import Optional

from pathlib import Path   # Gestion des chemins de fichiers
import json                # Pour sauvegarder les chunks en JSON


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Taille cible d'un chunk en nombre de mots
# 400 mots ~ 500-600 tokens (un token ~ 0.75 mot en anglais)
# C'est le sweet spot pour les modeles d'embeddings comme BGE
# Assez grand pour avoir du contexte, assez petit pour etre precis
CHUNK_SIZE_WORDS = 400

# Overlap entre chunks consecutifs en nombre de mots
# 80 mots = environ 20% de CHUNK_SIZE_WORDS
# Regle empirique : overlap = 15-25% de la taille du chunk
# Trop peu : on perd le contexte aux jonctions
# Trop : beaucoup de redondance, index plus lourd
OVERLAP_WORDS = 80

# Taille minimum d'un chunk pour qu'il soit garde
# Les chunks trop petits sont du bruit : une phrase isolee
# n'a pas assez de contexte pour etre utile en retrieval
MIN_CHUNK_WORDS = 50


# ==============================================================================
# STRUCTURE DE DONNEES : le Chunk
# ==============================================================================

@dataclass
class Chunk:
    """
    Represente un morceau de texte extrait d'un paper arXiv.

    On utilise une dataclass plutot qu'un dictionnaire pour :
        - L'autocompletion dans VSCode (on sait exactement quels champs existent)
        - La lisibilite (chunk.text plutot que chunk["text"])
        - La validation implicite (Python signale si on oublie un champ)

    Ce chunk sera ensuite :
        1. Transforme en embedding (vecteur de nombres)
        2. Stocke dans Qdrant avec ses metadonnees
        3. Retrouve lors d'une requete utilisateur

    Les metadonnees (paper_id, chunk_index, etc.) sont cruciales :
    elles nous permettront d'afficher les sources dans la reponse finale.
    "Selon le paper 2401.12345, section 3..." → necessaire pour les citations
    """

    # Identifiant unique du chunk, format : "paper_id_chunk_N"
    # Exemple : "2401_12345v1_chunk_3"
    # Utilise comme cle primaire dans Qdrant
    chunk_id: str

    # Identifiant du paper source (nom du fichier sans extension)
    # Exemple : "2401_12345v1"
    # Permet de retrouver le paper original a partir d'un chunk
    paper_id: str

    # Le texte du chunk, c'est ce qui sera transforme en embedding
    # et ce qui sera donne au LLM comme contexte
    text: str

    # Position de ce chunk dans la sequence des chunks du paper
    # Chunk 0 = debut du paper, chunk N = fin du paper
    # Utile pour reconstituer le contexte autour d'un chunk retrouve
    chunk_index: int

    # Nombre total de mots dans ce chunk
    # Utile pour les statistiques et le debugging
    word_count: int

    # Numero de la page du PDF ou commence ce chunk (approximatif)
    # None si on ne peut pas determiner la page
    # Utile pour les citations : "voir page 4 du paper"
    page_number: Optional[int] = None

    def to_dict(self) -> dict:
        """
        Convertit le chunk en dictionnaire Python.

        Necessaire pour la serialisation JSON (sauvegarde sur disque)
        et pour l'insertion dans Qdrant (qui attend des dicts).

        Returns:
            Dictionnaire avec tous les champs du chunk
        """
        return {
            "chunk_id":    self.chunk_id,
            "paper_id":    self.paper_id,
            "text":        self.text,
            "chunk_index": self.chunk_index,
            "word_count":  self.word_count,
            "page_number": self.page_number,
        }


# ==============================================================================
# FONCTIONS DE CHUNKING
# ==============================================================================

def split_into_words(text: str) -> list[str]:
    """
    Decoupe un texte en liste de mots.

    On utilise split() sans argument : il gere automatiquement
    les espaces multiples, tabulations, et sauts de ligne.
    C'est plus robuste que split(" ") qui cree des strings vides.

    Exemple :
        "hello   world\nfoo" -> ["hello", "world", "foo"]

    Args:
        text : texte a decouper

    Returns:
        Liste de mots (strings non-vides)
    """
    return text.split()


def words_to_text(words: list[str]) -> str:
    """
    Reconstitue un texte a partir d'une liste de mots.

    Inverse de split_into_words().
    On rejoint avec un espace simple entre chaque mot.

    Args:
        words : liste de mots

    Returns:
        Texte reconstitue
    """
    return " ".join(words)


def extract_page_number(text: str) -> Optional[int]:
    """
    Extrait le numero de page depuis un marqueur [PAGE N].

    Lors du parsing, on a insere des marqueurs [PAGE 1], [PAGE 2], etc.
    Cette fonction retrouve le dernier marqueur avant un chunk
    pour savoir sur quelle page du PDF il se trouve.

    Exemple :
        "[PAGE 3]\nThe attention mechanism..." -> 3

    Args:
        text : texte contenant potentiellement un marqueur [PAGE N]

    Returns:
        Numero de page (entier) ou None si pas de marqueur trouve
    """
    import re

    # On cherche le pattern [PAGE suivi d'un ou plusieurs chiffres suivi de ]
    # re.findall retourne toutes les occurrences
    matches = re.findall(r'\[PAGE (\d+)\]', text)

    if matches:
        # On prend la derniere occurrence (la page la plus recente)
        return int(matches[-1])

    return None


def chunk_text(
    text: str,
    paper_id: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = OVERLAP_WORDS,
    min_chunk_size: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """
    Decoupe un texte en chunks avec overlap.

    Algorithme :
        1. On decoupe le texte en mots individuels
        2. On avance dans la liste de mots avec une fenetre glissante
        3. La fenetre a une taille de chunk_size mots
        4. A chaque etape, on avance de (chunk_size - overlap) mots
           pour creer le chevauchement avec le chunk suivant
        5. On filtre les chunks trop petits (fin de document)

    Visualisation de l'overlap :
        Chunk 1 : [mot_0  .......  mot_399]
        Chunk 2 :              [mot_320  .......  mot_719]
                               ^--- overlap de 80 mots ---^
        Chunk 3 :                           [mot_640  .......  mot_1039]

    Args:
        text       : texte complet du paper (apres parsing et nettoyage)
        paper_id   : identifiant du paper (pour construire chunk_id)
        chunk_size : nombre de mots par chunk (defaut : CHUNK_SIZE_WORDS)
        overlap    : nombre de mots de chevauchement (defaut : OVERLAP_WORDS)
        min_chunk_size : taille minimum pour garder un chunk

    Returns:
        Liste de Chunk, dans l'ordre d'apparition dans le paper
    """

    # --- Preparation ---

    # On decoupe tout le texte en mots
    words = split_into_words(text)
    total_words = len(words)

    # Si le texte est trop court, on retourne un seul chunk
    # Pas la peine de decouper un texte de 30 mots
    if total_words <= chunk_size:
        if total_words >= min_chunk_size:
            return [Chunk(
                chunk_id    = f"{paper_id}_chunk_0",
                paper_id    = paper_id,
                text        = words_to_text(words),
                chunk_index = 0,
                word_count  = total_words,
                page_number = extract_page_number(text),
            )]
        else:
            # Texte trop court meme pour un seul chunk : on ignore
            return []

    # --- Decoupage avec fenetre glissante ---

    chunks = []       # Liste des chunks produits
    chunk_index = 0   # Compteur pour les IDs des chunks
    start = 0         # Position de debut de la fenetre courante

    # On calcule le pas d'avancement :
    # A chaque iteration, on avance de (chunk_size - overlap) mots
    # Exemple : chunk_size=400, overlap=80 -> pas=320
    # On avance de 320 mots, mais le chunk fait 400 mots
    # Donc les 80 derniers mots du chunk N = les 80 premiers du chunk N+1
    step = chunk_size - overlap

    while start < total_words:

        # Position de fin de ce chunk
        # min() pour ne pas depasser la fin du texte
        end = min(start + chunk_size, total_words)

        # Extraction des mots de ce chunk
        chunk_words = words[start:end]
        chunk_word_count = len(chunk_words)

        # On ignore les chunks trop petits
        # (typiquement le tout dernier bout du texte)
        if chunk_word_count >= min_chunk_size:

            # Reconstruction du texte du chunk
            chunk_text_str = words_to_text(chunk_words)

            # Extraction du numero de page depuis le texte
            page_num = extract_page_number(chunk_text_str)

            # Creation de l'objet Chunk
            chunk = Chunk(
                chunk_id    = f"{paper_id}_chunk_{chunk_index}",
                paper_id    = paper_id,
                text        = chunk_text_str,
                chunk_index = chunk_index,
                word_count  = chunk_word_count,
                page_number = page_num,
            )
            chunks.append(chunk)
            chunk_index += 1

        # On avance la fenetre de `step` mots
        start += step

    return chunks


# ==============================================================================
# TRAITEMENT EN BATCH DE TOUS LES FICHIERS TEXTE
# ==============================================================================

def chunk_all_papers(
    input_dir:  Path,
    output_dir: Path,
) -> dict[str, int]:
    """
    Decoupe tous les fichiers texte d'un dossier en chunks.

    Pour chaque fichier .txt dans input_dir :
        - On lit le texte nettoye
        - On le decoupe en chunks
        - On sauvegarde les chunks en JSON dans output_dir

    Format de sortie JSON :
        Un fichier par paper : "2401_12345v1_chunks.json"
        Contenu : liste de dictionnaires (un par chunk)

    On choisit JSON plutot que pickle ou CSV car :
        - Lisible par un humain (utile pour le debugging)
        - Portable (fonctionne dans n'importe quel langage)
        - Facile a charger en Python avec json.load()

    Args:
        input_dir  : dossier contenant les .txt (data/processed/)
        output_dir : dossier de destination des .json (data/chunks/)

    Returns:
        Statistiques : {"papers": nb, "chunks": nb_total, "skipped": nb}
    """

    # Creation du dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Liste de tous les fichiers .txt a traiter
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"Aucun fichier .txt trouve dans {input_dir}")
        return {"papers": 0, "chunks": 0, "skipped": 0}

    print(f"Fichiers a traiter : {len(txt_files)}")

    # Statistiques globales
    stats = {
        "papers":  0,    # nombre de papers traites
        "chunks":  0,    # nombre total de chunks produits
        "skipped": 0,    # papers deja traites (idempotence)
    }

    for txt_path in txt_files:

        # L'ID du paper = nom du fichier sans extension
        # "2401_12345v1.txt" -> "2401_12345v1"
        paper_id = txt_path.stem

        # Fichier de sortie JSON pour ce paper
        output_path = output_dir / f"{paper_id}_chunks.json"

        # Idempotence : skip si deja traite
        if output_path.exists():
            stats["skipped"] += 1
            continue

        # Lecture du texte nettoye
        # encoding="utf-8" : meme encodage que lors de la sauvegarde
        text = txt_path.read_text(encoding="utf-8")

        # Decoupage en chunks
        chunks = chunk_text(text=text, paper_id=paper_id)

        # Si le paper est trop court pour produire des chunks, on skip
        if not chunks:
            print(f"Pas de chunks pour {paper_id} (texte trop court)")
            stats["skipped"] += 1
            continue

        # Sauvegarde en JSON
        # [chunk.to_dict() for chunk in chunks] : liste de dicts
        # indent=2 : JSON lisible avec indentation
        # ensure_ascii=False : garde les caracteres speciaux (accents, etc.)
        chunks_data = [chunk.to_dict() for chunk in chunks]
        output_path.write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        stats["papers"] += 1
        stats["chunks"] += len(chunks)

    return stats