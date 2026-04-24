"""
Module : generation.py
Auteur : Franklin
Date   : 2026-04

Description :
    Generation de reponses via l'API Groq (LLM cloud gratuit).

    On utilise Groq plutot qu'Ollama car :
        - Gratuit : tier free genereux (14 400 requetes/jour)
        - Ultra-rapide : inference sur puces LPU proprietaires de Groq
          (~500 tokens/seconde vs ~10 tokens/seconde en local CPU)
        - Zero espace disque : les modeles tournent sur leurs serveurs
        - Credible en entretien : Groq est tres utilise en prod

    On appelle l'API Groq directement avec httpx (deja installe)
    sans SDK supplementaire. L'API Groq est compatible OpenAI :
    meme format de requete/reponse que l'API OpenAI.

    Ce module constitue la 7eme et derniere etape du pipeline RAG :
        Chunks pertinents → prompt → LLM Groq → reponse avec citations

Utilisation :
    Importe depuis rag.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# httpx : client HTTP moderne, deja installe dans notre env
# On l'utilise pour appeler l'API REST de Groq
import httpx

# Pour charger la cle API depuis le fichier .env
from dotenv import load_dotenv
import os

# Pour typer les resultats du retrieval
from arxiv_rag.retrieval import SearchResult

# dataclass pour structurer la reponse finale
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Chargement des variables d'environnement depuis .env
# Doit etre appele avant os.getenv()
load_dotenv()

# URL de l'API Groq
# Compatible avec l'API OpenAI : meme format de requete/reponse
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Modele a utiliser sur Groq
# "llama-3.1-8b-instant" : rapide et gratuit, tres bon pour notre usage
# Autres options gratuites sur Groq :
#   - "llama-3.3-70b-versatile" : plus puissant mais plus lent
#   - "mixtral-8x7b-32768"      : bon pour les textes longs
MODEL_NAME = "llama-3.1-8b-instant"

# Timeout en secondes
# Groq est tres rapide (~2-5s), 30s est largement suffisant
TIMEOUT_SECONDS = 30

# Nombre maximum de chunks a inclure dans le contexte
MAX_CONTEXT_CHUNKS = 5

# Temperature : 0.1 = reponses factuelles et deterministiques
TEMPERATURE = 0.1

# Nombre maximum de tokens dans la reponse
MAX_TOKENS = 1024


# ==============================================================================
# STRUCTURE DE DONNEES : RAGResponse
# ==============================================================================

@dataclass
class RAGResponse:
    """
    Represente une reponse complete du systeme RAG.

    Champs :
        question : la question posee par l'utilisateur
        answer   : la reponse generee par le LLM
        sources  : liste des chunks utilises comme contexte
        model    : nom du modele utilise
    """
    question : str
    answer   : str
    sources  : list[SearchResult]
    model    : str


# ==============================================================================
# CONSTRUCTION DU CONTEXTE
# ==============================================================================

def format_context(chunks: list[SearchResult]) -> str:
    """
    Formate les chunks en un bloc de contexte lisible pour le LLM.

    Format :
        [1] (paper: 2401_12345v1, page: 3)
        <texte du chunk>

        [2] (paper: 2401_67890v1, page: 7)
        <texte du chunk>

    Les numeros [1], [2]... permettent au LLM de citer precisement
    quelle source supporte quelle affirmation dans sa reponse.

    Args:
        chunks : liste de SearchResult tries par pertinence decroissante

    Returns:
        Chaine de texte formatee representant le contexte documentaire
    """

    context_parts = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):

        # Nettoyage des marqueurs [PAGE N] inseres lors du parsing
        clean_text = chunk.text
        if chunk.page_number:
            clean_text = clean_text.replace(
                f"[PAGE {chunk.page_number}]", ""
            ).strip()

        # Reference de la source pour la tracabilite
        if chunk.page_number:
            source_ref = f"paper: {chunk.paper_id}, page: {chunk.page_number}"
        else:
            source_ref = f"paper: {chunk.paper_id}"

        context_parts.append(f"[{i}] ({source_ref})\n{clean_text}")

    return "\n\n".join(context_parts)


# ==============================================================================
# CONSTRUCTION DU PROMPT
# ==============================================================================

def build_messages(question: str, context: str) -> list[dict]:
    """
    Construit la liste de messages pour l'API Groq (format OpenAI).

    Format :
        [
            {"role": "system", "content": "...instructions..."},
            {"role": "user",   "content": "...question + contexte..."}
        ]

    Le system prompt impose les regles strictes du RAG pour
    eviter les hallucinations et forcer les citations.

    Args:
        question : question de l'utilisateur
        context  : contexte formate par format_context()

    Returns:
        Liste de messages au format API OpenAI/Groq
    """

    system_prompt = """You are a scientific assistant specialized in NLP and Machine Learning research.

Your role is to answer questions based EXCLUSIVELY on the provided research paper excerpts.

Rules you MUST follow:
1. Base your answer ONLY on the provided documents. Do not use your training knowledge.
2. Cite your sources using [N] notation (e.g., "According to [1]..." or "...as shown in [2]").
3. If the answer cannot be found in the provided documents, say explicitly:
   "I could not find information about this in the provided papers."
4. Be precise and technical. Your audience is ML engineers and researchers.
5. Structure your answer clearly with the most important information first.

Never invent information. Never answer from memory. Only use what is in the documents."""

    user_prompt = f"""Here are excerpts from research papers that may help answer your question:

---
{context}
---

Question: {question}

Please answer based exclusively on the excerpts above, citing sources with [N] notation."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


# ==============================================================================
# APPEL A L'API GROQ
# ==============================================================================

def generate_answer(
    question : str,
    chunks   : list[SearchResult],
) -> RAGResponse:
    """
    Genere une reponse via l'API Groq.

    L'API Groq est compatible OpenAI : on envoie une requete POST
    avec les messages et les options, on recupere la reponse en JSON.

    Authentification :
        La cle API est lue depuis GROQ_API_KEY dans le fichier .env.
        Elle est passee dans le header HTTP "Authorization: Bearer <key>".
        C'est le standard d'authentification des APIs REST.

    Args:
        question : question de l'utilisateur
        chunks   : chunks recuperes par le retrieval

    Returns:
        RAGResponse avec la reponse et les sources

    Raises:
        ValueError    : si GROQ_API_KEY n'est pas definie
        httpx.Error   : si l'appel API echoue
    """

    # Recuperation de la cle API depuis les variables d'environnement
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY non trouvee. "
            "Verifie que ton .env contient : GROQ_API_KEY=ta_cle"
        )

    # Formatage du contexte et construction des messages
    context  = format_context(chunks)
    messages = build_messages(question, context)

    # Corps de la requete HTTP
    request_body = {
        "model"      : MODEL_NAME,
        "messages"   : messages,
        "temperature": TEMPERATURE,
        "max_tokens" : MAX_TOKENS,
    }

    # Headers HTTP
    # Authorization: Bearer <key> = standard pour les APIs REST
    # Content-Type: application/json = on envoie du JSON
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
    }

    try:
        # Appel HTTP POST vers l'API Groq
        response = httpx.post(
            url     = GROQ_API_URL,
            json    = request_body,
            headers = headers,
            timeout = TIMEOUT_SECONDS,
        )

        # Verification du code HTTP (leve une exception si 4xx ou 5xx)
        response.raise_for_status()

        # Parsing de la reponse JSON
        # Format OpenAI : choices[0].message.content contient le texte
        data   = response.json()
        answer = data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as e:
        # Erreur HTTP (401 = cle invalide, 429 = rate limit, etc.)
        raise RuntimeError(
            f"Erreur API Groq ({e.response.status_code}) : {e.response.text}"
        )

    except httpx.TimeoutException:
        raise TimeoutError(
            f"Groq a depasse le timeout de {TIMEOUT_SECONDS}s."
        )

    return RAGResponse(
        question = question,
        answer   = answer,
        sources  = chunks,
        model    = MODEL_NAME,
    )