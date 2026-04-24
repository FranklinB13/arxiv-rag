"""
Script : evaluate.py
Auteur : Franklin
Date   : 2026-04

Description :
    Évaluation quantitative du pipeline RAG.

    On implémente les métriques manuellement via l'API Groq
    plutôt que d'utiliser Ragas directement.
    Pourquoi ? Ragas dépend d'instructor qui dépend de mistralai,
    une lib avec un conflit de dépendances dans notre environnement.

    Métriques calculées :

        Faithfulness (fidélité) :
            On demande au LLM : "Chaque affirmation de cette réponse
            est-elle supportée par ces documents ?"
            Score = nb affirmations supportées / nb affirmations total
            Mesure les hallucinations.

        Answer Relevancy (pertinence) :
            On demande au LLM : "Cette réponse répond-elle bien
            à la question ?"
            Score entre 0 et 1.

        Context Recall (rappel) :
            On demande au LLM : "Les éléments de la réponse de référence
            sont-ils couverts par les chunks récupérés ?"
            Score = nb éléments couverts / nb éléments total

Utilisation :
    uv run python scripts/evaluate.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import json
import os
import time
import httpx
from pathlib import Path
from dotenv import load_dotenv
from arxiv_rag.rag import RAGPipeline


# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()

# URL et modèle Groq (même config que generation.py)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME   = "llama-3.1-8b-instant"
TIMEOUT      = 30

# Délai entre appels pour respecter le rate limit Groq
# On fait beaucoup d'appels dans ce script (collecte + 3 métriques × 20 questions)
DELAY_COLLECT  = 20   # secondes entre chaque question lors de la collecte
DELAY_METRIC   = 5    # secondes entre chaque appel de métrique


# ==============================================================================
# DATASET D'ÉVALUATION
# ==============================================================================

# 20 questions avec leurs réponses de référence
# La réponse de référence = ce qu'on attend idéalement comme réponse
# Elle sert à mesurer le Context Recall (est-ce que le retrieval
# a trouvé les chunks qui permettraient de répondre à ça ?)
EVAL_DATASET = [
    {
        "question" : "What are the main components of a RAG system?",
        "reference": "A RAG system consists of a retrieval component that fetches relevant documents and a generation component that produces answers conditioned on the retrieved context."
    },
    {
        "question" : "What are the limitations of RAG compared to fine-tuning?",
        "reference": "RAG depends on retrieval quality and can fail with noisy or incomplete contexts, while fine-tuning requires substantial computational resources and may not generalize to new knowledge."
    },
    {
        "question" : "How does dense retrieval work in RAG systems?",
        "reference": "Dense retrieval encodes queries and documents into dense vector representations and retrieves documents by computing similarity between query and document vectors."
    },
    {
        "question" : "What is hybrid retrieval in RAG?",
        "reference": "Hybrid retrieval combines dense vector search with sparse keyword-based search such as BM25 to improve retrieval quality by capturing both semantic similarity and exact keyword matches."
    },
    {
        "question" : "What is RLHF and why is it used?",
        "reference": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align language models with human preferences by training a reward model on human comparisons and using it to fine-tune the LLM."
    },
    {
        "question" : "What are the main steps of RLHF training?",
        "reference": "RLHF involves three steps: supervised fine-tuning on demonstrations, training a reward model on human preference comparisons, and optimizing the language model using PPO."
    },
    {
        "question" : "What is DPO and how does it differ from RLHF?",
        "reference": "DPO (Direct Preference Optimization) directly optimizes the language model on preference data without requiring a separate reward model or reinforcement learning."
    },
    {
        "question" : "What are the main techniques to reduce hallucinations in LLMs?",
        "reference": "Main techniques include RAG to ground responses in external knowledge, fine-tuning on factual data, chain-of-thought prompting, and post-hoc verification methods."
    },
    {
        "question" : "How can RAG reduce hallucinations?",
        "reference": "RAG reduces hallucinations by grounding the model's responses in retrieved context, forcing it to base answers on provided documents rather than parametric memory."
    },
    {
        "question" : "What is the difference between intrinsic and extrinsic hallucinations?",
        "reference": "Intrinsic hallucinations contradict the source context while extrinsic hallucinations add information that cannot be verified from the source."
    },
    {
        "question" : "When should you use RAG instead of fine-tuning?",
        "reference": "RAG is preferred when knowledge needs to be frequently updated, when source attribution is required, or when computational resources for fine-tuning are limited."
    },
    {
        "question" : "What are the advantages of fine-tuning over RAG?",
        "reference": "Fine-tuning internalizes knowledge into model weights enabling faster inference without retrieval overhead and can improve performance on specific domains."
    },
    {
        "question" : "What is chain of thought prompting?",
        "reference": "Chain of thought prompting encourages language models to generate intermediate reasoning steps before producing a final answer, improving performance on complex reasoning tasks."
    },
    {
        "question" : "How does chain of thought improve LLM reasoning?",
        "reference": "Chain of thought improves reasoning by breaking complex problems into sequential steps, allowing the model to solve multi-step problems requiring arithmetic or symbolic reasoning."
    },
    {
        "question" : "How does the attention mechanism work in transformers?",
        "reference": "The attention mechanism computes weighted sums of value vectors, where weights are determined by compatibility between query and key vectors using scaled dot-product attention."
    },
    {
        "question" : "What is the difference between self-attention and cross-attention?",
        "reference": "Self-attention computes attention within the same sequence while cross-attention computes attention between two sequences like encoder outputs and decoder inputs."
    },
    {
        "question" : "How are large language models evaluated?",
        "reference": "LLMs are evaluated using benchmarks measuring reasoning, knowledge, and language understanding, as well as human evaluation for helpfulness, harmlessness, and honesty."
    },
    {
        "question" : "What is the MMLU benchmark?",
        "reference": "MMLU (Massive Multitask Language Understanding) tests LLMs across 57 academic subjects including mathematics, history, law, and medicine to measure world knowledge."
    },
    {
        "question" : "What are sentence embeddings and how are they used in NLP?",
        "reference": "Sentence embeddings are dense vector representations capturing semantic meaning, used for semantic search, clustering, and retrieval in applications such as RAG systems."
    },
    {
        "question" : "What is the difference between sparse and dense representations in information retrieval?",
        "reference": "Sparse representations like BM25 use high-dimensional vectors based on word occurrences, while dense representations use low-dimensional vectors learned by neural networks to capture semantic meaning."
    },
]


# ==============================================================================
# APPEL API GROQ (fonction utilitaire)
# ==============================================================================

def call_groq(prompt: str, system: str = "") -> str:
    """
    Appel simple à l'API Groq et retourne le texte de la réponse.

    Fonction utilitaire réutilisée par les 3 métriques.
    On utilise httpx directement, sans SDK, pour éviter
    les problèmes de dépendances.

    Args:
        prompt : message utilisateur
        system : instruction système optionnelle

    Returns:
        Texte de la réponse du LLM
    """

    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        url     = GROQ_API_URL,
        json    = {"model": MODEL_NAME, "messages": messages, "temperature": 0},
        headers = headers,
        timeout = TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==============================================================================
# LES 3 MÉTRIQUES
# ==============================================================================

def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Mesure si la réponse est fidèle aux documents fournis.

    Principe :
        On décompose la réponse en affirmations individuelles.
        Pour chaque affirmation, on vérifie si elle est supportée
        par au moins un des chunks récupérés.
        Score = nb affirmations supportées / nb affirmations total

    Un score proche de 1 = le LLM ne fait que répéter ce qui est
    dans les documents (pas d'hallucination).
    Un score proche de 0 = le LLM invente ou contredit les documents.

    Args:
        answer   : réponse générée par notre RAG
        contexts : chunks récupérés par le retrieval

    Returns:
        Score entre 0.0 et 1.0
    """

    context_text = "\n\n".join(contexts[:3])   # on prend les 3 premiers chunks

    prompt = f"""Given the following context documents and an answer, evaluate faithfulness.

CONTEXT:
{context_text}

ANSWER:
{answer}

Task: List each factual claim in the answer (one per line, numbered).
Then for each claim, write YES if it is supported by the context, NO if it is not.

Format your response exactly like this:
CLAIMS:
1. [claim]
2. [claim]

SUPPORTED:
1. YES/NO
2. YES/NO

Be strict: only mark YES if the claim is explicitly supported by the context."""

    try:
        response = call_groq(prompt)
        time.sleep(DELAY_METRIC)

        # Extraction des YES/NO depuis la réponse
        lines = response.upper().split("\n")
        supported = [l for l in lines if "YES" in l or "NO" in l]

        if not supported:
            return 0.5   # pas de réponse exploitable

        yes_count = sum(1 for l in supported if "YES" in l)
        return round(yes_count / len(supported), 4)

    except Exception as e:
        print(f"    Erreur faithfulness : {e}")
        return 0.0


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Mesure si la réponse répond bien à la question posée.

    Principe :
        On demande au LLM d'évaluer sur une échelle de 0 à 10
        dans quelle mesure la réponse répond à la question.
        On normalise entre 0 et 1.

    Un score proche de 1 = la réponse est directement pertinente.
    Un score proche de 0 = la réponse passe à côté de la question.

    Args:
        question : question posée par l'utilisateur
        answer   : réponse générée par notre RAG

    Returns:
        Score entre 0.0 et 1.0
    """

    prompt = f"""Evaluate how well the following answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate the relevancy of the answer on a scale from 0 to 10, where:
- 10 = The answer directly and completely addresses the question
- 5  = The answer partially addresses the question
- 0  = The answer is completely off-topic

Respond with ONLY a single integer from 0 to 10. Nothing else."""

    try:
        response = call_groq(prompt).strip()
        time.sleep(DELAY_METRIC)

        # Extraction du score numérique
        score = int(''.join(filter(str.isdigit, response.split()[0])))
        score = max(0, min(10, score))   # borné entre 0 et 10
        return round(score / 10, 4)

    except Exception as e:
        print(f"    Erreur answer_relevancy : {e}")
        return 0.0


def compute_context_recall(reference: str, contexts: list[str]) -> float:
    """
    Mesure si le retrieval a trouvé les chunks qui couvrent la réponse attendue.

    Principe :
        On décompose la réponse de référence en éléments d'information.
        Pour chaque élément, on vérifie s'il est couvert par les chunks.
        Score = nb éléments couverts / nb éléments total

    Un score proche de 1 = le retrieval a bien trouvé les bons chunks.
    Un score proche de 0 = des informations importantes manquent dans les chunks.

    Args:
        reference : réponse de référence (ce qu'on attendait idéalement)
        contexts  : chunks récupérés par le retrieval

    Returns:
        Score entre 0.0 et 1.0
    """

    context_text = "\n\n".join(contexts[:3])

    prompt = f"""Given the following context documents and a reference answer, evaluate context recall.

CONTEXT:
{context_text}

REFERENCE ANSWER:
{reference}

Task: List each key piece of information from the reference answer (one per line, numbered).
Then for each piece, write YES if it is covered by the context, NO if it is not.

Format your response exactly like this:
KEY INFORMATION:
1. [info]
2. [info]

COVERED:
1. YES/NO
2. YES/NO"""

    try:
        response = call_groq(prompt)
        time.sleep(DELAY_METRIC)

        lines = response.upper().split("\n")
        covered = [l for l in lines if "YES" in l or "NO" in l]

        if not covered:
            return 0.5

        yes_count = sum(1 for l in covered if "YES" in l)
        return round(yes_count / len(covered), 4)

    except Exception as e:
        print(f"    Erreur context_recall : {e}")
        return 0.0


# ==============================================================================
# COLLECTE DES OUTPUTS RAG
# ==============================================================================

def collect_rag_outputs(rag: RAGPipeline, dataset: list[dict]) -> list[dict]:
    """
    Fait tourner le RAG sur toutes les questions du dataset.

    Pour chaque question on récupère :
        - La réponse générée
        - Les chunks récupérés (contextes)
        - La réponse de référence

    Args:
        rag     : pipeline RAG initialisé
        dataset : liste de dicts {question, reference}

    Returns:
        Liste de dicts avec question, answer, contexts, reference
    """

    results = []
    total   = len(dataset)

    for i, item in enumerate(dataset):
        question  = item["question"]
        reference = item["reference"]

        print(f"  [{i+1}/{total}] {question[:55]}...")

        try:
            response = rag.ask(question, top_k=5, verbose=False)
            contexts = [chunk.text for chunk in response.sources]

            results.append({
                "question" : question,
                "answer"   : response.answer,
                "contexts" : contexts,
                "reference": reference,
            })
            print(f"  ✅ OK")

        except Exception as e:
            print(f"  ❌ Erreur : {e}")
            results.append({
                "question" : question,
                "answer"   : "",
                "contexts" : [],
                "reference": reference,
            })

        # Délai rate limit Groq
        if i < total - 1:
            print(f"  ⏱  Attente {DELAY_COLLECT}s...")
            time.sleep(DELAY_COLLECT)

    return results


# ==============================================================================
# CALCUL DES MÉTRIQUES
# ==============================================================================

def compute_all_metrics(rag_outputs: list[dict]) -> dict:
    """
    Calcule les 3 métriques sur tous les outputs RAG.

    Pour chaque question valide (avec réponse et contextes),
    on calcule les 3 scores. On retourne les moyennes.

    Args:
        rag_outputs : outputs collectés par collect_rag_outputs()

    Returns:
        Dictionnaire {métrique: score_moyen}
    """

    faith_scores   = []
    relev_scores   = []
    recall_scores  = []

    valid = [o for o in rag_outputs if o["answer"] and o["contexts"]]
    total = len(valid)

    print(f"\n{total} questions valides à évaluer")
    print("(3 appels LLM par question × 3 métriques)\n")

    for i, output in enumerate(valid):

        q   = output["question"]
        a   = output["answer"]
        ctx = output["contexts"]
        ref = output["reference"]

        print(f"  [{i+1}/{total}] {q[:50]}...")

        # Calcul des 3 métriques
        faith  = compute_faithfulness(a, ctx)
        relev  = compute_answer_relevancy(q, a)
        recall = compute_context_recall(ref, ctx)

        faith_scores.append(faith)
        relev_scores.append(relev)
        recall_scores.append(recall)

        print(f"    Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Recall: {recall:.2f}")

    # Calcul des moyennes
    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "faithfulness"    : mean(faith_scores),
        "answer_relevancy": mean(relev_scores),
        "context_recall"  : mean(recall_scores),
        "n_questions"     : total,
    }


# ==============================================================================
# AFFICHAGE DES RÉSULTATS
# ==============================================================================

def display_results(scores: dict) -> None:
    """
    Affiche les résultats finaux de façon lisible.

    Args:
        scores : dictionnaire des scores moyens
    """

    n = scores.get("n_questions", 0)

    print("\n" + "="*55)
    print("  RÉSULTATS DE L'ÉVALUATION")
    print("="*55)
    print(f"  Questions évaluées : {n}")
    print()

    metrics = {
        "faithfulness"    : "Fidélité aux documents (anti-hallucination)",
        "answer_relevancy": "Pertinence des réponses aux questions",
        "context_recall"  : "Qualité du retrieval (bons chunks trouvés)",
    }

    for key, label in metrics.items():
        score = scores.get(key, 0)

        if score >= 0.8:
            symbol = "🟢 Excellent"
        elif score >= 0.6:
            symbol = "🟡 Bon"
        elif score >= 0.4:
            symbol = "🟠 Moyen"
        else:
            symbol = "🔴 À améliorer"

        print(f"  {label}")
        print(f"    {score:.4f}  {symbol}")
        print()

    print("="*55)
    print()
    print("  Ce que tu peux dire en entretien :")
    print(f'  "Mon RAG obtient {scores["faithfulness"]:.2f} de faithfulness,')
    print(f'   {scores["answer_relevancy"]:.2f} de answer relevancy et')
    print(f'   {scores["context_recall"]:.2f} de context recall')
    print(f'   sur un eval set de {n} questions construites manuellement."')
    print("="*55)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":

    print("="*55)
    print("  Évaluation du pipeline ArXiv RAG")
    print("="*55)
    print()

    # Étape 1 : pipeline RAG
    print("Étape 1/3 : Initialisation du pipeline RAG...")
    rag = RAGPipeline()

    # Étape 2 : collecte des outputs
    print(f"\nÉtape 2/3 : Génération sur {len(EVAL_DATASET)} questions...")
    print(f"(Délai de {DELAY_COLLECT}s entre chaque question)\n")
    rag_outputs = collect_rag_outputs(rag, EVAL_DATASET)

    # Sauvegarde des outputs bruts
    out_file = Path("evaluation_outputs.json")
    out_file.write_text(
        json.dumps(rag_outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nOutputs sauvegardés dans {out_file}")

    # Étape 3 : calcul des métriques
    print("\nÉtape 3/3 : Calcul des métriques...")
    scores = compute_all_metrics(rag_outputs)

    # Affichage
    display_results(scores)

    # Sauvegarde des scores
    scores_file = Path("evaluation_scores.json")
    scores_file.write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8",
    )
    print(f"\nScores sauvegardés dans {scores_file}")