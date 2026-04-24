"""
Script : compute_metrics.py
Description :
    Calcule les métriques sur les outputs déjà collectés.
    Utilise evaluation_outputs.json généré par evaluate.py.
    Délais augmentés pour respecter le rate limit Groq.

Utilisation :
    uv run python scripts/compute_metrics.py
"""

import json
import os
import time
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME     = "llama-3.1-8b-instant"
TIMEOUT        = 30

# Délai ENTRE chaque appel LLM individuel
# 3 métriques × 15 questions = 45 appels
# On attend 15s entre chaque appel = ~11 minutes total
# Largement suffisant pour le rate limit de 6000 tokens/minute
DELAY_BETWEEN_CALLS = 15


def call_groq(prompt: str) -> str:
    """Appel simple à l'API Groq avec retry automatique sur 429."""

    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
    }
    messages = [{"role": "user", "content": prompt}]

    # Retry jusqu'à 3 fois si rate limit
    for attempt in range(3):
        try:
            response = httpx.post(
                url     = GROQ_API_URL,
                json    = {"model": MODEL_NAME, "messages": messages, "temperature": 0},
                headers = headers,
                timeout = TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limit : on attend plus longtemps et on réessaie
                wait = 30 * (attempt + 1)
                print(f"    ⚠️  Rate limit, attente {wait}s (tentative {attempt+1}/3)...")
                time.sleep(wait)
            else:
                raise

    return ""   # si tous les retries échouent


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fidélité de la réponse aux documents fournis."""

    context_text = "\n\n".join(contexts[:3])
    prompt = f"""Given the context and answer below, list each factual claim in the answer (numbered), then write YES if it is supported by the context or NO if not.

CONTEXT:
{context_text}

ANSWER:
{answer}

Format:
CLAIMS:
1. [claim]

SUPPORTED:
1. YES/NO"""

    try:
        response = call_groq(prompt)
        lines    = response.upper().split("\n")
        verdicts = [l for l in lines if "YES" in l or "NO" in l]
        if not verdicts:
            return 0.5
        yes = sum(1 for l in verdicts if "YES" in l)
        return round(yes / len(verdicts), 4)
    except Exception as e:
        print(f"    Erreur faithfulness : {e}")
        return 0.0


def compute_answer_relevancy(question: str, answer: str) -> float:
    """Pertinence de la réponse par rapport à la question."""

    prompt = f"""Rate how well the answer addresses the question on a scale of 0 to 10.
10 = completely answers the question
5  = partially answers
0  = off-topic

QUESTION: {question}
ANSWER: {answer}

Reply with ONLY a single integer from 0 to 10."""

    try:
        response = call_groq(prompt).strip()
        score    = int(''.join(filter(str.isdigit, response.split()[0])))
        return round(max(0, min(10, score)) / 10, 4)
    except Exception as e:
        print(f"    Erreur relevancy : {e}")
        return 0.0


def compute_context_recall(reference: str, contexts: list[str]) -> float:
    """Qualité du retrieval : les chunks couvrent-ils la réponse attendue ?"""

    context_text = "\n\n".join(contexts[:3])
    prompt = f"""List each key piece of information from the reference answer (numbered), then write YES if it is covered by the context or NO if not.

CONTEXT:
{context_text}

REFERENCE ANSWER:
{reference}

Format:
KEY INFO:
1. [info]

COVERED:
1. YES/NO"""

    try:
        response = call_groq(prompt)
        lines    = response.upper().split("\n")
        verdicts = [l for l in lines if "YES" in l or "NO" in l]
        if not verdicts:
            return 0.5
        yes = sum(1 for l in verdicts if "YES" in l)
        return round(yes / len(verdicts), 4)
    except Exception as e:
        print(f"    Erreur recall : {e}")
        return 0.0


if __name__ == "__main__":

    # Chargement des outputs déjà collectés
    outputs_file = Path("evaluation_outputs.json")
    if not outputs_file.exists():
        print("evaluation_outputs.json introuvable. Lance d'abord evaluate.py")
        exit(1)

    rag_outputs = json.loads(outputs_file.read_text(encoding="utf-8"))
    valid       = [o for o in rag_outputs if o["answer"] and o["contexts"]]

    print("="*55)
    print("  Calcul des métriques RAG")
    print("="*55)
    print(f"  Questions valides : {len(valid)}")
    print(f"  Délai entre appels : {DELAY_BETWEEN_CALLS}s")
    print(f"  Durée estimée : ~{len(valid) * 3 * DELAY_BETWEEN_CALLS // 60} minutes")
    print()

    faith_scores  = []
    relev_scores  = []
    recall_scores = []

    for i, output in enumerate(valid):

        q   = output["question"]
        a   = output["answer"]
        ctx = output["contexts"]
        ref = output["reference"]

        print(f"[{i+1}/{len(valid)}] {q[:55]}...")

        # Faithfulness
        print("  → Faithfulness...")
        faith = compute_faithfulness(a, ctx)
        faith_scores.append(faith)
        time.sleep(DELAY_BETWEEN_CALLS)

        # Answer Relevancy
        print("  → Answer Relevancy...")
        relev = compute_answer_relevancy(q, a)
        relev_scores.append(relev)
        time.sleep(DELAY_BETWEEN_CALLS)

        # Context Recall
        print("  → Context Recall...")
        recall = compute_context_recall(ref, ctx)
        recall_scores.append(recall)

        print(f"  ✅ Faith: {faith:.2f} | Relev: {relev:.2f} | Recall: {recall:.2f}")

        # Délai après chaque question complète (sauf la dernière)
        if i < len(valid) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    # Calcul des moyennes
    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    scores = {
        "faithfulness"    : mean(faith_scores),
        "answer_relevancy": mean(relev_scores),
        "context_recall"  : mean(recall_scores),
        "n_questions"     : len(valid),
    }

    # Affichage final
    print("\n" + "="*55)
    print("  RÉSULTATS FINAUX")
    print("="*55)
    print(f"  Questions : {scores['n_questions']}")
    print()

    labels = {
        "faithfulness"    : "Fidélité aux documents",
        "answer_relevancy": "Pertinence des réponses",
        "context_recall"  : "Qualité du retrieval",
    }

    for key, label in labels.items():
        score = scores[key]
        if score >= 0.8:   emoji = "🟢 Excellent"
        elif score >= 0.6: emoji = "🟡 Bon"
        elif score >= 0.4: emoji = "🟠 Moyen"
        else:              emoji = "🔴 À améliorer"
        print(f"  {label} : {score:.4f}  {emoji}")

    print()
    print("  Ce que tu peux dire en entretien :")
    print(f'  "Mon RAG obtient {scores["faithfulness"]:.2f} de faithfulness,')
    print(f'   {scores["answer_relevancy"]:.2f} de answer relevancy et')
    print(f'   {scores["context_recall"]:.2f} de context recall')
    print(f'   sur {scores["n_questions"]} questions."')
    print("="*55)

    # Sauvegarde
    Path("evaluation_scores.json").write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8",
    )
    print("\nScores sauvegardés dans evaluation_scores.json")