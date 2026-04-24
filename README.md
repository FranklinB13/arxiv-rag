# ArXiv RAG — Assistant de recherche NLP/ML

Système RAG (Retrieval-Augmented Generation) qui répond à des questions sur la recherche en NLP et Machine Learning, en s'appuyant sur 313 papers arXiv.

Construit from scratch en Python — pas de LangChain, pas de raccourcis.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Le problème que ça résout

Quand tu lis de la littérature ML, retrouver "qui a proposé quoi, quand, et comment ça se compare" est pénible. Ce système permet de poser des questions en langage naturel sur un corpus de papers récents et d'obtenir des réponses sourcées avec citations précises (paper + page).

Questions typiques :
- *"Quelles sont les techniques principales pour réduire les hallucinations des LLMs ?"*
- *"Comment fonctionne RLHF pour aligner les modèles de langage ?"*
- *"Quelle est la différence entre RAG et fine-tuning ?"*

Exemple de réponse générée :

> D'après [1], le RAG réduit les hallucinations en ancrant les réponses dans un contexte récupéré. Cependant, [1] note également que le RAG reste sensible aux contextes bruités ou incomplets. Le fine-tuning [2] peut aussi aider mais nécessite des ressources computationnelles importantes...

---

## Architecture du pipeline

```
[313 PDFs arXiv]
      ↓
  Parsing          PyMuPDF    → extraction et nettoyage du texte
      ↓
  Chunking         fenêtre glissante, 400 mots, overlap de 80 mots
      ↓
  Embeddings       BAAI/bge-small-en-v1.5  → vecteurs de 384 dimensions
      ↓
  Vector DB        Qdrant (index HNSW)     → 10 432 vecteurs indexés
      ↓
      ↓  ← question de l'utilisateur
      ↓
  Dense retrieval  embeddings BGE  → top-20 par similarité sémantique
  Sparse retrieval BM25            → top-20 par correspondance de mots-clés
  Fusion           union + déduplication  → ~40 candidats
  Reranking        cross-encoder ms-marco → top-5 final
      ↓
  Génération       Groq LLaMA 3.1  → réponse avec citations [N]
      ↓
  [Réponse + sources traçables]
```

---

## Choix techniques

| Composant | Choix | Pourquoi |
|-----------|-------|----------|
| Embeddings | `BAAI/bge-small-en-v1.5` | Top du benchmark MTEB, open source, tourne en local |
| Vector DB | Qdrant (mode local) | Index HNSW, prêt pour la production, sans Docker |
| Retrieval | Hybride BM25 + dense | BM25 pour les correspondances exactes, dense pour le sens |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Évalue la paire (question, passage) ensemble — bien plus précis |
| LLM | Groq LLaMA 3.1 8B | API gratuite, inférence ultra-rapide (~500 tok/s) |
| UI | Gradio | Standard pour les démos ML, déployable sur HF Spaces |

### Pourquoi le retrieval hybride + reranking ?

Un RAG naïf utilise uniquement la recherche dense (similarité cosinus). Le problème : les embeddings capturent bien le sens général mais peuvent rater les correspondances exactes — noms de papers, acronymes, numéros de version. BM25 compense ça.

Le reranker évalue ensuite les candidats en regardant la paire (question, chunk) ensemble, ce qui est bien plus précis que la similarité cosinus seule. C'est ce qui sépare un RAG de tutoriel d'un RAG utilisable en production.

---

## Installation

**Prérequis :** Python 3.11+, Git

```bash
# 1. Cloner le repo
git clone https://github.com/FranklinB13/arxiv-rag.git
cd arxiv-rag

# 2. Installer les dépendances
pip install uv
uv sync

# 3. Configurer la clé API
# Créer un compte gratuit sur console.groq.com et générer une clé
echo "GROQ_API_KEY=ta_clé_ici" > .env

# 4. Construire le pipeline (à faire une seule fois)
uv run python scripts/download_papers.py   # télécharge les papers
uv run python scripts/parse_papers.py      # extrait le texte des PDFs
uv run python scripts/chunk_papers.py      # découpe en chunks
uv run python scripts/build_embeddings.py  # génère les embeddings (~15 min)
uv run python scripts/build_index.py       # construit l'index Qdrant

# 5. Lancer l'interface
uv run python scripts/app.py
# Ouvrir http://127.0.0.1:7860
```

---

## Structure du projet

```
arxiv-rag/
├── src/
│   └── arxiv_rag/
│       ├── parsing.py        # PDF → texte propre (PyMuPDF)
│       ├── chunking.py       # découpage avec fenêtre glissante et overlap
│       ├── embeddings.py     # génération des embeddings BGE
│       ├── vectorstore.py    # gestion de l'index Qdrant
│       ├── retrieval.py      # retrieval hybride BM25 + dense + reranking
│       ├── generation.py     # LLM Groq avec prompt de citation
│       └── rag.py            # orchestration du pipeline complet
├── scripts/
│   ├── download_papers.py    # ingestion arXiv (requêtes ciblées)
│   ├── parse_papers.py       # parsing PDF en batch
│   ├── chunk_papers.py       # chunking en batch
│   ├── build_embeddings.py   # génération embeddings en batch
│   ├── build_index.py        # construction de l'index Qdrant
│   ├── test_retrieval.py     # test qualité du retrieval
│   └── app.py                # interface web Gradio
├── data/
│   ├── raw/                  # PDFs téléchargés (gitignored)
│   ├── processed/            # textes extraits (gitignored)
│   ├── chunks/               # chunks JSON (gitignored)
│   └── embeddings/           # vecteurs numpy (gitignored)
├── pyproject.toml
└── README.md
```

---

## Corpus

**313 papers** arXiv, téléchargés sur 5 sujets ciblés :
- RAG et retrieval-augmented generation
- RLHF et alignement des LLMs
- Détection et réduction des hallucinations
- Comparaison fine-tuning vs RAG
- Chain-of-thought reasoning

**10 432 chunks** au total — moyenne de 33 chunks par paper, 400 mots chacun.

---

## Ce que j'ai appris en construisant ce projet

**La qualité du corpus est aussi importante que le pipeline.**
La première version utilisait 100 papers récents pris au hasard. Les scores de retrieval étaient autour de 1.5. En passant à 313 papers ciblés sur nos sujets, les scores sont montés à 5-6 sur les mêmes questions. Un bon corpus mal indexé reste utilisable. Un mauvais corpus bien indexé reste inutile.

**Le chunking impacte tout ce qui suit.**
Couper le texte tous les N caractères sans tenir compte des paragraphes casse les idées en deux. La fenêtre glissante avec overlap préserve le contexte aux jonctions et améliore significativement la précision du retrieval.

**Le reranking n'est pas optionnel pour un usage sérieux.**
Le retrieval dense seul donne des résultats plausibles. Le retrieval hybride + reranking donne des résultats corrects. La différence vient du cross-encoder qui voit la question et le passage ensemble — il évalue la pertinence de la paire, pas juste la proximité vectorielle.

---

## Roadmap

- [ ] Évaluation quantitative avec Ragas (faithfulness, answer relevancy)
- [ ] Streaming des réponses dans l'interface
- [ ] Déploiement sur Hugging Face Spaces
- [ ] Filtrage par métadonnées (année, topic, auteur)
- [ ] Extension du corpus à 1000+ papers

---

## Auteur

**Franklin** — Étudiant ingénieur Mathématiques Appliquées / IA — CY Tech  
[GitHub](https://github.com/FranklinB13)