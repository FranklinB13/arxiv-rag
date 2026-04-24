"""
Script : parse_papers.py
Description :
    Lance le parsing de tous les PDFs dans data/raw/
    et sauvegarde les textes nettoyés dans data/processed/

    C'est la 2ème étape du pipeline RAG, après le téléchargement.

Utilisation :
    uv run python scripts/parse_papers.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path

# On importe notre module de parsing depuis src/arxiv_rag/
# C'est pour ça qu'on a créé la structure src/ : le code est importable
from arxiv_rag.parsing import parse_all_papers


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dossiers d'entrée et de sortie
INPUT_DIR  = Path("data/raw")        # PDFs téléchargés
OUTPUT_DIR = Path("data/processed")  # Textes extraits


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Étape 2 : Parsing des PDFs")
    print("=" * 50)

    # Lancement du parsing en batch
    stats = parse_all_papers(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Affichage du résumé
    print(f"\nRésultats :")
    print(f"  ✅ Parsés avec succès : {stats['success']}")
    print(f"  ⏭️  Déjà présents      : {stats['skipped']}")
    print(f"  ❌ Échecs              : {stats['failed']}")
    print(f"\n📁 Textes sauvegardés dans : {OUTPUT_DIR.absolute()}")