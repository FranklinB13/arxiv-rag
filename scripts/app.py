"""
Script : app.py
Auteur : Franklin
Date   : 2026-04

Description :
    Interface web du systeme ArXiv RAG, construite avec Gradio 6.13.

    Gradio est la bibliotheque standard pour creer des interfaces
    de demonstration ML. Elle est utilisee par Hugging Face Spaces
    et reconnue par tous les recruteurs du domaine.

    L'interface propose :
        - Un champ de saisie pour la question
        - Un affichage de la reponse avec citations [N]
        - Un panneau de sources avec paper_id, page, score
        - Un historique des questions/reponses de la session

    Notes Gradio 6.13 :
        - gr.Chatbot attend des dicts {"role": ..., "content": ...}
        - Le theme ne se passe plus dans Blocks() ni launch()
        - type="messages" n'existe pas encore dans cette version

Utilisation :
    uv run python scripts/app.py
    Puis ouvre http://127.0.0.1:7860 dans ton navigateur
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import gradio as gr    # Interface web ML standard
import time            # Pour le rate limiting Groq

# Notre pipeline RAG complet
from arxiv_rag.rag import RAGPipeline


# ==============================================================================
# INITIALISATION DU PIPELINE
# ==============================================================================

# On initialise le pipeline UNE SEULE FOIS au demarrage de l'app
# Charger les modeles prend ~10 secondes.
# Si on le faisait a chaque requete, l'interface serait inutilisable.
print("Demarrage du pipeline RAG...")
rag = RAGPipeline()
print("Pipeline pret ! Lancement de l'interface...")

# Timestamp de la derniere requete pour le rate limiting Groq
# Tier gratuit Groq = 6000 tokens/minute
# On impose un delai minimum entre les requetes
last_request_time = 0.0
MIN_DELAY_SECONDS = 15


# ==============================================================================
# FONCTION PRINCIPALE : traitement d'une question
# ==============================================================================

def answer_question(
    question : str,
    history  : list,
) -> tuple[str, str, list]:
    """
    Traite une question et retourne la reponse avec ses sources.

    Cette fonction est appelee par Gradio a chaque soumission.
    Elle reçoit la question et l'historique, et retourne :
        - Le champ question vide (efface apres soumission)
        - Le HTML des sources
        - L'historique mis a jour

    Format historique Gradio 6.13 :
        Liste de dicts {"role": "user"/"assistant", "content": "..."}
        C'est le format standard OpenAI adopte par Gradio 6+.

    Gestion du rate limiting :
        On attend si necessaire pour eviter l'erreur 429 de Groq.

    Args:
        question : texte de la question saisie par l'utilisateur
        history  : liste des messages precedents

    Returns:
        Tuple (champ_vide, sources_html, historique_mis_a_jour)
    """

    global last_request_time

    # Validation : on ignore les questions vides
    if not question.strip():
        return "", "Veuillez saisir une question.", history

    # Rate limiting : attendre si on a envoye une requete recemment
    elapsed = time.time() - last_request_time
    if elapsed < MIN_DELAY_SECONDS:
        wait_time = MIN_DELAY_SECONDS - elapsed
        time.sleep(wait_time)

    try:
        # Appel du pipeline RAG complet
        # ask() fait : retrieval hybride + reranking + generation Groq
        response = rag.ask(question, top_k=5, verbose=False)

        # Mise a jour du timestamp pour le rate limiting
        last_request_time = time.time()

        # Formatage des sources en HTML
        sources_html = format_sources_html(response.sources)

        # Mise a jour de l'historique
        # Format Gradio 6+ : dicts avec role et content
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": response.answer})

        # On retourne "" pour vider le champ de saisie
        return "", sources_html, history

    except Exception as e:
        # En cas d'erreur, on l'affiche dans le chat
        error_msg = f"Erreur : {str(e)}"
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return "", "", history


# ==============================================================================
# FORMATAGE DES SOURCES
# ==============================================================================

def format_sources_html(sources) -> str:
    """
    Formate les sources en HTML pour l'affichage dans Gradio.

    Cree un tableau HTML avec pour chaque source :
        - Le numero [N] correspondant aux citations dans la reponse
        - Le paper_id avec lien cliquable vers arxiv.org
        - Le numero de page dans le PDF original
        - Le score de pertinence avec code couleur

    Args:
        sources : liste de SearchResult du pipeline RAG

    Returns:
        Chaine HTML representant le tableau des sources
    """

    if not sources:
        return "<p style='color:gray;'>Aucune source trouvee.</p>"

    # En-tete du tableau
    html = """
    <table style='width:100%; border-collapse:collapse; font-size:14px;'>
        <thead>
            <tr style='background:#f0f0f0;'>
                <th style='padding:8px; border:1px solid #ddd;
                           text-align:center; width:40px;'>Ref</th>
                <th style='padding:8px; border:1px solid #ddd;
                           text-align:left;'>Paper</th>
                <th style='padding:8px; border:1px solid #ddd;
                           text-align:center; width:60px;'>Page</th>
                <th style='padding:8px; border:1px solid #ddd;
                           text-align:center; width:70px;'>Score</th>
            </tr>
        </thead>
        <tbody>
    """

    for source in sources:

        # Construction du lien arXiv
        # "2401_12345v1" → "2401.12345v1"
        # On remplace uniquement le premier underscore par un point
        arxiv_id  = source.paper_id.replace("_", ".", 1)
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

        # Affichage de la page (N/A si inconnue)
        page_display = str(source.page_number) if source.page_number else "N/A"

        # Code couleur selon le score de pertinence
        # Seuils bases sur les observations de nos tests
        if source.score > 3:
            score_color = "#28a745"    # vert : tres pertinent
        elif source.score > 1:
            score_color = "#fd7e14"    # orange : pertinent
        else:
            score_color = "#dc3545"    # rouge : peu pertinent

        html += f"""
            <tr>
                <td style='padding:8px; border:1px solid #ddd;
                           font-weight:bold; text-align:center;'>
                    [{source.rank}]
                </td>
                <td style='padding:8px; border:1px solid #ddd;'>
                    <a href='{arxiv_url}' target='_blank'
                       style='color:#0066cc; text-decoration:none;
                              font-family:monospace; font-size:12px;'>
                        {source.paper_id}
                    </a>
                </td>
                <td style='padding:8px; border:1px solid #ddd;
                           text-align:center;'>
                    {page_display}
                </td>
                <td style='padding:8px; border:1px solid #ddd;
                           color:{score_color}; font-weight:bold;
                           text-align:center;'>
                    {source.score:.2f}
                </td>
            </tr>
        """

    html += "</tbody></table>"
    return html


# ==============================================================================
# CONSTRUCTION DE L'INTERFACE GRADIO
# ==============================================================================

def build_interface() -> gr.Blocks:
    """
    Construit et retourne l'interface Gradio.

    On utilise gr.Blocks() pour un controle total sur la mise en page.
    C'est l'API moderne de Gradio vs gr.Interface() qui est plus simple
    mais moins flexible pour des layouts complexes.

    Returns:
        Interface Gradio configuree, prete a etre lancee
    """

    # gr.Blocks() sans arguments pour Gradio 6.13
    # Le theme n'est plus supporte dans Blocks() ni launch()
    with gr.Blocks(title="ArXiv RAG - NLP/ML Research Assistant") as interface:

        # --- En-tete ---
        gr.Markdown("""
        # 📚 ArXiv RAG — NLP/ML Research Assistant
        Ask questions about recent NLP and Machine Learning research papers.
        The system retrieves relevant excerpts from **313 arXiv papers** and
        generates answers with citations.

        > 💡 **Tip:** Ask in English for best results.
        > ⏱️ 15 seconds between requests (free API tier limit).
        """)

        # --- Layout principal en deux colonnes ---
        with gr.Row():

            # Colonne gauche : interface de chat (70% de la largeur)
            with gr.Column(scale=7):

                # Composant Chatbot
                # Gradio 6.13 : pas de parametre type=
                # mais accepte les dicts {"role": ..., "content": ...}
                chatbot = gr.Chatbot(
                    label  = "Conversation",
                    height = 500,
                )

                # Champ de saisie de la question
                question_input = gr.Textbox(
                    label       = "Your question",
                    placeholder = "e.g. What are the main techniques to reduce hallucinations in LLMs?",
                    lines       = 2,
                )

                # Boutons sur la meme ligne
                with gr.Row():
                    submit_btn = gr.Button(
                        value   = "🔍 Ask",
                        variant = "primary",
                    )
                    clear_btn = gr.Button(
                        value   = "🗑️ Clear",
                        variant = "secondary",
                    )

            # Colonne droite : sources (30% de la largeur)
            with gr.Column(scale=3):

                gr.Markdown("### 📄 Sources used")
                gr.Markdown(
                    "Papers retrieved for the last question. "
                    "Click a paper ID to open it on arXiv."
                )

                # Affichage HTML du tableau des sources
                sources_display = gr.HTML(
                    value = "<p style='color:gray;'>Sources will appear here after your first question.</p>"
                )

        # --- Exemples de questions cliquables ---
        gr.Examples(
            examples = [
                ["What are the main techniques to reduce hallucinations in LLMs?"],
                ["How does RLHF work for aligning language models?"],
                ["What is the difference between RAG and fine-tuning?"],
                ["What is chain of thought prompting?"],
                ["How does self-attention work in transformers?"],
            ],
            inputs = question_input,
            label  = "💡 Example questions (click to use)",
        )

        # --- Etat de l'historique cote serveur ---
        # gr.State persiste entre les requetes sans etre affiche
        history_state = gr.State([])

        # --- Connexion des evenements ---

        # Clic sur "Ask"
        submit_btn.click(
            fn      = answer_question,
            inputs  = [question_input, history_state],
            outputs = [question_input, sources_display, history_state],
        ).then(
            # Apres la reponse, on met a jour l'affichage du chatbot
            fn      = lambda h: h,
            inputs  = [history_state],
            outputs = [chatbot],
        )

        # Appui sur Entree dans le champ texte
        question_input.submit(
            fn      = answer_question,
            inputs  = [question_input, history_state],
            outputs = [question_input, sources_display, history_state],
        ).then(
            fn      = lambda h: h,
            inputs  = [history_state],
            outputs = [chatbot],
        )

        # Clic sur "Clear" : remet tout a zero
        clear_btn.click(
            fn      = lambda: (
                [],
                [],
                "<p style='color:gray;'>Sources will appear here after your first question.</p>",
            ),
            inputs  = [],
            outputs = [chatbot, history_state, sources_display],
        )

    return interface


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":

    # Construction de l'interface
    interface = build_interface()

    # Lancement du serveur Gradio local
    # share=False : local uniquement
    # server_port=7860 : port par defaut Gradio
    # Ouvre http://127.0.0.1:7860 dans ton navigateur
    interface.launch(
        share       = False,
        server_port = 7860,
        show_error  = True,
    )