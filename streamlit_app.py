"""
Streamlit UI for a Customizable Conversational Agentic RAG System
"""
import streamlit as st
from pathlib import Path
import sys
import time
import uuid
from pypdf import PdfReader
from langchain_core.documents import Document
import os

# Configuration to reduce file watching issues
st.set_page_config(
    page_title="🤖 AgenticRAGChat - Chat RAG Personnalisé",
    page_icon="✨",
    layout="wide"
)

# Disable file watcher if running in production/problematic environment
if "DISABLE_STREAMLIT_WATCHER" in os.environ:
    st._config.set_option("server.fileWatcherType", "none")

# Ajoute le répertoire src au chemin pour que les imports fonctionnent
sys.path.append(str(Path(__file__).resolve().parent))

# --- Imports de votre Logique RAG Réelle ---
try:
    from langchain_community.document_loaders import UnstructuredURLLoader
    from src.config.config import Config
    from src.document_ingestion.document_processor import DocumentProcessor
    from src.vectorstore.vectorstore import VectorStore
    from src.graph_builder.graph_builder import GraphBuilder
except ImportError as e:
    st.error(f"Erreur d'importation critique : {e}")
    st.error("Veuillez vous assurer que votre projet a la bonne structure de dossiers et que toutes les dépendances sont installées.")
    st.stop()

# --- CSS Personnalisé pour un Design Moderne ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #6366F1;
    --secondary-color: #F8FAFC;
    --text-color: #1E293B;
    --text-light: #64748B;
    --bg-color: #F1F5F9;
    --border-color: #E2E8F0;
}

body {
    font-family: 'Inter', sans-serif;
    color: var(--text-color);
}

#MainMenu, footer {
    display: none;
}

.main .block-container {
    padding: 1rem 2rem;
}

[data-testid="stSidebar"] {
    background-color: var(--secondary-color);
    border-right: 1px solid var(--border-color);
}

[data-testid="stSidebar"] h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
    padding-top: 1rem;
}

[data-testid="stSidebar"] .stButton > button {
    background-image: linear-gradient(to right, #6366F1, #8B5CF6);
    color: white;
    border: none;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 10px rgba(0,0,0,0.1);
}

.stChatMessage {
    background: none;
    border: none;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border-radius: 0.75rem;
    padding: 1rem;
}

.stChatMessage[data-testid="chat-message-user"] {
    background-color: var(--primary-color);
    color: white;
}

.stChatMessage[data-testid="chat-message-assistant"] {
    background-color: white;
}

[data-testid="stChatInput"] {
    background-color: white;
    border-top: 1px solid var(--border-color);
    padding: 1rem 0;
}

.welcome-container {
    text-align: center;
    padding: 4rem 2rem;
}

.welcome-container h1 {
    font-size: 3rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #6366F1, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome-container p {
    font-size: 1.1rem;
    color: var(--text-light);
    max-width: 600px;
    margin: 1rem auto;
}
"""

# --- Initialisation de l'État de Session ---
def init_session_state():
    """Initialise les variables nécessaires pour la session utilisateur."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"cogniflow-{str(uuid.uuid4())[:8]}"

# --- Fonctions Logiques ---
@st.cache_resource
def get_base_components():
    """Charge les composants de base qui ne dépendent pas des documents."""
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    return llm, doc_processor

def build_rag_system(documents):
    """Construit le système RAG à partir d'une liste de documents."""
    progress_bar = st.progress(0, text="Construction du système RAG...")
    try:
        llm, _ = get_base_components()
        progress_bar.progress(25, text="Création du Vector Store...")
        vector_store = VectorStore()
        vector_store.create_vectorstore(documents)
        
        progress_bar.progress(75, text="Construction du graphe conversationnel...")
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        st.session_state.rag_system = graph_builder
        st.session_state.initialized = True
        progress_bar.progress(100, text="Système prêt !")
        time.sleep(1)
        progress_bar.empty()
        st.success(f"✅ Système RAG initialisé avec **{len(documents)}** fragments de documents.")
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Erreur lors de la construction du système : {e}")
        st.exception(e)

# --- Fonction de test du retriever ---
def test_retriever(query):
    """Teste le retriever avec une requête donnée."""
    if not st.session_state.initialized or not st.session_state.rag_system:
        st.warning("⚠️ Le système RAG n'est pas encore initialisé.")
        return None
    
    try:
        # Accéder au retriever via le graph_builder
        retriever = st.session_state.rag_system.retriever
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception as e:
        st.error(f"Erreur lors de la récupération des documents : {e}")
        return None

# --- Interface Utilisateur ---
def main():
    """Fonction principale de l'application Streamlit."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    init_session_state()

    # Display system info and potential fixes
    if st.sidebar.button("🔧 Diagnostics système"):
        import subprocess
        
        with st.sidebar.expander("Informations système", expanded=True):
            try:
                max_watches = subprocess.run(['cat', '/proc/sys/fs/inotify/max_user_watches'], 
                                           capture_output=True, text=True).stdout.strip()
                max_instances = subprocess.run(['cat', '/proc/sys/fs/inotify/max_user_instances'], 
                                             capture_output=True, text=True).stdout.strip()
                
                st.write(f"Max watches: {max_watches}")
                st.write(f"Max instances: {max_instances}")
                
                if int(max_instances) < 1024:
                    st.warning("⚠️ inotify instances limit trop bas. Exécutez:\n```bash\necho 8192 | sudo tee /proc/sys/fs/inotify/max_user_instances\n```")
                    
            except Exception as e:
                st.error(f"Impossible de récupérer les informations système: {e}")

    # --- Barre latérale ---
    with st.sidebar:
        st.markdown("<h1>✨ AgenticRAGChat</h1>", unsafe_allow_html=True)
        st.caption("Votre assistant RAG agentique personnalisable")
        st.divider()

        st.header("1. 📤 Importez vos sources")
        urls_text = st.text_area("Entrez les URLs (une par ligne)", height=100, placeholder="https://example.com/page1\nhttps://example.com/page2")
        raw_text = st.text_area("Collez du texte brut ici", height=100, placeholder="Collez un article, un rapport, etc.")
        uploaded_files = st.file_uploader("Ou téléversez des fichiers PDF", type="pdf", accept_multiple_files=True)

        st.header("2. ⚙️ Construisez le système")
        if st.button("Lancer la construction", use_container_width=True, type="primary"):
            # Réinitialise l'état avant une nouvelle construction
            st.session_state.initialized = False
            st.session_state.rag_system = None
            st.session_state.messages = []

            _, doc_processor = get_base_components()
            all_documents = []

            with st.spinner("Analyse et traitement des sources..."):
                if urls_text:
                    urls = [url.strip() for url in urls_text.split('\n') if url.strip() and url.startswith('http')]
                    if urls:
                        try:
                            loader = UnstructuredURLLoader(urls=urls, ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"})
                            loaded_docs = loader.load()
                            url_docs = doc_processor.splitter.split_documents(loaded_docs)
                            all_documents.extend(url_docs)
                        except Exception as e:
                            st.error(f"Impossible de charger les URLs : {e}")
                if raw_text:
                    text_docs = doc_processor.splitter.split_documents([Document(page_content=raw_text, metadata={"source": "Texte brut"})])
                    all_documents.extend(text_docs)
                if uploaded_files:
                    for file in uploaded_files:
                        try:
                            pdf_reader = PdfReader(file)
                            pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)
                            pdf_docs = doc_processor.splitter.split_documents([Document(page_content=pdf_text, metadata={"source": file.name})])
                            all_documents.extend(pdf_docs)
                        except Exception as e:
                            st.error(f"Erreur lors de la lecture du fichier PDF {file.name}: {e}")
            
            if all_documents:
                build_rag_system(all_documents)
            else:
                st.warning("⚠️ Veuillez fournir au moins une source de document valide.")

        # --- Section de test du retriever ---
        if st.session_state.initialized:
            st.header("3. 🔍 Tester le retriever")
            test_query = st.text_input("Entrez une requête de test", placeholder="donner un papier")
            if st.button("Tester", use_container_width=True):
                if test_query:
                    with st.spinner("Test du retriever..."):
                        docs = test_retriever(test_query)
                        if docs:
                            st.success(f"✅ {len(docs)} documents trouvés")
                            with st.expander("Voir les résultats", expanded=True):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"""
                                    **Document {i+1}:**
                                    - Source: {doc.metadata.get('source', 'N/A')}
                                    - Score: {doc.metadata.get('score', 'N/A')}
                                    - Contenu: {doc.page_content[:200]}...
                                    """)
                        else:
                            st.warning("Aucun document trouvé ou erreur.")
                else:
                    st.warning("Veuillez saisir une requête.")

        st.divider()
        st.caption(f"ID de session : `{st.session_state.thread_id}`")

    # --- Zone de Chat principale ---
    if st.session_state.initialized and st.session_state.rag_system:
        # Afficher l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Gérer la nouvelle question de l'utilisateur
        if question := st.chat_input("Posez votre question ici..."):
            # Vérifier si c'est une commande de test du retriever
            if question.startswith("/test-retriever "):
                query = question.replace("/test-retriever ", "").strip()
                st.session_state.messages.append({"role": "user", "content": f"Test retriever: {query}"})
                
                with st.chat_message("user"):
                    st.markdown(f"Test retriever: {query}")
                
                with st.chat_message("assistant"):
                    docs = test_retriever(query)
                    if docs:
                        response = f"✅ {len(docs)} documents trouvés:\n\n"
                        for i, doc in enumerate(docs):
                            response += f"**Document {i+1}:**\n"
                            response += f"- Source: {doc.metadata.get('source', 'N/A')}\n"
                            response += f"- Contenu: {doc.page_content[:200]}...\n\n"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "❌ Aucun document trouvé ou erreur."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Traitement normal de la question
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    with st.spinner("Recherche et génération de la réponse..."):
                        try:
                            start_time = time.time()
                            result = st.session_state.rag_system.run(question, st.session_state.thread_id)
                            elapsed_time = time.time() - start_time
                            
                            answer = result.get("answer", "Désolé, une erreur est survenue lors de la génération de la réponse.")
                            full_response += answer
                            message_placeholder.markdown(full_response)

                            retrieved_docs = result.get("retrieved_docs", [])
                            if retrieved_docs:
                                with st.expander("📄 Voir les sources consultées"):
                                    for doc in retrieved_docs:
                                        st.markdown(f"""
                                        <div style="border-left: 3px solid var(--primary-color); padding: 0.5rem 1rem; margin-bottom: 0.5rem; background-color: #F9FAFB;">
                                            <strong>Source: {doc.metadata.get('source', 'N/A')}</strong>
                                            <p style="font-size: 0.9rem; color: var(--text-light);">{doc.page_content[:300]}...</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                            st.caption(f"⏱️ Temps de réponse : {elapsed_time:.2f}s")

                        except Exception as e:
                            full_response = "Désolé, une erreur technique est survenue. Veuillez vérifier les logs."
                            message_placeholder.error(full_response)
                            st.exception(e)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Écran d'accueil
        st.markdown("""
            <div class="welcome-container">
                <h1>✨ Bienvenue sur AgenticRAGChat</h1>
                <p>Votre plateforme pour créer des assistants conversationnels intelligents basés sur vos propres documents. Importez vos sources via la barre latérale pour commencer.</p>
                <h3>Comment ça marche ?</h3>
                <p>
                    <strong>1. Importez :</strong> Fournissez des URLs, du texte ou des PDFs.<br>
                    <strong>2. Construisez :</strong> Le système analyse et indexe vos documents.<br>
                    <strong>3. Chattez :</strong> Posez des questions et obtenez des réponses sourcées.<br>
                    <strong>4. Explorez :</strong> Si une information manque, l'agent recherche intelligemment sur le web (Tavily, Wikipedia, Arxiv) pour vous fournir une réponse complète.
                </p>
                <h4>💡 Astuces :</h4>
                <p>
                    • Une fois le système construit, utilisez la section "Tester le retriever" dans la barre latérale<br>
                    • Ou tapez <code>/test-retriever votre_requête</code> dans le chat pour tester directement le retriever
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()