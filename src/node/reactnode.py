"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List
from src.state.rag_state import RAGState # Assurez-vous que RAGState est un TypedDict ou Pydantic Model

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Imports for tools and schemas
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_tavily import TavilySearch 
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Définition Robuste des Schémas d'Outils ---
class ToolInput(BaseModel):
    """Schéma d'entrée générique pour les outils de recherche."""
    query: str = Field(description="la requête de recherche")

# --- Classe des Nœuds du Graph ---
class RAGNodes:
    """Contient les fonctions des nœuds et la logique des outils pour le workflow RAG"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    # --- NŒUDS DU GRAPH ---
    def retrieve_docs(self, state: RAGState) -> dict:
        """Nœud pour récupérer les documents."""
        print("---🔎 NŒUD: RÉCUPÉRATION DES DOCUMENTS---")
        # BUG FIX: LangGraph passe l'état comme un dictionnaire. Utilisez state['key'] pour y accéder.
        question = state.question
        try:
            docs = self.retriever.invoke(question)
            return {"retrieved_docs": docs}
        except Exception as e:
            print(f"Erreur dans retrieve_docs : {e}")
            return {"retrieved_docs": []}

    def generate_answer(self, state: RAGState) -> dict:
        """Nœud qui exécute l'agent ReAct pour générer une réponse."""
        print("---🤖 NŒUD: EXÉCUTION DE L'AGENT AVEC OUTILS---")
        if self._agent is None:
            self._build_agent()
        
        # BUG FIX: LangGraph passe l'état comme un dictionnaire. Utilisez state['key'] pour y accéder.
        question = state.question
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=question)]})
            answer = result["messages"][-1].content
            return {"answer": answer or "Impossible de générer une réponse."}
        except Exception as e:
            print(f"Erreur dans generate_answer : {e}")
            return {"answer": f"Erreur de l'agent : {str(e)}"}

    # --- LOGIQUE DES OUTILS ---
    def _execute_retriever(self, query: str) -> str:
        """Exécute la recherche dans les documents indexés."""
        try:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "Aucun document trouvé pour cette requête."
            return "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        except Exception as e:
            return f"Erreur lors de la récupération des documents : {str(e)}"

    # --- CONSTRUCTION DE L'AGENT ---
    def _build_tools(self) -> List[Tool]:
        """Construit la liste des outils en utilisant les méthodes et classes appropriées."""
        
        retriever_tool = Tool(
            name="retriever",
            description="Recherche des informations dans les documents fournis par l'utilisateur. À utiliser en priorité pour les questions spécifiques au contexte.",
            func=self._execute_retriever,
            args_schema=ToolInput
        )
        
        # CORRECTION: Laissez les outils pré-construits utiliser leurs schémas par défaut pour une meilleure fiabilité.
        # Le `args_schema` n'est pas nécessaire pour eux.
        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="fr")
        )

        tavily_tool = TavilySearch(max_results=5)

        arxiv_tool = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        )
        
        return [retriever_tool, wikipedia_tool, tavily_tool, arxiv_tool]

    def _build_agent(self):
        """Construit l'agent ReAct avec les outils définis."""
        tools = self._build_tools()
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful and expert research assistant. "
                    "You must use the most appropriate tool to answer the user's question. "
                    "Prioritize the 'retriever' tool if the question seems related to the user's documents. "
                    "Return only the final, concise answer in the user's language."
                ),
            ),
            MessagesPlaceholder(variable_name="messages")]
    )
        # Corrected line
        self._agent = create_react_agent(self.llm, tools, prompt=prompt)