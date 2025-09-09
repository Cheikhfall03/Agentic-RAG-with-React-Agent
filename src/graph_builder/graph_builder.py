"""Graph builder for an intelligent RAG workflow using an LLM for decision logic"""

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver

# Assurez-vous que RAGState est un TypedDict et que les nœuds retournent des dictionnaires
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes
from src.config.config import Config



class GraphBuilder:
    """Builds and manages the intelligent LangGraph workflow using an LLM for decision-making"""
    
    def __init__(self, retriever, llm):
        """
        Initialise le constructeur du graph.
        
        Args:
            retriever: L'instance du retriever de documents.
            llm: L'instance du modèle de langage.
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
        self.llm = llm
    
    def _decide_next_step(self, state: RAGState) -> str:
        """
        Fonction de décision utilisant le LLM pour déterminer si les documents récupérés sont suffisants.
        """
        print("---DECISION: Évaluation de la pertinence des documents avec le LLM---")
        docs = state.retrieved_docs
        question = state.question
        
        # Si aucun document n'est trouvé, il faut obligatoirement utiliser l'agent
        if not docs:
            print("---DECISION: Aucun document trouvé. Passage à l'agent de recherche.---")
            return "agent_search"
        
        # Formate le contenu des documents pour le LLM
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Crée un prompt pour demander au LLM d'évaluer la pertinence
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "Vous êtes un arbitre expert. Votre rôle est de déterminer si les documents fournis contiennent suffisamment d'informations pour répondre de manière complète à la question de l'utilisateur. "
             "Répondez uniquement par 'oui' si les documents sont suffisants, et 'non' s'ils ne le sont pas ou si des informations externes sont nécessaires."),
            ("user", 
             "Question: {question}\n\n"
             "Documents:\n{context}")
        ])
        
        # Crée une chaîne et appelle le LLM
        chain = prompt_template | self.llm
        response = chain.invoke({"question": question, "context": context})
        
        decision = response.content.strip().lower()
        print(f"---DECISION: Le LLM a répondu '{decision}'.---")
        
        if "oui" in decision:
            print("---DECISION: Documents jugés suffisants. Passage à la génération directe.---")
            return "generate_direct"
        else:
            print("---DECISION: Documents jugés insuffisants. Passage à l'agent de recherche.---")
            return "agent_search"

    def _generate_direct_answer(self, state: RAGState) -> dict:
        """
        Génère une réponse directement à partir des documents récupérés.
        """
        print("---📝 NŒUD: Génération directe avec documents---")
        docs = state.retrieved_docs
        question = state.question
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant expert en RAG. Répondez à la question en vous basant uniquement sur le contexte suivant:\n\n{context}"),
            ("user", "Question: {question}")
        ])
        
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context, "question": question})
        
        # Les nœuds doivent retourner un dictionnaire des champs mis à jour
        return {"answer": response.content}
    
    def build(self):
        """
        Construit le graph RAG intelligent avec la logique de décision du LLM.
        """
        builder = StateGraph(RAGState)
        
        # Ajout des nœuds
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("generate_direct", self._generate_direct_answer)
        builder.add_node("agent_search", self.nodes.generate_answer)
        
        # Définition du point d'entrée
        builder.set_entry_point("retriever")
        
        # Ajout des arêtes conditionnelles basées sur la décision du LLM
        builder.add_conditional_edges(
            "retriever",
            self._decide_next_step,
            {
                "generate_direct": "generate_direct",
                "agent_search": "agent_search"
            }
        )
        
        # Ajout des arêtes finales
        builder.add_edge("generate_direct", END)
        builder.add_edge("agent_search", END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)
        
        # Sauvegarde de l'image du graph
        try:
            image_data = self.graph.get_graph().draw_mermaid_png()
            with open("intelligent_graph.png", "wb") as f:
                f.write(image_data)
            print("Image du graph sauvegardée dans 'intelligent_graph.png'")
        except Exception as e:
            print(f"Impossible de sauvegarder la visualisation du graph : {e}")
            
        return self.graph
    
    def run(self, question: str, thread_id: str) -> dict:
        """
        Exécute le workflow RAG intelligent pour un fil de discussion spécifique.
        """
        if self.graph is None:
            self.build()
        
        initial_state = {"question": question}
        config = Config.get_thread_config(thread_id)
        
        return self.graph.invoke(initial_state, config=config)