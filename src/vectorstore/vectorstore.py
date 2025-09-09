"""Vector store module for document embedding and retrieval with advanced features"""
from typing import List
from langchain.schema import Document

# Pour la recherche sémantique (dense retriever)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Pour la recherche par mots-clés (sparse retriever)
from langchain_community.retrievers import BM25Retriever

# Pour combiner les deux types de recherche (hybrid search)
from langchain.retrievers import EnsembleRetriever

# Pour le reranking et la compression contextuelle avec un modèle Hugging Face
# NOUVEAU : Nous importons HuggingFaceCrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import CrossEncoderReranker


class VectorStore:
    """Manages vector store operations with hybrid search and a Hugging Face reranker."""

    def __init__(self):
        """
        Initialize vector store with Hugging Face embeddings.
        Assurez-vous que votre clé API GROQ_API_KEY
        est définie comme variable d'environnement.
        """
        self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1
                }
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create a hybrid search retriever with a reranker from documents.
        Args:
            documents: List of documents to embed.
        """
        # 1. Initialiser le retriever sémantique (dense) basé sur FAISS
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10}) # On augmente k pour donner plus de choix au reranker

        # 2. Initialiser le retriever par mots-clés (sparse) basé sur BM25
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10 # On augmente k également ici

        # 3. Combiner les deux en un retriever hybride (EnsembleRetriever)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        # 4. FIXED: Use CrossEncoderReranker with sentence-transformers model
        # Create a CrossEncoder model from sentence-transformers
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

        # Create the reranker using CrossEncoderReranker
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=3)

        # 5. Use the reranker as base_compressor
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_retriever
        )

    def get_retriever(self):
        """
        Get the final retriever instance.
        Returns:
            The contextual compression retriever instance.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant and reranked documents for a query.
        Args:
            query: Search query.
        Returns:
            A list of relevant documents, reranked for quality.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        return self.retriever.invoke(query)