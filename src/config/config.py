"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = "MyLangChainApp"
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANCHAIN_API_KEY")
    os.environ["USER_AGENT"] = USER_AGENT

    # Model Configuration
    LLM_MODEL = "llama-3.1-8b-instant"
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return init_chat_model(cls.LLM_MODEL, model_provider="groq")

    # NEW: Method to generate the LangGraph config for a specific thread
    @classmethod
    def get_thread_config(cls, thread_id: str) -> dict:
        """
        Generates the configuration dictionary for a specific conversation thread.
        This is used to ensure memory (checkpoints) is saved for the correct session.
        """
        return {"configurable": {"thread_id": thread_id}}