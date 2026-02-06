import os
import asyncio
from typing import List, Optional
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from app.core.config import get_settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

settings = get_settings()

class GraphRetriever:
    def __init__(self, working_dir: str = "./data/lightrag"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            
        # Initialize LightRAG with LOCAL Ollama model
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_name=settings.LOCAL_MODEL_NAME,  # exaone3.5:7.8b
            llm_model_func=ollama_model_complete,
            embedding_func=ollama_embed,
            llm_model_kwargs={"host": settings.LOCAL_MODEL_URL},  # http://localhost:11434
        )
        
        # Track initialization state
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized before use."""
        if not self._initialized:
            await self.rag.initialize_storages()
            self._initialized = True

    async def ingest(self, text: str):
        """Indexes text into the knowledge graph using LightRAG's native async method."""
        await self._ensure_initialized()
        # Use LightRAG's native async insert method
        await self.rag.ainsert(text)

    async def query(self, query: str, mode: str = "hybrid") -> str:
        """
        Queries the knowledge graph using LightRAG's native query method.
        Modes: 'local', 'global', 'hybrid', 'naive'
        """
        await self._ensure_initialized()
        param = QueryParam(mode=mode)
        # LightRAG's query is synchronous, but works in async context
        result = await self.rag.aquery(query, param=param)
        return result

# Singleton
graph_retriever = GraphRetriever()
