from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Service"
    
    # Langfuse
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", "difyai123456")
    LANGFUSE_HOST: str = "http://localhost:3000"
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    
    # OpenAI
    OPENAI_API_KEY: str | None = None
    
    # Model Config
    DEFAULT_MODEL_NAME: str = "gpt-4o"
    LOCAL_MODEL_NAME: str = "exaone3.5:7.8b"
    LOCAL_MODEL_URL: str = "http://localhost:11434"

    # Embedding Config
    EMBEDDING_BINDING: str = "ollama"
    EMBEDDING_MODEL: str = "bge-m3:latest"
    EMBEDDING_BINDING_HOST: str = "http://localhost:11434"

    # Internal Service URLs (for UI and inter-service comms)
    BASE_URL: str = "http://127.0.0.1:8000"
    @property
    def CHAT_API_URL(self) -> str: return f"{self.BASE_URL}/chat"
    @property
    def EVAL_API_URL(self) -> str: return f"{self.BASE_URL}/eval/run"
    @property
    def COLLECTIONS_API_URL(self) -> str: return f"{self.BASE_URL}/rag/collections"
    @property
    def INGEST_API_URL(self) -> str: return f"{self.BASE_URL}/rag/ingest"
    @property
    def FEEDBACK_API_URL(self) -> str: return f"{self.BASE_URL}/chat/feedback"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
