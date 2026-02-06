import os
from openai import OpenAI, AsyncOpenAI
from app.core.config import get_settings
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda *args, **kwargs: (lambda f: f)
    langfuse_context = None

settings = get_settings()

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class GenerationResult:
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class ModelRouter:
    def __init__(self):
        self._clients = {}

    def get_model(self, task_type: str = "simple") -> ChatOpenAI:
        """
        Returns a LangChain ChatOpenAI instance configured for the appropriate model.
        """
        if task_type == "complex" and settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-..."):
            # Cloud Model
            return ChatOpenAI(
                model=settings.DEFAULT_MODEL_NAME,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
                streaming=True
            )
        else:
            # Local Model (Ollama)
            return ChatOpenAI(
                model=settings.LOCAL_MODEL_NAME,
                openai_api_key="sk-dummy",
                base_url=f"{settings.LOCAL_MODEL_URL}/v1",
                temperature=0.7,
                streaming=True
            )

    @observe(as_type="generation")
    async def generate(self, prompt: str, task_type: str = "simple", system: str = None, config: Optional[Dict[str, Any]] = None) -> GenerationResult:
        model_instance = self.get_model(task_type)
        
        # Capture generation metadata
        if langfuse_context:
            langfuse_context.update_current_observation(
                name="llm_generation",
                input=prompt,
                model=model_instance.model_name
            )

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            # Pass config (callbacks, etc.) to ainvoke
            response = await model_instance.ainvoke(messages, config=config)
            
            # Extract usage if available
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input": response.usage_metadata.get("input_tokens", 0),
                    "output": response.usage_metadata.get("output_tokens", 0),
                    "total": response.usage_metadata.get("total_tokens", 0),
                    "unit": "TOKENS"
                }

            # Update generation output and usage
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=response.content,
                    usage=usage
                )

            return GenerationResult(
                content=response.content,
                model=model_instance.model_name,
                usage=usage,
                metadata=getattr(response, "response_metadata", {})
            )
        except Exception as e:
            print(f"Model call failed: {e}")
            raise e

router = ModelRouter()
