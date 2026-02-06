from app.models.router import ModelRouter
from app.core.config import get_settings

settings = get_settings()
router = ModelRouter()

async def classify_intent(message: str) -> str:
    """
    Classifies user intent into 'simple' (fast RAG) or 'complex' (reasoning agent).
    """
    # 1. Rule-based heuristic (fast)
    keywords_complex = ["분석", "비교", "계획", "정리", "검증", "비판", "데이터셋"]
    if any(keyword in message for keyword in keywords_complex):
        return "complex"
    
    # 2. Simple LLM classification (fallback)
    system_prompt = "You are an intent classifier. Categorize the user message into 'simple' (simple Q&A) or 'complex' (needs planning, analysis, or multi-step logic). Reply with ONLY the word 'simple' or 'complex'."
    
    try:
        result = await router.generate(
            prompt=message,
            task_type="simple",
            system=system_prompt
        )
        prediction = result.content.lower().strip()
        if "complex" in prediction:
            return "complex"
        return "simple"
    except Exception:
        return "simple" # Default to simple on error
