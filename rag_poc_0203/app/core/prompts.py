from typing import Dict, Any, Optional
from langfuse import Langfuse
from app.core.config import get_settings

settings = get_settings()

class LocalPrompt:
    """Wrapper for local fallback prompts to match Langfuse Prompt interface roughly."""
    def __init__(self, template: str, version: int = 0):
        self.template = template
        self.version = version
        
    def compile(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_key = e.args[0]
            return f"Error: Failed to render prompt. Missing variable: {{{missing_key}}}. check your Langfuse prompt variables."
        except Exception as e:
            return f"Error: Failed to render prompt. {str(e)}"

class PromptManager:
    def __init__(self):
        # Initialize Langfuse Client
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        
        # Fallback Defaults (Korean Optimized)
        self._defaults = {
            "system_default": "당신은 유능하고 친절한 AI 어시스턴트입니다.",
            "rag_context": """제공된 Context를 사용하여 사용자의 질문에 답변하세요.
만약 Context에 정보가 부족하다면 "주어진 정보로는 알 수 없습니다"라고 정중히 답하세요.

Context:
{retrieved_context}""",
            "task_rag_qa": """[지시사항]
다음 사용자 질문에 답변하세요.

사용자 질문:
{user_query}

작성 지침:
- 위 Context를 주 근거로 사용하세요.
- 필요하다면 소제목이나 불렛포인트를 사용하여 답변을 구조화하세요.
- 만약 가정이 포함된다면, 이를 명시하세요.""",
            "agent_planner": """[역할]
당신은 전략 기획자입니다. 사용자의 요청이 복잡하다면 2~3단계의 논리적인 단계로 나누세요.
간단한 요청이라면 단순히 그대로 반환하세요.

사용자 요청: {user_query}

출력 형식:
1. 1단계...
2. 2단계...""",
            "agent_critic": """[역할]
당신은 비평가입니다. 답변이 Context에 의해 완전히 뒷받침되는지 확인하세요.

Context:
{context}

답변:
{answer}

평가 기준:
1. 환각(Hallucination)이 없을 것.
2. Context에 기반한 사실일 것.

출력 형식:
[SCORE]: 0.0 ~ 1.0 (1.0은 완벽함)
[FEEDBACK]: 짧은 비평."""
        }

    def get_prompt(self, name: str, version: Optional[int] = None):
        """
        Fetches a prompt from Langfuse. 
        Falls back to local default if Langfuse fails or prompt is missing.
        """
        try:
            # Try fetching from Langfuse (defaults to label='production')
            return self.langfuse.get_prompt(name, version=version)
        except Exception as e:
            # print(f"Failed to fetch '{name}': {e}")
            
            # 1. Fallback to local default if key matches
            if name in self._defaults:
                return LocalPrompt(self._defaults[name])
            
            # 2. If prompt is custom (e.g. 'test') and fails, 
            # implies it exists in DB but maybe no 'production' label.
            # Return a safe error prompt to avoid 500.
            # We must escape braces to prevent .format() from crashing on JSON strings in str(e)
            start_msg = f"ERROR: Prompt '{name}' not found OR missing 'production' label used in Langfuse."
            detail_msg = f"Error: {str(e)}"
            safe_msg = f"{start_msg}\n{detail_msg}".replace("{", "{{").replace("}", "}}")
            
            return LocalPrompt(safe_msg)

prompt_manager = PromptManager()
