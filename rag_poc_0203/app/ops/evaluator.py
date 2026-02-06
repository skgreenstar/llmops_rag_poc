from typing import List, Dict, Any, Optional
import asyncio
from langfuse import Langfuse
from app.core.config import get_settings

# Ragas & LangChain imports
from app.ops.monitor import observable, langfuse_context
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    answer_correctness,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity
)
from ragas.metrics._aspect_critic import conciseness, coherence, harmfulness, maliciousness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset
import traceback

# Reference: https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas

settings = get_settings()

class EvaluationResult:
    def __init__(self, score: float, reasoning: str, metric_name: str):
        self.score = score
        self.reasoning = reasoning
        self.metric_name = metric_name

class Evaluator:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        
        # Determine which LLM to use for evaluation
        # Fallback to local model if OpenAI key is invalid or placeholder
        api_key = settings.OPENAI_API_KEY
        self.is_placeholder = api_key is None or api_key == "" or api_key.startswith("sk-...")
        
        if not self.is_placeholder:
            print("🚀 Using OpenAI for Ragas evaluation")
            llm = ChatOpenAI(
                model=settings.DEFAULT_MODEL_NAME,
                openai_api_key=api_key,
                temperature=0
            )
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key
            )
        else:
            print(f"🏠 Using Local Model ({settings.LOCAL_MODEL_NAME}) and Embeddings ({settings.EMBEDDING_MODEL}) for Ragas evaluation")
            llm = ChatOllama(
                model=settings.LOCAL_MODEL_NAME,
                base_url=settings.LOCAL_MODEL_URL,
                temperature=0
            )
            embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.EMBEDDING_BINDING_HOST
            )
            
        # Initialize LLM & Embeddings for Ragas with wrappers as recommended
        self.eval_llm = LangchainLLMWrapper(llm)
        self.eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        # Define the set of metrics to run (Full Suite)
        self.metrics = [
            faithfulness, 
            answer_relevancy, 
            answer_correctness,
            context_precision,
            context_recall,
            context_entity_recall,
            answer_similarity,
            conciseness,
            coherence,
            harmfulness,
            maliciousness
        ]

    def submit_score(self, trace_id: str, result: EvaluationResult):
        """
        Submits a score to Langfuse attached to a specific trace.
        """
        try:
            # Avoid submitting NaN values to Langfuse as it causes Bad Request errors
            import math
            if math.isnan(result.score):
                print(f"⚠️ Skipping NaN score: {result.metric_name}")
                return

            self.langfuse.score(
                trace_id=trace_id,
                name=result.metric_name,
                value=result.score,
                comment=result.reasoning
            )
            self.langfuse.flush()
            print(f"✅ Score submitted: {result.metric_name} = {result.score:.2f}")
        except Exception as e:
            print(f"❌ Failed to submit score: {e}")

    @observable(name="ragas_eval", as_type="span")
    async def run_ragas_eval(self, query: str, context: str, answer: str, reference: Optional[str] = None) -> List[EvaluationResult]:
        """
        Runs multiple Ragas metrics in a single batch.
        """
        # If reference is not provided, use query as a neutral reference 
        # (Note: This might lower correctness scores if they expect a specific ground truth)
        if reference is None:
            reference = query
            
        data = {
            "question": [query],
            "contexts": [[context]],
            "answer": [answer],
            "reference": [reference]
        }
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        try:
            # Update trace metadata with judge info
            if langfuse_context:
                langfuse_context.update_current_trace(
                    metadata={
                        "judge_llm": settings.DEFAULT_MODEL_NAME if not self.is_placeholder else settings.LOCAL_MODEL_NAME,
                        "judge_provider": "openai" if not self.is_placeholder else "ollama",
                        "metrics_count": len(self.metrics)
                    }
                )

            # We use wait_for to avoid hanging if there are network issues
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings
            )
            
            # Ragas 0.2.x returns an EvaluationResult object. 
            # We convert the first row to a dict to get scalar scores.
            df = result.to_pandas()
            if df.empty:
                print("⚠️ Ragas evaluation returned an empty result.")
                return []
            
            scores = df.iloc[0].to_dict()
            
            output = []
            
            # Mapping of internal keys to Korean display names and detailed reasoning
            metric_info = {
                "faithfulness": {
                    "display_name": "충실도 (Faithfulness)",
                    "reasoning": "답변이 주어진 문맥에 얼마나 충실하게 근거하고 있는지를 평가합니다 (할루시네이션 방지)."
                },
                "answer_relevancy": {
                    "display_name": "답변 관련성 (Answer Relevancy)",
                    "reasoning": "답변이 사용자의 질문에 얼마나 직접적으로 관련되어 해결책을 제시하는지 평가합니다."
                },
                "answer_correctness": {
                    "display_name": "답변 정확도 (Answer Correctness)",
                    "reasoning": "생성된 답변이 기준 정답(Ground Truth)과 비교했을 때 사실적으로 얼마나 정확한지 평가합니다."
                },
                "context_precision": {
                    "display_name": "문맥 정밀도 (Context Precision)",
                    "reasoning": "검색된 문맥 정보 중 질문에 답하는 데 필요한 핵심 문서가 상위 순위에 잘 배치되었는지 평가합니다."
                },
                "context_recall": {
                    "display_name": "문맥 재현율 (Context Recall)",
                    "reasoning": "정답을 작성하는 데 필요한 실제 정보들이 검색된 문맥 내에 모두 포함되어 있는지 평가합니다."
                },
                "context_entity_recall": {
                    "display_name": "개체 재현율 (Context Entity Recall)",
                    "reasoning": "기준 정답에 포함된 핵심 개체(Entity)들이 검색된 문맥 내에 얼마나 잘 포함되어 있는지 평가합니다."
                },
                "answer_similarity": {
                    "display_name": "답변 유사도 (Answer Similarity)",
                    "reasoning": "생성된 답변과 기준 정답 간의 의미적 유사성을 벡터 공간에서 측정합니다."
                },
                "conciseness": {
                    "display_name": "간결성 (Conciseness)",
                    "reasoning": "답변이 불필요한 사족 없이 핵심적인 정보만 간결하게 전달하는지 평가합니다."
                },
                "coherence": {
                    "display_name": "일관성 (Coherence)",
                    "reasoning": "답변의 문장 흐름과 구조가 논리적으로 일관성이 있는지 평가합니다."
                },
                "harmfulness": {
                    "display_name": "유해성 (Harmfulness)",
                    "reasoning": "답변에 사용자에게 불쾌감을 주거나 유해한 내용이 포함되어 있는지 검증합니다."
                },
                "maliciousness": {
                    "display_name": "악의성 (Maliciousness)",
                    "reasoning": "답변에 기만적이거나 악의적인 의도가 포함되어 있는지 검증합니다."
                }
            }
            
            for key, info in metric_info.items():
                if key in scores:
                    value = scores[key]
                    # Handle numpy types or lists that might come back
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        value = value[0]
                    
                    output.append(EvaluationResult(
                        score=float(value),
                        reasoning=info["reasoning"],
                        metric_name=info["display_name"]
                    ))
            
            return output
        except Exception as e:
            print(f"⚠️ Ragas evaluation failed: {e}")
            traceback.print_exc()
            return []

    # Keeping legacy methods for backward compatibility but re-routing to Ragas
    async def evaluate_faithfulness(self, context: str, answer: str) -> EvaluationResult:
        results = await self.run_ragas_eval("General Query", context, answer)
        for r in results:
            if "Faithfulness" in r.metric_name:
                return r
        return EvaluationResult(0.0, "Ragas Failed", "Faithfulness")

    async def evaluate_relevance(self, query: str, answer: str) -> EvaluationResult:
        # We need context for answer_relevancy in Ragas usually, or it uses the question embedding
        results = await self.run_ragas_eval(query, "No Context Provided", answer)
        for r in results:
            if "Relevancy" in r.metric_name:
                return r
        return EvaluationResult(0.0, "Ragas Failed", "Answer Relevancy")

evaluator = Evaluator()
