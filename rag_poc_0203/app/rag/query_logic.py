from typing import List, Optional, Any, Dict
from app.models.router import router
from langchain_core.runnables import RunnableConfig

async def generate_queries(query: str, n: int = 3, config: Optional[RunnableConfig] = None) -> List[str]:
    """
    Generates multiple variations of the user query to improve retrieval recall.
    """
    prompt = f"""
다음 질문에 대해 검색 엔진에서 더 좋은 결과를 얻을 수 있도록 {n}개의 다양한 검색 쿼리를 생성해주세요. 
각 쿼리는 한 줄에 하나씩 작성하고, 불필요한 서술이나 숫자는 제외하세요.

원래 질문: {query}

생성된 쿼리 목록:"""

    try:
        model = router.get_model(task_type="simple")
        gen_result = await model.ainvoke(prompt, config=config)
        queries = [q.strip() for q in gen_result.content.split("\n") if q.strip()]
        # Always include the original query
        if query not in queries:
            queries.insert(0, query)
        return queries[:n+1]
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return [query]

async def decompose_query(query: str, config: Optional[RunnableConfig] = None) -> List[str]:
    """
    Decomposes a complex query into simpler sub-queries.
    """
    prompt = f"""
다음은 복잡한 질문입니다. 이 질문에 답하기 위해 필요한 세부 질문들(sub-questions)로 나누어주세요.
각 질문은 한 줄에 하나씩 작성하세요.

질문: {query}

세부 질문 목록:"""

    try:
        model = router.get_model(task_type="simple")
        gen_result = await model.ainvoke(prompt, config=config)
        sub_queries = [q.strip() for q in gen_result.content.split("\n") if q.strip()]
        return sub_queries
    except Exception as e:
        print(f"Query decomposition failed: {e}")
        return [query]
