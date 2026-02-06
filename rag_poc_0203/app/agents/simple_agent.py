import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Dict, Any, Optional
import os

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

try:
    from langfuse.decorators import observe
except ImportError:
    observe = lambda *args, **kwargs: (lambda f: f)
from app.models.router import router
from app.rag.retriever import retriever
from app.core.prompts import prompt_manager
from app.rag.query_logic import generate_queries
from app.rag.web_tools import web_search_tool
from langchain_core.runnables import RunnableConfig

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    queries: List[str] # Variations for Multi-Query
    context: str
    is_relevant: bool 
    retrieved_docs: List[Dict]
    prompt_map: Dict[str, str]
    collection_name: str
    retrieval_config: Dict
    summary: str # Compressed history

# 2. Nodes & Helpers
async def llm_rerank(query: str, docs: List[Dict], top_k: int, config: Optional[RunnableConfig] = None) -> List[Dict]:
    """
    Reranks documents using a cheap LLM call for better precision.
    """
    doc_list = ""
    for idx, d in enumerate(docs):
        doc_list += f"[{idx}] {d['content'][:]}\n"
        
    prompt = f"""
다음 문서를 검색 쿼리에 가장 관련 있는 순서대로 정렬해주세요. 
관련 있는 문서의 인덱스 번호만 쉼표로 구분하여 상위 {top_k}개만 나열하세요. (예: 2, 0, 1)

쿼리: {query}

문서 목록:
{doc_list}

순위 (인덱스만):"""

    try:
        # Use simple task type for speed
        gen_result = await router.generate(prompt, task_type="simple", config=config)
        response = gen_result.content
        
        # Parse indices
        import re
        indices = [int(i) for i in re.findall(r"\d+", response)]
        
        reranked = []
        for idx in indices:
            if 0 <= idx < len(docs):
                reranked.append(docs[idx])
        
        if reranked:
            return reranked[:top_k]
    except Exception as e:
        print(f"Reranking failed: {e}. Falling back to original order.")
        
    return docs[:top_k]

@observe()
async def rewrite_query_node(state: AgentState, config: RunnableConfig):
    """
    Expands the user query into multiple variations for better retrieval.
    """
    user_query = state["messages"][-1].content
    summary = state.get("summary", "")
    
    # If summary exists, combine it for better query expansion context
    expansion_input = f"[맥락: {summary}] {user_query}" if summary else user_query
    
    queries = await generate_queries(expansion_input, n=2, config=config) # Generate 2 extra variations
    return {"queries": queries}

@observe()
async def retrieve_node(state: AgentState, config: RunnableConfig):
    original_query = state["messages"][-1].content
    queries = state.get("queries", [original_query])
    prompt_map = state.get("prompt_map", {})
    collection_name = state.get("collection_name", "knowledge_base")
    retrieval_config = state.get("retrieval_config", {})
    
    top_k = retrieval_config.get("top_k", 3)
    use_reranker = retrieval_config.get("use_reranker", False)
    score_threshold = retrieval_config.get("score_threshold", 0.0)
    search_type = retrieval_config.get("search_type", "vector")
    metadata_filter = retrieval_config.get("metadata_filter", None)
    
    fetch_k = top_k * 2 if use_reranker else top_k 
    
    try:
        import asyncio
        # Perform parallel retrieval for all queries
        tasks = [
            retriever.retrieve(
                q, 
                collection_name=collection_name, 
                limit=fetch_k,
                score_threshold=score_threshold,
                search_type=search_type,
                metadata_filter=metadata_filter
            ) for q in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten and De-duplicate
        all_docs = []
        seen_contents = set()
        for doc_list in results:
            for d in doc_list:
                if d['content'] not in seen_contents:
                    all_docs.append(d)
                    seen_contents.add(d['content'])
        
        # 2. Rerank if needed
        if (use_reranker or len(all_docs) > top_k) and all_docs:
            all_docs.sort(key=lambda x: x['score'], reverse=True)
            docs = await llm_rerank(original_query, all_docs, top_k, config=config)
        else:
            docs = all_docs[:top_k]
        
        # Format context
        prompt_name = prompt_map.get("rag_context", "rag_context")
        context_template = prompt_manager.get_prompt(prompt_name)
        
        formatted_context_str = "\n".join([
            f"- {d['content']} (Source: {d['source']}, Score: {d['score']:.2f})" 
            for d in docs
        ])
        
        context_template_str = getattr(context_template, "prompt", getattr(context_template, "template", ""))
        final_context = context_template_str.replace("{retrieved_context}", formatted_context_str)
        
        return {"context": final_context, "retrieved_docs": docs}
    except Exception as e:
        print(f"ERROR [retrieve_node]: {e}")
        return {"context": "Error: 정보를 검색하는 도중 기술적인 문제가 발생했습니다.", "retrieved_docs": []}

@observe()
async def grade_documents_node(state: AgentState, config: RunnableConfig):
    """
    Grades whether the retrieved documents are relevant to the user query.
    This is the 'Self-Correction' part of Self-RAG.
    """
    user_query = state["messages"][-1].content
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        print("DEBUG [Grader]: No documents retrieved. Result: NOT RELEVANT")
        return {"is_relevant": False}
    
    context_content = "\n".join([f"- {d['content']}" for d in docs])
    
    prompt = f"""
당신은 검색 결과의 부합성을 판단하는 공정한 평가자입니다.
아래 질문과 검색 문서를 비교하여, 검색 문서가 질문에 대한 핵심적인 실마리나 답변을 포함하고 있는지 판단하세요.
질문과 전혀 관계없는 내용이거나, 질문에 대한 답변이 전혀 포함되지 않는다면 반드시 'no'라고 답변하세요.

사용자 질문: {user_query}

검색된 문서 내용:
{context_content}

결과 (검색 문서가 질문에 답변하는 데 충분한 정보를 포함하고 있습니까? yes/no):"""

    try:
        model_instance = router.get_model(task_type="simple")
        gen_result = await model_instance.ainvoke(prompt, config=config)
        score = gen_result.content.lower().strip()
        # Look for 'yes' or 'no' strictly
        if "yes" in score and "no" not in score:
            is_relevant = True
        elif "no" in score:
            is_relevant = False
        else:
            # Fallback for ambiguous cases
            is_relevant = "yes" in score
            
        print(f"DEBUG [Grader]: Query='{user_query}', Score='{score}', Result={is_relevant}")
        return {"is_relevant": is_relevant}
    except Exception as e:
        print(f"DEBUG [Grader]: Error: {e}. Defaulting to True.")
        return {"is_relevant": True}

@observe()
async def web_search_node(state: AgentState, config: RunnableConfig):
    """
    Performs a web search when the internal retrieval is not relevant. (CRAG)
    """
    user_query = state["messages"][-1].content
    summary = state.get("summary", "")
    
    # Combine query with summary for better search context
    search_query = f"{summary} {user_query}" if summary else user_query
    
    print(f"INFO [CRAG]: Internal docs insufficient. Performing web search for: '{search_query}'")
    
    web_results = await web_search_tool.search(search_query)
    
    formatted_context = "\n".join([
        f"- {d['content']} (Source: {d['source']})" 
        for d in web_results
    ])
    
    # Prefix the context to let the user know it's from the web
    final_context = f"[Web Search Results]\n{formatted_context}"
    
    return {
        "context": final_context,
        "retrieved_docs": web_results # Store for inspector
    }

@observe()
async def generate_node(state: AgentState, config: RunnableConfig):
    user_query = state["messages"][-1].content
    context = state["context"]
    prompt_map = state.get("prompt_map", {})
    
    # Load Prompts
    sys_name = prompt_map.get("system_default", "system_default")
    task_name = prompt_map.get("task_rag_qa", "task_rag_qa")
    
    system_prompt = prompt_manager.get_prompt(sys_name)
    task_prompt = prompt_manager.get_prompt(task_name)
    
    # 1. Interpolate Task Prompt
    task_template_str = getattr(task_prompt, "prompt", getattr(task_prompt, "template", ""))
    qa_prompt_part = task_template_str.replace("{user_query}", user_query)
    
    # 2. Combine with Context
    final_user_content = f"{context}\n\n[지시사항]\n{qa_prompt_part}"
    
    print(f"DEBUG PROMPT:\n{final_user_content}")
    
    # Detect if context is from Web Search (CRAG)
    is_web_result = "[Web Search Results]" in context
    
    # Append Summary to System Prompt if exists
    summary = state.get("summary", "")
    system_content = system_prompt.compile()
    if summary:
        system_content += f"\n\n[이전 대화 요약]\n{summary}"
    
    # If this is a web result, add attribution instruction
    if is_web_result:
        system_content += "\n\nCRITICAL: 내부 지식 베이스에서 정보를 찾을 수 없어 웹 검색을 수행했습니다. 답변 시작 시 반드시 '내부 문서에서 관련 내용을 찾을 수 없어 웹 검색 결과로 답변드립니다'라는 취지의 안내를 포함하세요."

    try:
        # 1. Get Model Instance
        model_instance = router.get_model(task_type="simple")
        
        # 2. Invoke with LangChain compatible messsages
        from langchain_core.messages import SystemMessage, HumanMessage
        lc_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=final_user_content)
        ]
        
        # ainvoke will still trigger 'on_chat_model_stream' if the model is configured for streaming 
        # and we are using graph.astream_events
        response = await model_instance.ainvoke(lc_messages, config=config)
        
        response_text = response.content
        usage_metadata = getattr(response, "usage_metadata", {})
        res_metadata = getattr(response, "response_metadata", {})
        model_name = model_instance.model_name
        
    except Exception as e:
        print(f"ERROR [generate_node]: {e}")
        response_text = "죄송합니다. 답변을 생성하는 과정에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        usage_metadata = {}
        res_metadata = {"error": str(e)}
        model_name = "error-fallback"

    try:
        from langfuse.decorators import langfuse_context
        # Tag the trace with the specific prompts used
        langfuse_context.update_current_trace(
            tags=[sys_name, task_name],
            metadata={
                "system_prompt": sys_name,
                "task_prompt": task_name,
                "context_prompt": prompt_map.get("rag_context", "rag_context")
            }
        )
    except Exception:
        pass
    
    from app.ops.monitor import get_current_trace_id
    trace_id = get_current_trace_id()
    
    # Create rich AIMessage with metadata
    ai_message = AIMessage(
        content=response_text,
        usage_metadata=usage_metadata,
        response_metadata={
            "model": model_name,
            **res_metadata
        }
    )

    return {
        "messages": [ai_message],
        "metadata": {"trace_id": trace_id}
    }

@observe()
async def summarize_history_node(state: AgentState, config: RunnableConfig):
    """
    Summarizes the existing conversation history when it gets too long.
    """
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    
    if existing_summary:
        summary_prompt = f"""
이전 요약: {existing_summary}

기존 요약에 다음 대화 내용을 포함하여 다시 요약해주세요. 
대화의 핵심 맥락과 중요한 정보(특히 사용자의 선호도나 특정 지식 베이스 관련 내용)를 유지해야 합니다.

새로운 대화:
{messages[:-1]}
"""
    else:
        summary_prompt = f"""
다음은 사용자와 AI의 대화 기록입니다. 
이후 대화의 연속성을 위해 핵심 맥락과 중요한 정보를 요약해주세요. 
불필요한 인사나 사소한 내용은 제외하고 지식 베이스와 관련된 중요한 사실 위주로 작성하세요.

대화 내용:
{messages[:-1]}
"""
    
    try:
        model = router.get_model(task_type="simple")
        response = await model.ainvoke(summary_prompt)
        new_summary = response.content
    except Exception as e:
        print(f"ERROR [summarize_history_node]: {e}")
        new_summary = existing_summary
    
    # Prune messages: Keep only the new summary and the last 2-3 messages for immediate context
    # In LangGraph, we can't easily 'delete' messages from the shared state by returning a partial dict 
    # if the messages key is Annotated with operator.add.
    # We might need a more custom approach or just work with the summary in the prompt.
    # For now, we store the summary and continue.
    return {"summary": new_summary}

def should_summarize(state: AgentState):
    """
    Conditional logic to decide if we should run summarization.
    """
    if len(state["messages"]) > 10:
        return "summarize"
    return "end"

@observe()
async def missing_info_node(state: AgentState, config: RunnableConfig):
    """
    Called when the grader determines that the retrieved documents are not relevant.
    """
    response_text = "죄송합니다. 요청하신 질문에 대해 지식 베이스에서 관련 있는 정보를 찾지 못했습니다. 질문을 구체화해주시거나 다른 주제로 질문해주시겠어요?"
    
    ai_message = AIMessage(
        content=response_text,
        response_metadata={"model": "self-rag-fallback", "reason": "no_relevant_docs"}
    )
    
    return {
        "messages": [ai_message]
    }

# 3. Build Graph
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("missing_info", missing_info_node)
    workflow.add_node("summarize", summarize_history_node)
    
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    
    # Conditional routing based on relevance (CRAG)
    workflow.add_conditional_edges(
        "grade_docs",
        lambda x: "generate" if x["is_relevant"] else "web_search",
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    
    # After generation or missing_info, check if we need to summarize
    workflow.add_conditional_edges(
        "generate",
        should_summarize,
        {
            "summarize": "summarize",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "missing_info",
        should_summarize,
        {
            "summarize": "summarize",
            "end": END
        }
    )
    
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

agent_graph = build_agent_graph()
