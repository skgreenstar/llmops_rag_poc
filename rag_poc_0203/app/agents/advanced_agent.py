import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Dict, Any, Optional
try:
    from langfuse.decorators import observe
except ImportError:
    observe = lambda *args, **kwargs: (lambda f: f)
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app.models.router import router
from app.rag.retriever import retriever
from app.core.prompts import prompt_manager

# 1. Define State
class AdvancedAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: str
    context: str
    critique_score: float
    critique_feedback: str
    retry_count: int
    prompt_map: dict[str, str]
    collection_name: str # (NEW)
    retrieval_config: dict # (NEW)

# 2. Nodes
@observe()
async def planner_node(state: AdvancedAgentState, config: RunnableConfig):
    user_query = state["messages"][0].content
    prompt_map = state.get("prompt_map", {})
    
    prompt_name = prompt_map.get("agent_planner", "agent_planner")
    planner_prompt = prompt_manager.get_prompt(prompt_name)
    
    # Use direct string replacement for robustness
    planner_prompt_str = planner_prompt.compile() # Get raw template
    prompt = planner_prompt_str.replace("{user_query}", user_query)
    
    # Debug logging
    print(f"DEBUG [Planner]: prompt length={len(prompt)}")
    
    model = router.get_model(task_type="complex")
    gen_result = await model.ainvoke(prompt, config=config)
    return {"plan": gen_result.content, "retry_count": 0}

@observe()
async def executor_node(state: AdvancedAgentState, config: RunnableConfig):
    # ... (Retrieval Logic stays the same) ...
    user_query = state["messages"][0].content
    plan = state["plan"]
    prompt_map = state.get("prompt_map", {})
    collection_name = state.get("collection_name", "knowledge_base")
    retrieval_config = state.get("retrieval_config", {})
    
    top_k = retrieval_config.get("top_k", 3)
    use_reranker = retrieval_config.get("use_reranker", False)
    
    search_type = retrieval_config.get("search_type", "vector")
    metadata_filter = retrieval_config.get("metadata_filter", None)
    
    fetch_k = top_k * 3 if use_reranker else top_k
    docs = await retriever.retrieve(
        user_query, 
        collection_name=collection_name, 
        limit=fetch_k,
        search_type=search_type,
        metadata_filter=metadata_filter
    )
    
    if use_reranker and docs:
        from app.agents.simple_agent import llm_rerank
        docs = await llm_rerank(user_query, docs, top_k, config=config)
    else:
        docs = docs[:top_k]
    
    ctx_name = prompt_map.get("rag_context", "rag_context")
    context_template = prompt_manager.get_prompt(ctx_name)
    formatted_context = "\n".join([
        f"- {d['content']} (Source: {d['source']}, Score: {d['score']:.2f})" 
        for d in docs
    ])
    
    context_template_str = getattr(context_template, "prompt", getattr(context_template, "template", ""))
    final_context = context_template_str.replace("{retrieved_context}", formatted_context)
    
    # Generate
    sys_name = prompt_map.get("system_default", "system_default")
    task_name = prompt_map.get("task_rag_qa", "task_rag_qa")
    system_prompt = prompt_manager.get_prompt(sys_name)
    task_prompt = prompt_manager.get_prompt(task_name)
    
    task_prompt_str = getattr(task_prompt, "prompt", getattr(task_prompt, "template", ""))
    qa_prompt_part = task_prompt_str.replace("{user_query}", user_query)
    
    full_prompt = f"{final_context}\n\n[Plan]\n{plan}\n\n" + qa_prompt_part
    
    if state.get("critique_feedback"):
        full_prompt += f"\n\n[Previous Critique]\n{state['critique_feedback']}\nPlease fix this."

    from langchain_core.messages import SystemMessage, HumanMessage
    model = router.get_model(task_type="complex")
    response = await model.ainvoke([
        SystemMessage(content=system_prompt.compile()),
        HumanMessage(content=full_prompt)
    ], config=config)
    
    ai_message = AIMessage(
        content=response.content,
        usage_metadata=getattr(response, "usage_metadata", {}),
        response_metadata={
            "model": model.model_name,
            **getattr(response, "response_metadata", {})
        }
    )
    
    return {"messages": [ai_message], "context": formatted_context}

@observe()
async def critic_node(state: AdvancedAgentState, config: RunnableConfig):
    context = state["context"]
    answer = state["messages"][-1].content
    prompt_map = state.get("prompt_map", {})
    
    critic_name = prompt_map.get("agent_critic", "agent_critic")
    critic_prompt = prompt_manager.get_prompt(critic_name)
    critic_prompt_str = critic_prompt.compile()
    prompt = critic_prompt_str.replace("{context}", context).replace("{answer}", answer)
    
    model = router.get_model(task_type="simple")
    response = await model.ainvoke(prompt, config=config)
    critique = response.content
    
    # Simple parsing of Score
    score = 0.5
    try:
        import re
        match = re.search(r"\[SCORE\]:\s*([\d\.]+)", critique)
        if match:
            score = float(match.group(1))
    except:
        pass
        
    return {"critique_score": score, "critique_feedback": critique, "retry_count": state["retry_count"] + 1}

# 3. Conditional Logic
def check_critique(state: AdvancedAgentState):
    if state["critique_score"] >= 0.8:
        return "end"
    if state["retry_count"] > 2:
        return "end" # Give up after 2 retries
    return "retry"

# 4. Build Graph
def build_advanced_graph():
    workflow = StateGraph(AdvancedAgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "critic")
    
    workflow.add_conditional_edges(
        "critic",
        check_critique,
        {
            "end": END,
            "retry": "executor"
        }
    )
    
    return workflow.compile()

advanced_graph = build_advanced_graph()
