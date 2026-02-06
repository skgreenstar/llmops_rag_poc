import sys
import os
import asyncio
import json

# Fix path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.simple_agent import agent_graph
from app.ops.evaluator import evaluator
from app.ops.monitor import observable
from langchain_core.messages import HumanMessage

@observable(name="rag")
async def evaluate_case(case: dict, dataset_name: str = "manual"):
    """Evaluates a single test case with its own Langfuse trace."""
    query = case.get('query') or case.get('input')
    if not query:
        print("   ⚠️ Case missing query/input, skipping.")
        return

    print(f"\n🧪 Test Case: {query}")
    
    from langfuse.decorators import langfuse_context
    if langfuse_context:
        langfuse_context.update_current_trace(
            tags=[dataset_name, "eval-run"],
            metadata={
                "dataset_source": dataset_name,
                "is_evaluation": True
            }
        )

    # 1. Run Agent
    inputs = {"messages": [HumanMessage(content=query)]}
    result = await agent_graph.ainvoke(inputs)
    
    # Extract Results
    last_msg = result["messages"][-1]
    answer = last_msg.content
    context = result.get("context", "No context returned")
    trace_id = result.get("metadata", {}).get("trace_id")
    
    # Capture Agent Usage
    agent_usage = getattr(last_msg, "usage_metadata", {})
    if langfuse_context and agent_usage:
        langfuse_context.update_current_trace(
            metadata={"agent_usage": agent_usage}
        )

    print(f"   📝 Answer: {answer[:100]}...")
    print(f"   🆔 Trace ID: {trace_id}")
    
    if not trace_id:
        print("   ⚠️ No Trace ID found, skipping score submission.")
        return

    # Link to Dataset Item if available (Correct SDK usage)
    dataset_item = case.get('_item_object') # Pass the object itself if possible
    if dataset_item:
        try:
             # Standard Langfuse way to link observation to dataset item
             dataset_item.link(trace_id=trace_id)
        except Exception as e:
             # print(f"   ⚠️ Failed to link dataset item: {e}")
             pass
        
    # 2. Run Judges (Exhaustive Metrics)
    print("   ⚖️  Running Ragas Evaluation (Exhaustive Metrics)...")
    
    # Handle both hardcoded and Langfuse Dataset format for reference
    reference = case.get('ground_truth')
    if not reference and 'expected_output' in case:
        reference = case['expected_output']
        # If it's a JSON string (typical for Langfuse prompts), try to parse it
        try:
            val = json.loads(reference)
            if isinstance(val, str): reference = val
        except:
            pass

    eval_results = await evaluator.run_ragas_eval(
        query=query, 
        context=context, 
        answer=answer,
        reference=reference
    )
    
    for score_result in eval_results:
        evaluator.submit_score(trace_id, score_result)

async def run_evaluation():
    print("🚀 Starting Automated Evaluation...")
    
    test_cases = []
    
    # Try fetching from Langfuse Dataset
    try:
        from langfuse import Langfuse
        langfuse = Langfuse()
        
        dataset_name = "golden_set"
        print(f"📥 Fetching test cases from Langfuse Dataset: '{dataset_name}'...")
        # Get the dataset object
        try:
            dataset = langfuse.get_dataset(dataset_name)
            for item in dataset.items:
                test_cases.append({
                    "input": item.input,
                    "expected_output": item.expected_output,
                    "_item_object": item # Pass the item object for linking
                })
        except:
            print(f"ℹ️ Dataset '{dataset_name}' not found.")
            
        if test_cases:
            print(f"✅ Loaded {len(test_cases)} cases from Langfuse.")
    except Exception as e:
        print(f"ℹ️ Langfuse Dataset fetch failed: {e}. Using hardcoded fallback.")

    # Fallback if no dataset items found
    if not test_cases:
        dataset_name = "hardcoded"
        test_cases = [
            {
                "query": "What is LangGraph?",
                "ground_truth": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agentic workflows.",
            },
            {
                "query": "What framework is used for the API?",
                "ground_truth": "The API is built using the FastAPI framework.",
            }
        ]
        print(f"⚠️ Using {len(test_cases)} hardcoded test cases.")
    
    # Limit to first 2 cases if dataset is too large, to avoid long wait (optional, but good for demo)
    # if len(test_cases) > 2:
    #     print(f"⚡️ Limiting to first 2 cases for speed (Total: {len(test_cases)})")
    #     test_cases = test_cases[:2]

    for case in test_cases:
        await evaluate_case(case, dataset_name=dataset_name)
        
    print("\n✅ Evaluation Complete! Check Langfuse Dashboard for scores.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
