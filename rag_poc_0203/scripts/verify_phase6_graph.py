import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.retriever import retriever
from app.rag.graph_logic import graph_retriever

async def verify_graph_flow():
    print("--- Phase 6: Graph-based Retrieval Verification ---")
    
    test_text = """
    Antigravity is a powerful agentic AI coding assistant designed by Google Deepmind. 
    It is part of the Advanced Agentic Coding project. 
    Antigravity uses LangGraph for complex workflows and Langfuse for observability.
    The project aims to revolutionize how software is developed through AI pair programming.
    """
    
    print("\n[Step 1] Ingesting test document into Graph & Vector stores...")
    # Ingest using the main retriever which now calls graph_retriever.ingest (async)
    await retriever.ingest_documents(
        text=test_text,
        collection_name="verify_graph_kb",
        filename="antigravity_info.txt"
    )
    
    # Wait a bit for async graph processing if any (though we used asyncio.run in retriever)
    await asyncio.sleep(2)
    
    print("\n[Step 2] Querying Graph (Local Mode)...")
    query = "What is the relationship between Antigravity and LangGraph?"
    # We call retrieve with search_type="graph"
    results = await retriever.retrieve(
        query=query,
        collection_name="verify_graph_kb",
        search_type="graph",
        graph_mode="local"
    )
    
    if results and results[0]["source"] == "Knowledge Graph":
        print(f"Success! Graph Answer:\n{results[0]['content']}")
    else:
        print("Failed to get graph response.")

    print("\n[Step 3] Querying Graph (Global Mode)...")
    query_global = "What is the overall goal of the Antigravity project?"
    results_global = await retriever.retrieve(
        query=query_global,
        collection_name="verify_graph_kb",
        search_type="graph",
        graph_mode="global"
    )
    
    if results_global:
        print(f"Success! Global Graph Answer:\n{results_global[0]['content']}")
    else:
        print("Failed to get global graph response.")

if __name__ == "__main__":
    asyncio.run(verify_graph_flow())
