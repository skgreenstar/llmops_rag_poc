import asyncio
from app.rag.retriever import retriever
from qdrant_client import models

async def diagnostic():
    print("--- Qdrant Diagnostic ---")
    collections = retriever.list_collections()
    print(f"Available Collections: {collections}")
    
    for coll in collections:
        count = retriever.client.count(collection_name=coll).count
        print(f"Collection: {coll}, Point Count: {count}")
        
        # Peek at the first 5 records
        res = retriever.client.scroll(
            collection_name=coll,
            limit=5,
            with_vectors=True
        )[0]
        
        for i, point in enumerate(res):
            payload = point.payload or {}
            content = payload.get('content', '') or ''
            source = payload.get('source', 'unknown')
            
            vector_preview = str(point.vector)[:100] + "..."
            # Check if it's the dummy vector [0.1, 0.1, ...]
            is_dummy = all(abs(v - 0.1) < 1e-5 for v in point.vector[:10])
            print(f"  [{i}] ID: {point.id}, Source: {source}, Content Preview: {content[:50]}...")
            print(f"      Vector Preview: {vector_preview}")
            print(f"      Is Dummy Vector: {is_dummy}")

if __name__ == "__main__":
    asyncio.run(diagnostic())
