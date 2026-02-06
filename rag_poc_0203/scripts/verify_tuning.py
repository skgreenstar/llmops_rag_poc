import asyncio
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

async def test_hybrid_search():
    print("--- Testing Hybrid Search & Threshold ---")
    
    # 1. Ingest specific keyword document
    print("1. Ingesting keyword document...")
    ingest_payload = {
        "text": "The project code name is Project-Antigravity-X100. It is a secret RAG optimization project.",
        "collection_name": "performance_test",
        "filename": "keyword_test.txt",
        "preset": "granular"
    }
    resp = requests.post(f"{BASE_URL}/rag/ingest", json=ingest_payload)
    print(f"Ingest status: {resp.status_code}, {resp.json()}")
    
    # Wait for background task (simplified)
    await asyncio.sleep(2)
    
    # 2. Test Vector Search (might fail on exact code if embedding is not precise)
    print("\n2. Testing Vector Search for 'Antigravity-X100'...")
    chat_payload_vec = {
        "message": "What is Project-Antigravity-X100?",
        "search_type": "vector",
        "collection_name": "performance_test",
        "top_k": 1
    }
    resp_vec = requests.post(f"{BASE_URL}/chat", json=chat_payload_vec)
    print(f"Vector Result: {resp_vec.json().get('response')[:50]}...")
    
    # 3. Test Keyword Search (should succeed on exact matches)
    print("\n3. Testing Keyword Search for 'Antigravity-X100'...")
    chat_payload_kw = {
        "message": "Project-Antigravity-X100",
        "search_type": "keyword",
        "collection_name": "performance_test",
        "top_k": 1
    }
    resp_kw = requests.post(f"{BASE_URL}/chat", json=chat_payload_kw)
    print(f"Keyword Result: {resp_kw.json().get('response')[:50]}...")

    # 4. Test Score Threshold (High threshold should exclude irrelevant docs)
    print("\n4. Testing Score Threshold (High)...")
    chat_payload_thresh = {
        "message": "Tell me about something completely unrelated",
        "score_threshold": 0.9,
        "collection_name": "performance_test"
    }
    resp_thresh = requests.post(f"{BASE_URL}/chat", json=chat_payload_thresh)
    print(f"Threshold (0.9) Result: {resp_thresh.json().get('response')[:50]}...")

if __name__ == "__main__":
    asyncio.run(test_hybrid_search())
