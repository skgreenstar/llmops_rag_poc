import asyncio
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

async def test_reliability_features():
    print("--- Testing Phase 3: Reliability & Filtering ---")
    
    # 0. Ingest Test Data
    print("\n0. Ingesting test data for filtering...")
    ingest_payload = {
        "text": "Antigravity-X100 is a specialized propulsion system designed by the DeepMind team. It uses quantum entanglement for thrust.",
        "collection_name": "knowledge_base",
        "filename": "propulsion_manual.pdf"
    }
    requests.post(f"{BASE_URL}/rag/ingest", json=ingest_payload)
    print("Wait for ingestion...")
    await asyncio.sleep(5) # Wait for background ingestion

    # 1. Test Metadata Filtering (Positive)
    print("\n1. Testing Metadata Filtering (Source-based)...")
    payload_filter = {
        "message": "What is Antigravity-X100?",
        "task_type": "simple",
        "filters": {"source": "propulsion_manual.pdf"}
    }
    resp1 = requests.post(f"{BASE_URL}/chat", json=payload_filter)
    data1 = resp1.json()
    print(f"Response with valid filter: {data1.get('response')[:200]}...")
    
    # 2. Test Metadata Filtering (Negative - Should yield missing info)
    print("\n2. Testing Metadata Filtering with non-existent source...")
    payload_wrong_filter = {
        "message": "What is Antigravity-X100?",
        "task_type": "simple",
        "filters": {"source": "some_other_file.pdf"}
    }
    resp2 = requests.post(f"{BASE_URL}/chat", json=payload_wrong_filter)
    data2 = resp2.json()
    print(f"Response with wrong filter (Should use missing_info): {data2.get('response')}")

    # 3. Test Self-Correction (Self-RAG)
    print("\n3. Testing Self-Correction with irrelevant query...")
    payload_nonsense = {
        "message": "피자 맛있게 만드는 법 알려줘.",
        "task_type": "simple"
    }
    resp3 = requests.post(f"{BASE_URL}/chat", json=payload_nonsense)
    data3 = resp3.json()
    print(f"Response for irrelevant query (Self-RAG): {data3.get('response')}")

if __name__ == "__main__":
    asyncio.run(test_reliability_features())
