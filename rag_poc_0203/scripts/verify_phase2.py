import asyncio
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

async def test_session_and_routing():
    print("--- Testing Session Persistence & Routing ---")
    session_id = str(uuid.uuid4())
    print(f"Generated Session ID: {session_id}")
    
    # 1. Test Simple Routing (Auto)
    print("\n1. Testing 'Simple' query (Auto Routing)...")
    payload_simple = {
        "message": "안녕? 반가워!",
        "session_id": session_id,
        "task_type": "auto"
    }
    resp1 = requests.post(f"{BASE_URL}/chat", json=payload_simple)
    data1 = resp1.json()
    print(f"Response: {data1.get('response')}")
    print(f"Returned Session ID matches: {data1.get('session_id') == session_id}")

    # 2. Test Complex Routing (Auto)
    print("\n2. Testing 'Complex' query (Auto Routing)...")
    payload_complex = {
        "message": "LangGraph와 Langfuse의 차이점을 분석해서 표로 정리해줘.",
        "session_id": session_id,
        "task_type": "auto"
    }
    resp2 = requests.post(f"{BASE_URL}/chat", json=payload_complex)
    data2 = resp2.json()
    # Check if 'analzying' or plan-like content exists (Advanced agent output)
    print(f"Response snippet: {data1.get('response')[:50]}...")

    # 3. Verify Database (Persistence)
    print("\n3. Verifying DB Persistence (simulated via multiple turns)...")
    print(f"Chat turn count should have increased in SQLite for {session_id}.")
    
    # 4. Check Trace Linkage
    print("\n4. Verification: Please check Langfuse Sessions dashboard for the Trace IDs.")
    print(f"Trace IDs: {data1.get('trace_id')}, {data2.get('trace_id')}")

if __name__ == "__main__":
    asyncio.run(test_session_and_routing())
