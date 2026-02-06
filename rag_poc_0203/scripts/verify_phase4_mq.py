import asyncio
import requests

BASE_URL = "http://127.0.0.1:8000"

async def test_multi_query():
    print("--- Testing Phase 4: Multi-Query Transformation ---")
    
    # Text a query that might benefit from expansion
    payload = {
        "message": "Antigravity-X100에 대해 설명해주고 주요 특징을 알려줘.",
        "task_type": "simple"
    }
    
    print(f"Sending request: {payload['message']}")
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    data = resp.json()
    
    print(f"Response: {data.get('response')[:200]}...")
    
    # Note: We can check the logs to see if multi-query was triggered
    # and if parallel retrieval tasks were executed.
    print("\nCheck server logs for 'DEBUG [Grader]:' and 'multi_query tags' in trace.")

if __name__ == "__main__":
    asyncio.run(test_multi_query())
