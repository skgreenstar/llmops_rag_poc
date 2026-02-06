import json
import requests
import asyncio

BASE_URL = "http://127.0.0.1:8000"

async def test_crag_web_fallback():
    print("\n--- Testing Phase 5: CRAG Web Fallback ---")
    # A question definitely NOT in the propulsion manual
    payload = {
        "message": "2024년 노벨 문학상 수상자는 누구야?", 
        "task_type": "simple"
    }
    
    url = f"{BASE_URL}/chat/stream"
    print(f"Asking: {payload['message']}")
    
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    if data["event"] == "node":
                        if data["name"] == "web_search":
                            print("[Status: WEB SEARCH TRIGGERED! ✅]")
                    elif data["event"] == "done":
                        print(f"Response: {data['response'][:100]}...")

async def test_summarization():
    print("\n--- Testing Phase 5: Conversation Summarization ---")
    session_id = "test_sum_session"
    
    # Send 11 messages to trigger summarization (>10)
    for i in range(11):
        payload = {
            "message": f"Message {i+1}: Let's talk about testing.",
            "session_id": session_id,
            "task_type": "simple"
        }
        resp = requests.post(f"{BASE_URL}/chat", json=payload)
        # Check logs or trace if needed, but we look for the 'summarize' node if we used stream
        print(f"Sent message {i+1}...")

    # The 12th message should definitely trigger summary if not already
    payload = {
        "message": "Can you summarize our testing talk?",
        "session_id": session_id,
        "task_type": "simple"
    }
    
    print("\nSending 12th message to check for summary node...")
    with requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8")[6:])
                if line_data["event"] == "node" and line_data["name"] == "summarize":
                    print("[Status: SUMMARization TRIGGERED! ✅]")
                elif line_data["event"] == "done":
                    if line_data.get("summary"):
                        print(f"Final Summary: {line_data['summary'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_crag_web_fallback())
    asyncio.run(test_summarization())
