import json
import requests
import asyncio
import os
import sys

BASE_URL = "http://127.0.0.1:8000"

async def test_streaming():
    print("--- Testing Phase 4: Full Token Streaming & Multi-Query ---")
    
    payload = {
        "message": "Antigravity-X100에 대해 자세히 설명해줘.",
        "task_type": "simple",
        "top_k": 3
    }
    
    # We need to use /chat/stream
    url = f"{BASE_URL}/chat/stream"
    
    print(f"Requesting stream from: {url}")
    
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    event = data.get("event")
                    
                    if event == "chunk":
                        print(data["text"], end="", flush=True)
                    elif event == "node":
                        print(f"\n[Node: {data['name']}]", end=" ")
                    elif event == "done":
                        print("\n\n--- Done ---")
                        print(f"Total Retrieved Docs: {len(data.get('retrieved_docs', []))}")
                    elif event == "metadata":
                        print(f"[Trace ID: {data['trace_id']}]")

if __name__ == "__main__":
    asyncio.run(test_streaming())
