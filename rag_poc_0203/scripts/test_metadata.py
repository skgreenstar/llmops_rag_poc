import asyncio
import json
from app.agents.simple_agent import agent_graph
from langchain_core.messages import HumanMessage

async def test():
    inputs = {"messages": [HumanMessage(content="Hello")]}
    result = await agent_graph.ainvoke(inputs)
    
    # Get last message
    msg = result["messages"][-1]
    
    print("--- FULL MESSAGE JSON ---")
    # LangChain messages have a .dict() or .json() method or can be converted
    import pprint
    msg_data = {
        "type": msg.type,
        "content": msg.content,
        "usage_metadata": msg.usage_metadata,
        "response_metadata": msg.response_metadata
    }
    pprint.pprint(msg_data)

if __name__ == "__main__":
    asyncio.run(test())
