from typing import List, Dict
from ddgs import DDGS
from app.ops.monitor import observable

class WebSearchTool:
    def __init__(self):
        self.ddgs = DDGS()

    @observable(name="web_search", as_type="span")
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Performs a web search using DuckDuckGo.
        Returns a list of dicts with 'content', 'source', and 'score' (mocked).
        """
        results = []
        try:
            # Note: duckduckgo-search is synchronous by default, 
            # but we wrap it in a thread-safe way or just call carefully
            # For simplicity in this async env, we can use it directly if it's fast
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for i, r in enumerate(search_results):
                results.append({
                    "content": f"{r.get('title')}: {r.get('body')}",
                    "source": r.get("href"),
                    "score": 1.0 - (i * 0.1) # Naive score based on rank
                })
        except Exception as e:
            print(f"Web search failed: {e}")
            
        return results

web_search_tool = WebSearchTool()
