import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.ops.monitor import observable
from app.core.config import get_settings
from app.rag.graph_logic import graph_retriever
import re

# Custom implementation to avoid dependency issues
class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        length_function = len,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._length_function = length_function

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if self._length_function(text) <= self._chunk_size:
            return [text]
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if re.search(re.escape(_s), text):
                separator = _s
                break
        splits = text.split(separator) if separator else list(text)
        _good_splits = []
        _separator = separator if separator else ""
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged)
                    _good_splits = []
                final_chunks.extend(self.split_text(s))
        if _good_splits:
            final_chunks.extend(self._merge_splits(_good_splits, _separator))
        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (len(separator) if current_doc else 0) > self._chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    while total > self._chunk_overlap or (total + _len + len(separator) > self._chunk_size and total > 0):
                        total -= self._length_function(current_doc[0]) + (len(separator) if len(current_doc) > 1 else 0)
                        current_doc.pop(0)
            current_doc.append(d)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)
        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
        return docs

settings = get_settings()

import warnings
# Suppress QdrantUserWarning about insecure connection (we know it's local)
warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*")

class QdrantRetriever:
    def __init__(self):
        # Initialize Qdrant Client
        # Using memory mode if no host (development) 
        # or connecting to Docker/Cloud if specified
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            check_compatibility=False,  # Bypass 1.16 client vs 1.7 server warning
            # prefer_grpc=True
        )
        self.collection_name = "knowledge_base"
        self._ensure_collection()

    def _ensure_collection(self):
        # Compatibility fix: Client 1.16+ tries to use /exists endpoint which Server 1.8 doesn't have.
        # We try to get the collection info instead.
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # If get_collection fails (likely 404), assume it doesn't exist and create it
            print(f"Collection '{self.collection_name}' not found or error occurred. Creating...")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
                # Add Full-Text Index for Hybrid Search
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="content",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.MULTILINGUAL,
                        lowercase=True,
                    )
                )
                # Insert dummy data for demo
                self._insert_dummy_data()
            except Exception as e:
                print(f"Failed to create collection: {e}")

    def _insert_dummy_data(self):
        docs = [
            "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "Langfuse provides open source observability for LLM applications.",
            "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python."
        ]
        
        points = []
        for idx, text in enumerate(docs):
            try:
                vector = self._embed(text)
            except:
                vector = [0.1] * 1024

            points.append(models.PointStruct(
                id=idx,
                vector=vector,
                payload={"content": text, "source": "dummy_init"}
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            print(f"Failed to list collections: {e}")
            return []

    async def ingest_documents(self, text: str, collection_name: str, filename: str = "manual_ingest", chunk_size: int = 1000, chunk_overlap: int = 100, preset: str = "general"):
        # Mapping presets
        presets = {
            "general": {"size": 1000, "overlap": 100},
            "legal": {"size": 2000, "overlap": 300},  # Larger chunks for legal context
            "code": {"size": 800, "overlap": 50},    # Smaller, precise chunks for code
            "granular": {"size": 500, "overlap": 50}  # Very small chunks for FAQ style
        }
        
        config = presets.get(preset, {"size": chunk_size, "overlap": chunk_overlap})
        actual_size = config["size"]
        actual_overlap = config["overlap"]

        # Ensure collection exists
        try:
            self.client.get_collection(collection_name)
        except Exception:
             print(f"Collection '{collection_name}' not found. Creating...")
             self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
             # Add Full-Text Index
             self.client.create_payload_index(
                collection_name=collection_name,
                field_name="content",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                )
            )
        
        # Smart Chunking using custom implementation (embedded above)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=actual_size,
            chunk_overlap=actual_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        print(f"DEBUG: Splitting text into {len(chunks)} chunks...")
        
        points = []
        for idx, chunk in enumerate(chunks):
            try:
                vector = self._embed(chunk)
                # Use a simple deterministic ID or random UUID
                import uuid
                point_id = str(uuid.uuid4())
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"content": chunk, "source": filename}
                ))
            except Exception as e:
                print(f"Embedding failed for chunk: {chunk[:30]}... Error: {e}")

        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Ingested {len(points)} chunks into '{collection_name}' from '{filename}'")
            
            # Also ingest into Knowledge Graph (now properly async)
            try:
                await graph_retriever.ingest(text)
                print(f"Graph ingestion completed for '{filename}'")
            except Exception as e:
                print(f"Graph ingestion failed for '{filename}': {e}")

    @observable(name="rag_retrieval", as_type="span")
    async def retrieve(self, query: str, top_k: int = 3, collection_name: str = "knowledge_base", limit: int = None, score_threshold: float = 0.0, search_type: str = "vector", metadata_filter: Dict = None, graph_mode: str = "hybrid") -> List[Dict[str, str]]:
        try:
            # Ensure collection exists
            try:
                self.client.get_collection(collection_name)
            except:
                return []
            
            fetch_k = limit if limit else top_k
            results = []

            # Prepare models.Filter if metadata_filter is provided
            qdrant_filter = None
            if metadata_filter:
                must_conditions = []
                for key, value in metadata_filter.items():
                    must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                qdrant_filter = models.Filter(must=must_conditions)

            if search_type == "keyword":
                results = self._search_keyword(query, collection_name, fetch_k, qdrant_filter)
            elif search_type == "hybrid":
                results = await self._search_hybrid(query, collection_name, fetch_k, qdrant_filter)
            elif search_type == "graph":
                # LightRAG returns a full answer, we package it as a document
                graph_answer = await graph_retriever.query(query, mode=graph_mode)
                results = [{"content": graph_answer, "score": 1.0, "source": "Knowledge Graph"}]
            else: # Default: vector
                vector = self._embed(query)
                search_result = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=fetch_k,
                    query_filter=qdrant_filter
                ).points
                results = [
                    {"content": hit.payload.get("content", ""), "score": hit.score, "source": hit.payload.get("source", "unknown")}
                    for hit in search_result
                ]

            # Filter by score_threshold
            return [r for r in results if r["score"] >= score_threshold]
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []

    def _search_keyword(self, query: str, collection_name: str, limit: int, qdrant_filter: models.Filter = None) -> List[Dict]:
        """Performs full-text keyword search."""
        must_conditions = [
            models.FieldCondition(
                key="content",
                match=models.MatchText(text=query)
            )
        ]
        
        # Merge filters if both exist
        if qdrant_filter and qdrant_filter.must:
            must_conditions.extend(qdrant_filter.must)

        search_result = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=must_conditions),
            limit=limit,
            with_payload=True
        )[0]
        
        # Keyword search via scroll/filter doesn't provide a relevance score in the same way query_points does.
        # We assign a dummy high score for matched keywords to surface them.
        return [
            {"content": hit.payload.get("content", ""), "score": 1.0, "source": hit.payload.get("source", "unknown")}
            for hit in search_result
        ]

    async def _search_hybrid(self, query: str, collection_name: str, limit: int, qdrant_filter: models.Filter = None) -> List[Dict]:
        """Combines vector and keyword search results using a simple merge."""
        # 1. Vector Search
        vector = self._embed(query)
        vec_results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            query_filter=qdrant_filter
        ).points
        
        # 2. Keyword Search
        kw_results = self._search_keyword(query, collection_name, limit, qdrant_filter)
        
        # 3. Simple Merge (Priority to Keyword matches if exact, otherwise Vector)
        # In production, RRF (Reciprocal Rank Fusion) is preferred.
        seen_contents = set()
        merged = []
        
        # Add keyword results first (often highly relevant for exact matches)
        for r in kw_results:
            if r["content"] not in seen_contents:
                merged.append(r)
                seen_contents.add(r["content"])
        
        # Add vector results
        for hit in vec_results:
            content = hit.payload.get("content", "")
            if content not in seen_contents:
                merged.append({"content": content, "score": hit.score, "source": hit.payload.get("source", "unknown")})
                seen_contents.add(content)
                
        return merged[:limit]

    def _embed(self, text: str) -> List[float]:
        # 1. Check Binding Strategy
        if settings.EMBEDDING_BINDING == "openai":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            return embeddings.embed_query(text)
        
        # 2. Default to Ollama (Local)
        try:
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                base_url=settings.EMBEDDING_BINDING_HOST,
                model=settings.EMBEDDING_MODEL
            )
            return embeddings.embed_query(text)
        except Exception as e:
            print(f"Ollama embedding ({settings.EMBEDDING_MODEL}) failed: {e}. using dummy.")
            return [0.1] * 1024

# Singleton instance
retriever = QdrantRetriever()
