import asyncio
import json
import uuid
from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.agents.simple_agent import agent_graph
from app.agents.advanced_agent import advanced_graph
from app.ops.monitor import observable
from app.ops.router_logic import classify_intent
from app.core.database import get_db_session, ChatSession, ChatMessage, async_session
from app.core.prompts import prompt_manager
from langchain_core.messages import HumanMessage, AIMessage

try:
    from langfuse.decorators import langfuse_context, observe
except ImportError:
    langfuse_context = None
    observe = lambda *args, **kwargs: (lambda f: f) # Fallback

import subprocess
import sys
import os
from app.rag.retriever import retriever 

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.PROJECT_NAME}...")
    from app.core.database import init_db
    await init_db()
    yield
    print("Shutting down...")

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    task_type: str = "auto" # simple, complex, or auto
    prompt_map: dict[str, str] = {}
    collection_name: str = "knowledge_base"
    top_k: int = 3
    use_reranker: bool = False
    search_type: str = "vector"
    score_threshold: float = 0.0
    graph_mode: Optional[str] = "hybrid"
    filters: Optional[dict] = None

class IngestRequest(BaseModel):
    text: str
    collection_name: str = "knowledge_base"
    filename: Optional[str] = "manual_ingest"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    preset: str = "general"

class ChatResponse(BaseModel):
    response: str
    trace_id: str | None = None
    session_id: str

class FeedbackRequest(BaseModel):
    trace_id: str
    score: float 
    comment: Optional[str] = None
    name: str = "human-feedback"

@app.post("/chat/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    try:
        from app.ops.evaluator import evaluator
        evaluator.langfuse.score(
            trace_id=request.trace_id,
            name=request.name,
            value=request.score,
            comment=request.comment
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat", response_model=ChatResponse)
@observe(name="api_chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Update Langfuse context (Modern Best Practice)
    if langfuse_context:
        try:
            langfuse_context.update_current_trace(
                session_id=session_id,
                input=request.message,
                metadata={
                    "task_type": request.task_type,
                    "collection_name": request.collection_name,
                    "top_k": request.top_k
                }
            )
        except Exception as e:
            print(f"Langfuse context error: {e}")

    async with async_session() as db:

        # 1. Ensure Session exists in DB
        result = await db.execute(select(ChatSession).filter(ChatSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            session = ChatSession(id=session_id)
            db.add(session)
            await db.commit()

        # 2. Dynamic Routing (Auto-Intent)
        task_mode = request.task_type
        if task_mode == "auto":
            task_mode = await classify_intent(request.message)
        
        # Load History
        history_result = await db.execute(
            select(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
        )
        history_msgs = history_result.scalars().all()
        lc_history = []
        for m in history_msgs:
            if m.role == "user":
                lc_history.append(HumanMessage(content=m.content))
            else:
                lc_history.append(AIMessage(content=m.content))
                
        # Include latest message
        lc_history.append(HumanMessage(content=request.message))
        
        # 3. Choose Graph
        inputs = {
            "messages": lc_history,
            "summary": session.summary or "",
            "collection_name": request.collection_name,
            "retrieval_config": {
                "top_k": request.top_k,
                "use_reranker": request.use_reranker,
                "search_type": request.search_type,
                "graph_mode": request.graph_mode,
                "score_threshold": request.score_threshold,
                "metadata_filter": request.filters
            },
            "metadata": {"session_id": session_id}
        }
        
        if task_mode == "complex":
            graph = advanced_graph
            inputs.update({
                "plan": "",
                "context": "",
                "critique_score": 0.0,
                "critique_feedback": "",
                "retry_count": 0,
                "prompt_map": request.prompt_map,
            })
        else:
            graph = agent_graph
            inputs.update({
                "prompt_map": request.prompt_map,
            })

        # 4. Invoke Graph
        config = {
            "configurable": {"session_id": session_id},
            "metadata": {
                "langfuse_session_id": session_id
            }
        }
        
        final_response = await graph.ainvoke(inputs, config=config)
        final_message = final_response["messages"][-1].content
        final_summary = final_response.get("summary", session.summary)

        # 5. Persist messages and summary
        if final_summary != session.summary:
            session.summary = final_summary
            
        user_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="user", content=request.message)
        ai_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="assistant", content=final_message)
        db.add_all([user_msg, ai_msg])
        await db.commit()

        # Update output in Langfuse
        if langfuse_context:
            try:
                langfuse_context.update_current_trace(output=final_message)
            except Exception:
                pass

        return ChatResponse(
            response=final_message,
            session_id=session_id,
            trace_id=langfuse_context.get_current_trace_id() if langfuse_context else None
        )

@app.post("/chat/stream")
@observe(name="api_chat_stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming version of the chat endpoint using SSE.
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    # 1. Update Langfuse context
    if langfuse_context:
        try:
            langfuse_context.update_current_trace(
                session_id=session_id,
                input=request.message,
                metadata={
                    "task_type": request.task_type,
                    "collection_name": request.collection_name,
                    "top_k": request.top_k,
                    "search_type": request.search_type
                }
            )
        except Exception as e:
            print(f"Langfuse init error (stream): {e}")

    trace_id = langfuse_context.get_current_trace_id() if langfuse_context else None

    async def event_generator():
        async with async_session() as db:
            # 1. Ensure Session exists in DB
            result = await db.execute(select(ChatSession).filter(ChatSession.id == session_id))
            session = result.scalar_one_or_none()
            if not session:
                session = ChatSession(id=session_id)
                db.add(session)
                await db.commit()
                
            # 2. Load History
            history_result = await db.execute(
                select(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
            )
            history_msgs = history_result.scalars().all()
            lc_history = []
            for m in history_msgs:
                if m.role == "user":
                    lc_history.append(HumanMessage(content=m.content))
                else:
                    lc_history.append(AIMessage(content=m.content))
            
            lc_history.append(HumanMessage(content=request.message))
    
            # 3. Setup Graph Inputs
            task_mode = request.task_type
            if task_mode == "auto":
                task_mode = await classify_intent(request.message)
            
            graph = advanced_graph if task_mode == "complex" else agent_graph
            inputs = {
                "messages": lc_history,
                "summary": session.summary or "",
                "collection_name": request.collection_name,
                "retrieval_config": {
                    "top_k": request.top_k,
                    "use_reranker": request.use_reranker,
                    "search_type": request.search_type,
                    "graph_mode": request.graph_mode,
                    "score_threshold": request.score_threshold,
                    "metadata_filter": request.filters
                },
                "metadata": {"session_id": session_id}
            }
            
            # Initial Metadata
            yield f"data: {json.dumps({'event': 'metadata', 'session_id': session_id, 'trace_id': trace_id})}\n\n"
            
            full_response = ""
            retrieved_docs = []
            final_summary = session.summary or ""
            
            # 2. Iterate graph events
            config = {
                "configurable": {"session_id": session_id},
                "metadata": {
                    "langfuse_session_id": session_id
                }
            }

            async for event in graph.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                node = event["metadata"].get("langgraph_node", "")
                
                if kind == "on_chat_model_stream" and node in ["generate", "executor"]:
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'event': 'chunk', 'text': content})}\n\n"
                
                elif kind == "on_chain_start" and node:
                    yield f"data: {json.dumps({'event': 'node', 'name': node})}\n\n"
                
                elif kind == "on_chain_end" and node == "retrieve":
                    retrieved_docs = event["data"]["output"].get("retrieved_docs", [])
                
                elif kind == "on_chain_end" and node == "summarize":
                    final_summary = event["data"]["output"].get("summary", "")
    
            # 3. Finalize & Persist
            if final_summary != session.summary:
                session.summary = final_summary
                
            user_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="user", content=request.message)
            ai_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="assistant", content=full_response)
            db.add_all([user_msg, ai_msg])
            await db.commit()
            
            # Update Langfuse output via client (context might be lost in generator)
            if trace_id:
                try:
                    prompt_manager.langfuse.trace(id=trace_id).update(output=full_response)
                except Exception:
                    pass

            yield f"data: {json.dumps({'event': 'done', 'response': full_response, 'retrieved_docs': retrieved_docs, 'summary': final_summary})}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/eval/run")
async def run_evaluation_endpoint(background_tasks: BackgroundTasks):
    def _run_script():
        try:
            cmd = [sys.executable, "scripts/run_eval.py"]
            result = subprocess.run(
                cmd, 
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                capture_output=True, 
                text=True
            )
            print("Evaluation Output:", result.stdout)
        except Exception as e:
            print(f"Evaluation Failed: {e}")

    background_tasks.add_task(_run_script)
    return {"status": "started", "message": "Evaluation started in background."}

@app.get("/rag/collections")
async def list_collections_endpoint():
    return {"collections": retriever.list_collections()}


@app.post("/rag/ingest")
async def ingest_endpoint(request: IngestRequest):
    try:
        # Directly await ingestion (it's already optimized internally)
        await retriever.ingest_documents(
            request.text, 
            request.collection_name, 
            request.filename,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            preset=request.preset
        )
        return {
            "status": "completed", 
            "message": f"Ingestion of '{request.filename}' completed successfully."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ingestion failed: {str(e)}"
        }

@app.get("/chat/sessions")
async def list_sessions_endpoint():
    """
    Lists all chat sessions with their summaries and creation times.
    """
    async with async_session() as db:
        result = await db.execute(select(ChatSession).order_by(ChatSession.created_at.desc()))
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "summary": s.summary or "New Conversation",
                "created_at": s.created_at.isoformat() if s.created_at else None
            } for s in sessions
        ]

@app.get("/chat/sessions/{session_id}/messages")
async def get_session_messages_endpoint(session_id: str):
    """
    Retrieves all messages for a specific session.
    """
    async with async_session() as db:
        result = await db.execute(
            select(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None
            } for m in messages
        ]

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
