import streamlit as st
import requests
import os
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

# Config
API_URL = "http://127.0.0.1:8000/chat"
EVAL_API_URL = "http://127.0.0.1:8000/eval/run"
COLLECTIONS_API_URL = "http://127.0.0.1:8000/rag/collections"
INGEST_API_URL = "http://127.0.0.1:8000/rag/ingest"
FEEDBACK_API_URL = "http://127.0.0.1:8000/chat/feedback"
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
DATA_PATH = "data/golden_set.json"

st.set_page_config(page_title="AI 서비스 관리자", layout="wide")

st.title("🤖 AI 서비스 관리자 대시보드")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

# Tabs
tab_chat, tab_data, tab_eval = st.tabs(["💬 채팅 & 테스트", "📊 데이터셋 관리", "✅ 평가 실행"])

# --- TAB 1: Chat ---
with tab_chat:
    # Sidebar: Control & Observability
    with st.sidebar:
        st.header("⚙️ 환경 설정")
        
        st.divider()
        st.header("🕒 대화 기록 (History)")
        
        @st.cache_data(ttl=10)
        def get_all_sessions():
            try:
                resp = requests.get("http://127.0.0.1:8000/chat/sessions")
                if resp.status_code == 200:
                    return resp.json()
                return []
            except:
                return []

        sessions = get_all_sessions()
        if sessions:
            # Create a dictionary for display
            session_options = {f"{s['summary'][:30]}... ({s['created_at'][:10]})": s['id'] for s in sessions}
            session_labels = list(session_options.keys())
            
            # Find current session index if it exists in history
            curr_idx = 0
            curr_sess_id = st.session_state.current_session_id
            for i, sid in enumerate(session_options.values()):
                if sid == curr_sess_id:
                    curr_idx = i
                    break
            
            selected_session_label = st.selectbox(
                "과거 대화 불러오기",
                options=session_labels,
                index=curr_idx
            )
            
            target_sid = session_options[selected_session_label]
            if target_sid != st.session_state.current_session_id:
                # Load selected session
                try:
                    msg_resp = requests.get(f"http://127.0.0.1:8000/chat/sessions/{target_sid}/messages")
                    if msg_resp.status_code == 200:
                        st.session_state.messages = msg_resp.json()
                        st.session_state.current_session_id = target_sid
                        st.session_state.pop("last_trace_id", None)
                        st.session_state.pop("last_retrieved_docs", None)
                        st.rerun()
                except Exception as e:
                    st.error(f"대화 로드 실패: {e}")

        if st.button("➕ 새 대화 시작 (New Chat)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.pop("last_trace_id", None)
            st.session_state.pop("last_retrieved_docs", None)
            st.rerun()

        st.divider()

        agent_mode = st.radio(
            "🧩 에이전트 모드 (Agent Mode)",
            options=["auto", "simple", "complex"],
            index=0,
            format_func=lambda x: {
                "auto": "🧠 Automatic (지능형 라우팅)",
                "simple": "⚡️ Fast RAG (기본)",
                "complex": "🧩 Advanced Agent (Planner/Critic)"
            }[x]
        )
        task_type = agent_mode
        
        st.divider()
        
        st.header("📊 관찰 가능성 (Observability)")
        st.info(f"세션 ID: `{st.session_state.current_session_id}`")
        if "last_trace_id" in st.session_state:
            trace_id = st.session_state["last_trace_id"]
            st.success(f"Trace ID: `{trace_id}`")
            # Link to generic Langfuse (user can adjust project-id)
            st.markdown(f"🔗 [Langfuse 상세 보기]({LANGFUSE_HOST}/traces/{trace_id})")
        else:
            st.info("아직 트레이스가 없습니다.")
            
        st.divider()
        st.header("🎛️ 프롬프트 오버라이드")
        
        @st.cache_data(ttl=5)
        def get_all_prompt_names():
            try:
                import requests
                from requests.auth import HTTPBasicAuth
                resp = requests.get(
                    f"{LANGFUSE_HOST}/api/public/prompts",
                    auth=HTTPBasicAuth(os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"))
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return sorted(list(set([item["name"] for item in data.get("data", [])])))
                return []
            except Exception as e:
                st.sidebar.error(f"Langfuse API 연결 오류: {e}")
                return []
        
        if st.button("🔄 프롬프트 목록 새로고침"):
            get_all_prompt_names.clear()
            st.rerun()
            
        available_prompts = get_all_prompt_names()

        st.divider()
        st.header("📚 지식 베이스 (Knowledge Base)")
        
        @st.cache_data(ttl=10)
        def get_collections():
            try:
                resp = requests.get(COLLECTIONS_API_URL)
                if resp.status_code == 200:
                    return resp.json().get("collections", [])
                return []
            except:
                return []

        available_collections = get_collections()
        if not available_collections:
             available_collections = ["knowledge_base"]

        selected_collection = st.selectbox(
            "검색 대상 컬렉션 (Collection)", 
            options=available_collections
        )
        if st.button("🔄 컬렉션 목록 새로고침"):
            get_collections.clear()
            st.rerun()
            
        st.divider()
        st.header("🔎 검색 설정 (Retrieval)")
        top_k = st.slider("검색 결과 수 (Top-K)", min_value=1, max_value=10, value=3)
        search_type = st.radio("검색 모델 (Search Type)", options=["vector", "keyword", "hybrid", "graph"], horizontal=True)
        
        graph_mode = "hybrid"
        if search_type == "graph":
            graph_mode = st.selectbox("그래프 검색 모드 (Graph Mode)", options=["local", "global", "hybrid", "naive"], index=2)
            st.info("💡 Graph Search는 데이터 간의 관계를 분석하여 정교한 답변을 생성합니다.")

        use_reranker = st.checkbox("Reranking 적용 (정확도 향상)", value=False)
        score_threshold = st.slider("최소 유사도 점수 (Score Threshold)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        
        st.subheader("🎯 메타데이터 필터")
        filter_source = st.text_input("출처(Source) 필터 (예: manual.pdf)", value="")
        metadata_filters = {}
        if filter_source:
            metadata_filters["source"] = filter_source

        PROMPT_METADATA = {
            "system_default": {"label": "🤖 시스템 페르소나", "desc": "AI 역할 정의", "vars": []},
            "rag_context": {"label": "📄 Context 주입", "desc": "문맥 전달 형식", "vars": ["{retrieved_context}"]},
            "task_rag_qa": {"label": "❓ 답변 작성 지침", "desc": "구체적 지시사항", "vars": ["{user_query}"]},
            "agent_planner": {"label": "📅 계획 수립", "desc": "하위 작업 분할 논리", "vars": ["{user_query}"]},
            "agent_critic": {"label": "🧐 비평가", "desc": "사실 검증 논리", "vars": ["{context}", "{answer}"]}
        }
        
        prompt_map = {}
        for key, meta in PROMPT_METADATA.items():
            options = sorted(list(set([key] + available_prompts)))
            idx = options.index(key) if key in options else 0
            st.markdown(f"**{meta['label']}**")
            selected_val = st.selectbox(f"Select Prompt for {key}", options=options, index=idx, key=f"sb_{key}", label_visibility="collapsed")
            if selected_val != key:
                prompt_map[key] = selected_val
            st.markdown("---")
            
    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("지식 베이스에 대해 질문해보세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("생각 중...")
            
            try:
                payload = {
                    "message": prompt, 
                    "session_id": st.session_state.current_session_id,
                    "task_type": task_type,
                    "prompt_map": prompt_map,
                    "collection_name": selected_collection,
                    "top_k": top_k,
                    "use_reranker": use_reranker,
                    "search_type": search_type,
                    "graph_mode": graph_mode,
                    "score_threshold": score_threshold,
                    "filters": metadata_filters
                }
                
                # Use streaming endpoint
                STREAM_API_URL = API_URL + "/stream"
                
                status_placeholder = st.empty()
                
                def stream_generator():
                    with requests.post(STREAM_API_URL, json=payload, stream=True) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    data = json.loads(line_str[6:])
                                    if data["event"] == "chunk":
                                        yield data["text"]
                                    elif data["event"] == "metadata":
                                        st.session_state["last_trace_id"] = data["trace_id"]
                                    elif data["event"] == "node":
                                        node_name = data["name"]
                                        if node_name == "rewrite_query":
                                            status_placeholder.info("🔍 **질문 최적화 중...** (Multi-Query Expansion)")
                                        elif node_name == "retrieve":
                                            status_placeholder.info("📚 **지식 베이스 검색 중...** (Point-Cloud Search)")
                                        elif node_name == "grade_docs":
                                            status_placeholder.info("⚖️ **검색 결과 관련성 심사 중...** (Self-Correction)")
                                        elif node_name == "web_search":
                                            status_placeholder.warning("🌐 **내부 정보 부족: 웹 검색으로 보완 중...** (CRAG Fallback)")
                                        elif node_name == "planner":
                                            status_placeholder.info("📋 **수행 계획 수립 중...** (Advanced Reasoner)")
                                        elif node_name == "executor":
                                            status_placeholder.info("⚙️ **작업 수행 및 답변 생성 중...**")
                                        elif node_name == "critic":
                                            status_placeholder.info("🧐 **답변 검증 및 품질 평가 중...** (Refinement Loop)")
                                        elif node_name == "summarize":
                                            status_placeholder.info("🧠 **대화 요약 및 기억 업데이트 중...**")
                                        elif node_name == "generate":
                                            status_placeholder.empty()
                                    elif data["event"] == "done":
                                        st.session_state["last_retrieved_docs"] = data.get("retrieved_docs", [])
                                        status_placeholder.empty()

                answer = message_placeholder.write_stream(stream_generator())
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun() 
            except Exception as e:
                message_placeholder.error(f"오류 발생: {e}")

    # Trace & Feedback & Retrieval Inspector
    if "last_trace_id" in st.session_state and st.session_state.messages:
        curr_trace_id = st.session_state["last_trace_id"]
        
        st.divider()
        with st.expander("🔍 검색 결과 인스펙터 (Retrieval Inspector)", expanded=False):
            st.info("방금 전 질문에 사용된 지식 베이스 검색 결과입니다.")
            
            docs = st.session_state.get("last_retrieved_docs", [])
            if docs:
                for i, d in enumerate(docs):
                    st.markdown(f"**[{i+1}] {d['source']}** (유사도: `{d['score']:.4f}`)")
                    st.text_area(f"내용 {i+1}", d['content'], height=100, key=f"doc_{i}")
            else:
                st.warning("추출된 문서가 없습니다. (Self-RAG에 의해 거절되었거나 검색 결과가 없을 수 있습니다.)")
            
            st.markdown(f"🔗 [Langfuse에서 트레이스 자세히 보기]({LANGFUSE_HOST}/project/project-123/traces/{curr_trace_id})")

        st.subheader("📬 답변 평가 및 분석")
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("👍 좋아요", key="btn_thumbs_up"):
                requests.post(FEEDBACK_API_URL, json={"trace_id": curr_trace_id, "score": 1, "name": "user-thumb"})
                st.toast("피드백이 기록되었습니다!")
        with c2:
            if st.button("👎 싫어요", key="btn_thumbs_down"):
                requests.post(FEEDBACK_API_URL, json={"trace_id": curr_trace_id, "score": 0, "name": "user-thumb"})
                st.toast("피드백이 기록되었습니다.")
        with c3:
             st.markdown(f"🔗 [Langfuse 상세 분석]({LANGFUSE_HOST}/project/project-123/traces/{curr_trace_id})")

# --- TAB 2: Ingest ---
with tab_data:
    st.header("📥 데이터 적재 (Ingest)")
    with st.expander("데이터 적재 패널", expanded=False):
        uploaded_file = st.file_uploader("파일 업로드", type=["pdf", "txt", "md", "json", "csv", "py"])
        target_collection = st.text_input("컬렉션 이름", value="knowledge_base")
        ingest_preset = st.selectbox("청킹 프리셋", options=["general", "legal", "code", "granular"])
        
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.slider("청크 크기", 100, 4000, 1000, 100)
        with c2:
            overlap_percent = st.slider("오버랩 (%)", 0, 50, 10, 5)
            chunk_overlap = int(chunk_size * (overlap_percent / 100))
            st.caption(f"실제 오버랩: {chunk_overlap} 자")

        if st.button("🚀 실행"):
            if uploaded_file:
                with st.spinner("적재 중..."):
                    try:
                        file_content = ""
                        if uploaded_file.name.lower().endswith('.pdf'):
                            import tempfile, pdf4llm
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            file_content = pdf4llm.to_markdown(tmp_path)
                            os.remove(tmp_path)
                        else:
                            file_content = uploaded_file.getvalue().decode("utf-8")
                        
                        payload = {
                            "text": file_content,
                            "collection_name": target_collection,
                            "filename": uploaded_file.name,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "preset": ingest_preset
                        }
                        resp = requests.post(INGEST_API_URL, json=payload)
                        if resp.status_code == 200:
                            st.success(f"적재 완료! {resp.json().get('message')}")
                        else:
                            st.error(f"적재 실패: {resp.text}")
                    except Exception as e:
                        st.error(f"적재 오류: {e}")

# --- TAB 3: Evaluation ---
with tab_eval:
    st.header("평가 데이터셋 관리")
    try:
        with open(DATA_PATH, "r") as f:
            current_data = json.load(f)
        updated_data = st.data_editor(current_data, num_rows="dynamic")
        if st.button("💾 저장"):
            with open(DATA_PATH, "w") as f:
                json.dump(updated_data, f, indent=4)
            st.success("로컬 저장 완료!")
    except Exception as e:
        st.error(f"데이터셋 로드 오류: {e}")

    st.divider()
    if st.button("🚀 전체 평가 시작"):
        with st.spinner("진행 중..."):
            try:
                resp = requests.post(EVAL_API_URL)
                st.success(f"요청 성공: {resp.json().get('message')}")
            except Exception as e:
                st.error(f"실행 오류: {e}")
