#!/bin/bash
echo "🚀 Starting AI Service Stack..."

# 1. Docker Services
echo "1. Ensuring All Docker services (Apps + Dify) are up..."
docker compose up -d --remove-orphans

# 2. FastAPI Backend
echo "2. Starting Backend API (Port 8000)..."
# Check if port 8000 is free, if not, kill it (optional, risky for now)
poetry run uvicorn app.api.server:app --reload --port 8000 &
PID_API=$!

# 3. Streamlit UI
echo "3. Starting Operator UI (Port 8501)..."
poetry run streamlit run app/ui/streamlit_app.py &
PID_UI=$!

echo "✅ All services started!"
echo "   - API: http://localhost:8000"
echo "   - UI:  http://localhost:8501"
echo "   - Langfuse: http://localhost:3000"
echo "   - Dify: http://localhost:8081 (Admin Setup Required)"
echo "Press Ctrl+C to stop all."

# Wait for both
wait $PID_API $PID_UI
