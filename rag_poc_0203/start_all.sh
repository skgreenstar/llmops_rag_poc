#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting AI Service Stack..."

# ==========================================
# 0. Dependency Check & Setup
# ==========================================

# 0.1 Ensure dify/.env exists (Required for docker compose)
if [ ! -f "dify/.env" ]; then
    echo "⚠️ dify/.env not found."
    if [ -f "dify/.env.example" ]; then
        echo "   Creating dify/.env from example..."
        cp dify/.env.example dify/.env
    else
        # Fallback to creating a minimal .env if example is missing (unlikely but safe)
        echo "⚠️ dify/.env.example not found. Creating minimal .env..."
        cat <<EOF > dify/.env
LOG_LEVEL=INFO
DEBUG=false
DIFY_BIND_ADDRESS=0.0.0.0
DIFY_PORT=5001
DB_USERNAME=postgres
DB_PASSWORD=difyai123456
DB_HOST=db_postgres
DB_PORT=5432
DB_DATABASE=dify
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=difyai123456
EOF
    fi
fi

# 0.2 Check for Poison (Poetry)
echo "🔍 Checking for Poetry..."
POETRY_CMD=""

if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
elif [ -f "$HOME/.local/bin/poetry" ]; then
    POETRY_CMD="$HOME/.local/bin/poetry"
    export PATH="$HOME/.local/bin:$PATH"
elif [ -f "/opt/homebrew/bin/poetry" ]; then
    POETRY_CMD="/opt/homebrew/bin/poetry"
    export PATH="/opt/homebrew/bin/poetry:$PATH"
elif [ -f "/usr/local/bin/poetry" ]; then
    POETRY_CMD="/usr/local/bin/poetry"
else
    echo "⚠️ Poetry not found in standard locations. Attempting to install..."
    if command -v python3 &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        POETRY_CMD="$HOME/.local/bin/poetry"
    else
        echo "❌ Error: Python3 is not installed. Cannot install Poetry."
        exit 1
    fi
fi

# Verify Poetry is usable
if ! $POETRY_CMD --version &> /dev/null; then
    echo "❌ Error: Poetry is still not found or not executable at $POETRY_CMD"
    # Last ditch attempt to source profile if on Linux
    if [ -f "$HOME/.profile" ]; then
        source "$HOME/.profile"
    fi
    if ! command -v poetry &> /dev/null; then
         echo "   Please install Poetry manually: curl -sSL https://install.python-poetry.org | python3 -"
         exit 1
    else
         POETRY_CMD="poetry"
    fi
fi
echo "✅ Poetry found: $($POETRY_CMD --version)"


# ==========================================
# 1. Docker Services
# ==========================================
echo "1. Ensuring All Docker services (Apps + Dify) are up..."
# Check if docker compose command exists (v2) or docker-compose (v1)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "❌ Error: Docker Compose not found. Please install Docker."
    exit 1
fi

$DOCKER_COMPOSE_CMD up -d --remove-orphans


# ==========================================
# 2. Python Services
# ==========================================

# Install dependencies if needed (Optional but good for first run)
# We check if 'poetry install' has run by checking for valid env, but simplest is just to run install (it's fast if done)
echo "📦 Checking/Installing Python dependencies..."
$POETRY_CMD install --no-root

# 2. FastAPI Backend
echo "2. Starting Backend API (Port 8000)..."
$POETRY_CMD run uvicorn app.api.server:app --reload --port 8000 &
PID_API=$!

# 3. Streamlit UI
echo "3. Starting Operator UI (Port 8501)..."
$POETRY_CMD run streamlit run app/ui/streamlit_app.py &
PID_UI=$!

echo "✅ All services started!"
echo "   - API: http://localhost:8000"
echo "   - UI:  http://localhost:8501"
echo "   - Langfuse: http://localhost:3000"
echo "   - Dify: http://localhost:8081 (Admin Setup Required)"
echo "Press Ctrl+C to stop all."

# Wait for both
wait $PID_API $PID_UI
