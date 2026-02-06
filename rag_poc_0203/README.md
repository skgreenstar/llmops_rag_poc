# AI Service Architecture Scaffolding

This is a scaffold for an Observability-First AI Service.

## Documentation
- **[Operator Manual (운영자 매뉴얼)](OPERATOR_MANUAL.md)**: Detailed guide for operations, maintenance, and scripts.

## Architecture
- **App Layer**: FastAPI
- **Agent Layer**: LangGraph / LangChain
- **RAG Layer**: Plug-and-play Retriever
- **Model Layer**: Router for Local/Cloud LLMs
- **Ops Layer**: Langfuse Tracing (Core)

## Quick Setup
1. Install dependencies:
   ```bash
   poetry install
   ```
2. Setup Env:
   ```bash
   cp .env.example .env
   ```
3. Run All Services:
   ```bash
   ./start_all.sh
   ```

For more details, please refer to the `OPERATOR_MANUAL.md`.
