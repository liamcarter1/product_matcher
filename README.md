---
title: ProductMatchPro
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# ProductMatchPro - Hydraulic Product Cross-Reference

A RAG-powered application for cross-referencing competitor hydraulic products against your company's catalogue.

## Features

- **Distributor Product Finder** - Chat interface where distributors enter a competitor model code and get the best equivalent with a confidence score
- **Knowledge Base Q&A** - Distributors can ask general questions about products and get answers grounded in uploaded documentation
- **Admin Console** - Upload PDF catalogues and user guides, manage products, review feedback
- **Dual-Provider LLM** - Supports both Anthropic Claude and OpenAI, switchable via config flag (`LLM_PROVIDER`). Claude is the default provider with tiered models (Opus for critical extraction, Sonnet for standard tasks, Haiku for fast classification)
- **Specialized Agent Architecture** - 7 focused agent modules in `tools/agents/` each handling a specific extraction task with the right model tier
- **Comprehensive PDF Extraction** - PyMuPDF (fitz) primary extractor with pdfplumber supplementary pass captures ALL pages including complex layouts, operating data tables, and technical specifications. Full-document LLM processing in batches (no truncation)
- **Ordering Code Generation** - Automatically reads "How to Order" tables from datasheets and generates ALL product variants as separate database entries with fully populated specs
- **Spool Type Cross-Referencing** - Extracts spool/function designations with center condition descriptions from user guides and ordering codes, enabling cross-manufacturer matching (e.g. Danfoss "2A" ≈ Bosch Rexroth "D" — both P-to-A, B-to-T)
- **Smart Matching** - 12-dimension weighted scoring with fuzzy string tolerance (e.g. "24VDC" matches "24 VDC") and DB fallback when vector store is empty
- **Fuzzy Model Code Lookup** - Partial codes accepted (e.g. "4WE6" finds "4WE6D6X/EG24N9K4")
- **75% Confidence Threshold** - Below threshold directs distributors to contact their sales representative
- **Diagnostic No-Match Responses** - When no equivalent is found, explains why (no products uploaded, index empty, category mismatch) with actionable guidance

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your API key (required for PDF extraction and query parsing):
   ```bash
   # Anthropic (default provider)
   export ANTHROPIC_API_KEY=your_key_here
   export LLM_PROVIDER=anthropic

   # OR OpenAI (fallback provider)
   export OPENAI_API_KEY=your_key_here
   export LLM_PROVIDER=openai
   ```

3. (Optional) Set login credentials to protect the web interface:
   ```bash
   export ADMIN_USERNAME=admin
   export ADMIN_PASSWORD=your_password
   ```

4. Run the combined app:
   ```bash
   python app.py
   ```

   Or run each app separately:
   ```bash
   python distributor_app.py  # Port 7860
   python admin_app.py        # Port 7861
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required if anthropic) | Anthropic API key for Claude models |
| `OPENAI_API_KEY` | (required if openai) | OpenAI API key for GPT models |
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `ADMIN_USERNAME` | (none) | Login username (enables Gradio auth when set) |
| `ADMIN_PASSWORD` | (none) | Login password |
| `PORT` | `7860` | Port for combined app.py (auto-set by HF Spaces) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |

See `CLAUDE.md` for the full list of host/port configuration variables.

## Deploying to Hugging Face Spaces

1. Create a new Space with **Gradio** SDK
2. Push the `product_matcher/` directory as the repo contents
3. Set **Space Secrets**: `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`), `LLM_PROVIDER`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`
4. The app binds to `0.0.0.0` and reads the `PORT` env var automatically
5. Both the Product Finder and Admin Console are available in a single tabbed interface

## Architecture

- **Dual-provider LLM system** (`tools/llm_client.py`) — tier-based model selection (HIGH/MID/LOW) mapped to Claude Opus/Sonnet/Haiku or GPT-4.1/4.1-mini/4o-mini
- **7 specialized agent modules** (`tools/agents/`) — each extraction task gets its own focused agent with the right model tier
- **Multi-extractor PDF pipeline** - PyMuPDF (fitz) for primary text, pdfplumber for tables and supplementary text, batched agent extraction covering ALL pages (not truncated)
- **Two-pass guide indexing** - Page-level chunks (with page number metadata) + full-document chunks for maximum retrieval coverage
- **Ordering code combinatorial engine** - Parses "How to Order" tables via ordering code agent, generates all product variants via `itertools.product()` (capped at 500)
- **ChatAgent class** — simple state machine for distributor chat (replaces LangGraph)
- **Fuzzy spec matching** with normalisation tolerance and DB fallback when the vector index is empty
- **Numpy-based vector store** with sentence-transformers embeddings + cross-encoder reranking
- **SQLite** for structured product data, model code patterns, confirmed equivalents, and feedback
- **Gradio** for both the distributor and admin interfaces

## Docker Deployment

A `Dockerfile` and `docker-compose.yml` are included for VPS deployment:

```bash
docker compose up -d --build
```

The app runs on port 7860 behind Nginx. See `deploy/nginx-match.conf` for the reverse proxy config with WebSocket support (required for Gradio).

## Testing

Run the test suite (320 tests covering ingestion, parsing, storage, and agents):

```bash
python -m pytest tests/ -v
```

## Security

- Optional Gradio login authentication (username/password via env vars)
- Server-side PDF validation (magic bytes, size limits)
- Input length limits on all user-facing fields
- Sanitised error messages (no file paths or API keys exposed)
- Thread-safe database writes with locking
- Atomic file writes for vector store persistence
- LLM call resilience with regex-based fallback parsing and exponential backoff retry
