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
- **Ordering Code Generation** - Automatically reads "How to Order" tables from datasheets and generates ALL product variants as separate database entries with fully populated specs
- **Smart Matching** - 12-dimension weighted scoring with fuzzy string tolerance (e.g. "24VDC" matches "24 VDC") and DB fallback when vector store is empty
- **Fuzzy Model Code Lookup** - Partial codes accepted (e.g. "4WE6" finds "4WE6D6X/EG24N9K4")
- **75% Confidence Threshold** - Below threshold directs distributors to contact their sales representative
- **Diagnostic No-Match Responses** - When no equivalent is found, explains why (no products uploaded, index empty, category mismatch) with actionable guidance

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key (required for PDF extraction and query parsing):
   ```bash
   export OPENAI_API_KEY=your_key_here
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
| `OPENAI_API_KEY` | (required) | OpenAI API key for GPT-4o-mini |
| `ADMIN_USERNAME` | (none) | Login username (enables Gradio auth when set) |
| `ADMIN_PASSWORD` | (none) | Login password |
| `PORT` | `7860` | Port for combined app.py (auto-set by HF Spaces) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |

See `CLAUDE.md` for the full list of host/port configuration variables.

## Deploying to Hugging Face Spaces

1. Create a new Space with **Gradio** SDK
2. Push the `product_matcher/` directory as the repo contents
3. Set **Space Secrets**: `OPENAI_API_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`
4. The app binds to `0.0.0.0` and reads the `PORT` env var automatically
5. Both the Product Finder and Admin Console are available in a single tabbed interface

## Architecture

- **LangGraph** StateGraph (7 nodes) with MemorySaver for conversation persistence
- **Ordering code combinatorial engine** - Parses "How to Order" tables via GPT-4o-mini, generates all product variants via `itertools.product()` (capped at 500)
- **Fuzzy spec matching** with normalisation tolerance and DB fallback when the vector index is empty
- **Numpy-based vector store** with sentence-transformers embeddings + cross-encoder reranking
- **SQLite** for structured product data, model code patterns, confirmed equivalents, and feedback
- **Gradio** for both the distributor and admin interfaces

## Security

- Optional Gradio login authentication (username/password via env vars)
- Server-side PDF validation (magic bytes, size limits)
- Input length limits on all user-facing fields
- Sanitised error messages (no file paths or API keys exposed)
- Thread-safe database writes with locking
- Atomic file writes for vector store persistence
- LLM call resilience with regex-based fallback parsing
