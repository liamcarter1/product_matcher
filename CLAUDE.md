# CLAUDE.md - ProductMatchPro

## Project Overview

ProductMatchPro is a RAG-powered hydraulic product cross-reference application. Distributors type a competitor model code (partial allowed) and get the best equivalent Danfoss product with a confidence score. If confidence is below 75%, they are directed to contact their sales representative.

Two separate Gradio interfaces:
- **Distributor app** (`distributor_app.py`, port 7860) - Clean chat interface for product lookup
- **Admin console** (`admin_app.py`, port 7861) - PDF upload, product management, feedback review
- **Combined entry** (`app.py`, port 7860) - Both apps in a tabbed interface (for HF Spaces)

## Quick Start

```bash
cd product_matcher
pip install -r requirements.txt
# Set OPENAI_API_KEY in .env file
python distributor_app.py   # or python app.py for combined
```

## Architecture

### Data Flow
```
PDF Upload (admin) -> Parse & Extract -> Ordering Code Generation -> Dedup -> Review -> Store
                              |                    |
                     Table extraction       Combinatorial generation
                     LLM text extraction    from ordering code tables
                     Header mapping         (all segment permutations)
                                                          |
                                            SQLite + Vector Store
                                                          |
Distributor Query -> Fuzzy Code Lookup -> Identify Competitor Product
                                                          |
                                    Semantic Search + Spec Comparison
                                                          |
                                    Confidence-Ranked Results -> Chat Response
```

### LangGraph Workflow (graph.py)
7 nodes in a StateGraph with MemorySaver:
```
START -> parse_query -> [routing by intent]
  intent="info"      -> retrieve_kb_context -> generate_kb_answer -> END  (KB Q&A path)
  intent="match"     -> lookup_competitor -> [routing by lookup result]
    "confirmed"      -> generate_response -> END  (manual override)
    "found"          -> find_equivalents -> generate_response -> END
    "ambiguous"      -> clarify -> END  (ask user to pick)
    "not_found"      -> generate_response -> END  (show "not found" message)
```

LLM: `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`

### Storage

**SQLite** (`storage/product_db.py`) - 5 tables:
- `products` - 34 spec columns + metadata, indexed on model_code/company/category/coil_voltage/valve_size
- `model_code_patterns` - Decode rules for model code segments (extracted from user guides)
- `confirmed_equivalents` - Manual overrides from admin (bypasses algorithmic matching)
- `feedback` - Distributor thumbs up/down on results
- `synonyms` - Brand name mappings (e.g. "Rexroth" -> "Bosch Rexroth")

**Numpy Vector Store** (`storage/vector_store.py`) - 3 collections:
- `danfoss_products` - Searched with cross-encoder reranking
- `competitor_products` - Semantic fallback when fuzzy lookup fails
- `product_guides` - User guide text chunks

Persistence: `{name}_index.json` + `{name}_vectors.npz` in `data/` directory.

No ChromaDB - incompatible with Python 3.14 (pydantic v1 dependency).

## Key Constants & Config

### Confidence Scoring (`models.py`)
```
CONFIDENCE_THRESHOLD = 0.75  (below this -> "contact sales rep")

SCORE_WEIGHTS (total = 1.0):
  category_match:       0.10  (gate: if 0.0, total capped at 0.3)
  pressure_match:       0.10  (numerical closeness)
  flow_match:           0.10  (numerical closeness)
  valve_size_match:     0.10  (fuzzy string match)
  coil_voltage_match:   0.10  (fuzzy string match - "24VDC" ≈ "24 VDC")
  actuator_type_match:  0.08  (fuzzy string match)
  spool_function_match: 0.08  (fuzzy string match)
  mounting_match:       0.08  (fuzzy, falls back to mounting_pattern)
  port_match:           0.06  (fuzzy string match)
  seal_material_match:  0.03  (fuzzy string match)
  temp_range_match:     0.02  (range coverage)
  semantic_similarity:  0.15  (from vector search + reranking)
```

Missing specs score 0.5 (neutral), not 0.0.

### String Matching (`product_db.py:_exact_match`)
String comparisons use fuzzy tolerance, not strict equality:
- **1.0**: exact match (case-insensitive)
- **0.95**: normalised match after stripping non-alphanumeric chars (e.g. "24VDC" vs "24 VDC", "proportional_solenoid" vs "proportional solenoid")
- **0.75**: containment (one value inside the other, e.g. "solenoid" in "solenoid operated")
- **0.8+**: fuzzywuzzy token_sort_ratio for partial similarity
- **0.5**: one side is None/empty (unknown)
- **0.0**: completely different strings (e.g. "FKM" vs "NBR")

### Equivalent Search Fallback (`lookup_tools.py:find_my_company_equivalents`)
If the vector store has no indexed Danfoss products, the search falls back to pulling candidates directly from the SQLite database (by category first, then all Danfoss products) and running spec comparison without a semantic score. This prevents silent empty results when vector indexing hasn't happened.

### Fuzzy Matching Thresholds (`tools/lookup_tools.py`, `storage/product_db.py`)
- General lookup: threshold 60%
- Confirmed equivalent check: threshold 80%
- Single strong match: score > 85%
- Ambiguous detection: top two scores within 10% of each other

### ML Models (`storage/vector_store.py`)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Text Chunking (`ingest.py`)
- RecursiveCharacterTextSplitter: chunk_size=1000, overlap=200

### Numeric Parsing (`parse_tools.py`)
`_parse_numeric_if_possible()` only converts values that ARE pure numbers (or number+unit like "315 bar"). Preserves mixed alpha-numeric values: "24VDC", "G3/8", "FKM", "ISO 4401-03" are all returned as strings, not corrupted.

### Product Categories (`models.py:ProductCategory`)
```
directional_valves, proportional_directional_valves, pressure_valves, flow_valves,
pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
```
Categories must be consistent across: `models.py` enum, `admin_app.py` CATEGORIES, `distributor_app.py` CATEGORIES + CATEGORY_MAP, `graph.py` regex fallback hints, `parse_tools.py` LLM prompts, `prompts.py` query parser prompt.

### Configurable Values (`graph.py`)
- `MY_COMPANY_NAME = "Danfoss"` - Company name used in prompts and responses
- `SALES_CONTACT = "sales@danfoss.com | +44 (0)XXX XXX XXXX"` - Update with real contact number

## File Structure

```
product_matcher/
  app.py                 (~40 lines)  Combined HF Spaces entry point (auth-enabled)
  distributor_app.py    (~270 lines)  Distributor chat UI (Gradio 5/6 compat, dynamic dropdowns)
  admin_app.py          (~605 lines)  Admin console UI (interactive tables, auth, gr.State)
  graph.py              (~650 lines)  LangGraph matching workflow (7 nodes, diagnostics)
  ingest.py             (~240 lines)  PDF ingestion pipeline + ordering code generation
  models.py             (~240 lines)  Pydantic models & constants (incl. OrderingCode*)
  prompts.py            (~100 lines)  LLM system prompts (incl. KB_QA_PROMPT)
  requirements.txt       (32 lines)   Dependencies
  .env                               OPENAI_API_KEY + auth credentials (gitignored)
  storage/
    product_db.py       (~630 lines)  SQLite CRUD + fuzzy lookup + fuzzy string matching
    vector_store.py     (~400 lines)  Numpy vector store + reranking (atomic saves)
  tools/
    lookup_tools.py     (~250 lines)  Product identification & matching (with DB fallback)
    parse_tools.py      (~520 lines)  PDF extraction + ordering code combinatorial generation
  data/                              Gitignored runtime data (.db, .npz, .json)
```

## Known Issues & Gotchas

### fuzzywuzzy Tuple Format
`process.extract()` returns `(match, score)` tuples (2-element), NOT `(match, score, index)`. Always use `match_tuple[0]`, `match_tuple[1]` - never 3-element destructuring.

### Gradio Version
Code is compatible with Gradio 5 and 6. `_chatbot_kwargs()` in `distributor_app.py` auto-detects the version and adds `type="messages"` for Gradio 5 (where the default is tuple format). Key changes in Gradio 6:
- `theme` and `css` moved from `gr.Blocks()` to `.launch()`
- `gr.Chatbot(type="messages")` removed - messages format is default
- `gr.Chatbot(show_copy_button=True)` removed
- `gr.update()` removed - return component instances directly (e.g. `gr.Dropdown(choices=[...])`)
- `gr.TabbedInterface` doesn't accept `gr.Blocks` - use `gr.Blocks` + `gr.Tabs` + `.render()`

**HF Spaces**: `sdk_version` in README frontmatter must be `6.x` to match the code. Gradio 5 chatbots default to tuple format and will error when receiving message dicts.

### Distributor App Dynamic Dropdowns
The competitor dropdown in `distributor_app.py` uses `allow_custom_value=True` so distributors can type any competitor name. The choices refresh from the database on every focus event via `refresh_competitor_dropdown()`, so newly uploaded companies appear without restarting.

### Admin Interactive Tables
All `gr.Dataframe` components in `admin_app.py` are `interactive=True` so cell text can be selected and copied. The data is effectively read-only since handlers always return fresh data on search/refresh.

### numpy argpartition
`np.argpartition(-scores, k)` fails when `k >= len(scores)`. Always check bounds first, fallback to `np.argsort`.

### OpenAI Client
Don't initialize at module level. Use lazy `_get_client()` pattern (`tools/parse_tools.py:24-28`) to avoid crashes when API key isn't set at import time.

### Python 3.14
ChromaDB is incompatible (pydantic v1). The vector store uses numpy instead. The `langchain_core` pydantic v1 deprecation warning is expected and harmless.

### Admin State
`pending_extractions` and `pending_metadata` in `admin_app.py` use `gr.State({})` — session-scoped and safe for concurrent admin users.

## Development Patterns

### Adding a New Spec Field
1. Add field to `HydraulicProduct` in `models.py`
2. Add to `SPEC_FIELDS` list (and `NUMERICAL_FIELDS` or `EXACT_MATCH_FIELDS`)
3. Add column to `products` table in `product_db.py:_create_tables()`
4. Add to `insert_product()` SQL and parameter list
5. Add to `_build_indexable_text()` in `vector_store.py` (for embedding)
6. If it needs its own score weight: add to `ScoreBreakdown`, `SCORE_WEIGHTS`, and `spec_comparison()`

### Adding a New Competitor
Upload their catalogue/user guide PDF via the admin console. The pipeline:
1. pdfplumber extracts tables -> maps ~90 header patterns to product fields
2. pypdf extracts text -> GPT-4o-mini structures products (31 spec fields, fallback)
3. GPT-4o-mini extracts ordering code breakdown tables -> combinatorial generator creates all product variants (capped at 500)
4. For user guides/datasheets: GPT-4o-mini also extracts model code decode patterns
5. Deduplication merges products from all sources (keeps richest specs)
6. Admin reviews extracted products (with Source column: table/llm/ordering_code), then confirms to index

### Ordering Code Combinatorial Generation (`parse_tools.py`)
When a PDF contains "Ordering code" / "How to Order" tables:
- `extract_ordering_code_with_llm()` identifies segment positions, fixed/variable flags, separators, and field mappings
- `generate_products_from_ordering_code()` creates all permutations via `itertools.product()`
- `assemble_model_code()` reconstructs model codes from a template like `{01}{02}{03}-{04}/{05}`
- Empty "no code" options handled (double separators cleaned up)
- `MAX_COMBINATIONS = 500` safety cap
- Data models: `OrderingCodeSegment`, `OrderingCodeDefinition` in `models.py`

### Field Alias Rescue (`ingest.py`)
`_FIELD_ALIASES` maps ~60 common LLM/table output names to canonical HydraulicProduct fields (e.g. "pressure" -> "max_pressure_bar", "voltage" -> "coil_voltage"). Applied before product construction to catch non-standard field names.

### No-Match Diagnostics (`graph.py:generate_response`)
When `find_equivalents` returns no matches, the response includes diagnostic information:
- How many Danfoss products are in the database
- How many are indexed in the vector store
- The competitor product's category (so the admin knows what to upload)
- Actionable suggestions (upload Danfoss products, re-index, check category coverage)

The diagnostics are gathered in `find_equivalents` and passed via `_diagnostics` in the state dict.

### Adding a New Product Category
Must be added in **6 places**:
1. `models.py` — `ProductCategory` enum
2. `admin_app.py` — `CATEGORIES` list
3. `distributor_app.py` — `CATEGORIES` list + `CATEGORY_MAP` dict
4. `graph.py` — `category_hints` dict in `_regex_parse_fallback()`
5. `tools/parse_tools.py` — LLM prompt category lists (both extraction and ordering code prompts)
6. `prompts.py` — `QUERY_PARSER_PROMPT` category list

### Modifying the Matching Pipeline
- Scoring weights: `SCORE_WEIGHTS` in `models.py`
- Confidence threshold: `CONFIDENCE_THRESHOLD` in `models.py`
- Comparison logic: `spec_comparison()` in `storage/product_db.py`
- String matching tolerance: `_exact_match()` in `storage/product_db.py`
- Graph routing thresholds: `_route_after_lookup()` in `graph.py`
- Fuzzy match thresholds: `identify_competitor_product()` in `tools/lookup_tools.py`
- DB fallback: `find_my_company_equivalents()` in `tools/lookup_tools.py`

## Dependencies

Core: openai, langchain-openai, langgraph, sentence-transformers, numpy, pydantic, gradio
PDF: pypdf, pdfplumber
Matching: fuzzywuzzy, python-Levenshtein
Data: pandas (admin CSV export)
Config: python-dotenv

## Security Hardening (v2)

The following security and resilience fixes have been applied:

### Authentication
- Admin and distributor apps support Gradio login via `ADMIN_USERNAME` / `ADMIN_PASSWORD` env vars
- Combined `app.py` uses the same credentials
- Authentication is optional in local dev but recommended for any network-facing deployment

### Session Isolation
- `pending_extractions` and `pending_metadata` in `admin_app.py` now use `gr.State({})` instead of module-level globals — safe for concurrent admin users

### Input Validation & Sanitisation
- Server-side PDF validation: checks magic bytes (`%PDF`), file size limits, and empty files
- Input length limits on all user-facing text fields (500 chars for chat, 200 for model codes/search)
- Error messages are sanitised — no file paths, API keys, or stack traces leak to the UI

### Thread Safety
- `threading.Lock` on all write operations in `product_db.py` and `vector_store.py`
- Atomic file writes in vector store (temp file + rename pattern) to prevent corruption on crash

### LLM Resilience
- `graph.py:parse_query()` wraps the LLM call in try/except with a regex-based fallback parser
- If OpenAI is unreachable, the regex fallback extracts model codes, competitor names, and categories from user input
- Cross-encoder reranking scores are clamped to `[0.0, 1.0]` to prevent Pydantic validation errors

### Configurable Binding
- All apps use env vars for host/port — critical for HF Spaces (`0.0.0.0` binding required)
- `max_file_size` set on Gradio launch to enforce upload limits server-side

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Used by GPT-4o-mini for query parsing, product extraction, and comparison narratives |
| `ADMIN_USERNAME` | (none) | Login username for Gradio auth (optional, but recommended for deployed apps) |
| `ADMIN_PASSWORD` | (none) | Login password for Gradio auth (optional, but recommended for deployed apps) |
| `ADMIN_HOST` | `127.0.0.1` | Bind address for admin_app.py |
| `ADMIN_PORT` | `7861` | Port for admin_app.py |
| `DISTRIBUTOR_HOST` | `0.0.0.0` | Bind address for distributor_app.py |
| `DISTRIBUTOR_PORT` | `7860` | Port for distributor_app.py |
| `PORT` | `7860` | Port for combined app.py (HF Spaces sets this automatically) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size in megabytes |
