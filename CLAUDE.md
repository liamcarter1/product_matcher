# CLAUDE.md - ProductMatchPro

## Project Overview

ProductMatchPro is a RAG-powered hydraulic product cross-reference application. Distributors type a competitor model code (partial allowed) and get the best equivalent product from {my_company} with a confidence score. If confidence is below 75%, they are directed to contact their sales representative.

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
PDF Upload (admin) -> Parse & Extract -> Review -> Store (SQLite + Vector Store)
                                                          |
Distributor Query -> Fuzzy Code Lookup -> Identify Competitor Product
                                                          |
                                    Semantic Search + Spec Comparison
                                                          |
                                    Confidence-Ranked Results -> Chat Response
```

### LangGraph Workflow (graph.py)
5 nodes in a StateGraph with MemorySaver:
```
START -> parse_query -> lookup_competitor -> [routing]
  routing:
    "confirmed"  -> generate_response -> END  (skip matching, use manual override)
    "found"      -> find_equivalents -> generate_response -> END
    "ambiguous"  -> clarify -> END  (ask user to pick)
    "not_found"  -> generate_response -> END  (show "not found" message)
```

LLM: `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`

### Storage

**SQLite** (`storage/product_db.py`) - 5 tables:
- `products` - 34 spec columns + metadata, indexed on model_code/company/category/coil_voltage/valve_size
- `model_code_patterns` - Decode rules for model code segments (extracted from user guides)
- `confirmed_equivalents` - Manual overrides from admin (bypasses algorithmic matching)
- `feedback` - Distributor thumbs up/down on results
- `synonyms` - Brand name mappings (e.g. "Vickers" -> "Eaton")

**Numpy Vector Store** (`storage/vector_store.py`) - 3 collections:
- `my_company_products` - Searched with cross-encoder reranking
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
  valve_size_match:     0.10  (exact match)
  coil_voltage_match:   0.10  (exact match - 24VDC != 110VAC)
  actuator_type_match:  0.08  (exact match)
  spool_function_match: 0.08  (exact match)
  mounting_match:       0.08  (exact, falls back to mounting_pattern)
  port_match:           0.06  (exact match)
  seal_material_match:  0.03  (exact match)
  temp_range_match:     0.02  (range coverage)
  semantic_similarity:  0.15  (from vector search + reranking)
```

Missing specs score 0.5 (neutral), not 0.0.

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

### Configurable Values (`graph.py`)
- `MY_COMPANY_NAME = "my_company"` - Replace with actual company name
- `SALES_CONTACT = "sales@my_company.com | +44 (0)XXX XXX XXXX"` - Update with real contact

## File Structure

```
product_matcher/
  app.py                 (28 lines)   Combined HF Spaces entry point
  distributor_app.py    (235 lines)   Distributor chat UI
  admin_app.py          (421 lines)   Admin console UI
  graph.py              (473 lines)   LangGraph matching workflow
  ingest.py             (220 lines)   PDF ingestion pipeline
  models.py             (211 lines)   Pydantic models & constants
  prompts.py             (69 lines)   LLM system prompts
  requirements.txt       (32 lines)   Dependencies
  .env                               OPENAI_API_KEY (gitignored)
  storage/
    product_db.py       (604 lines)   SQLite CRUD + fuzzy lookup + spec comparison
    vector_store.py     (355 lines)   Numpy vector store + reranking
  tools/
    lookup_tools.py     (237 lines)   Product identification & matching
    parse_tools.py      (279 lines)   PDF table/text extraction + LLM extraction
  data/                              Gitignored runtime data (.db, .npz, .json)
```

## Known Issues & Gotchas

### fuzzywuzzy Tuple Format
`process.extract()` returns `(match, score)` tuples (2-element), NOT `(match, score, index)`. Always use `match_tuple[0]`, `match_tuple[1]` - never 3-element destructuring.

### Gradio Version
Code is compatible with Gradio 5 and 6. Key changes in Gradio 6:
- `theme` and `css` moved from `gr.Blocks()` to `.launch()`
- `gr.Chatbot(type="messages")` removed - messages format is default
- `gr.Chatbot(show_copy_button=True)` removed
- `gr.update()` removed - return component instances directly (e.g. `gr.Dropdown(choices=[...])`)
- `gr.TabbedInterface` doesn't accept `gr.Blocks` - use `gr.Blocks` + `gr.Tabs` + `.render()`

### numpy argpartition
`np.argpartition(-scores, k)` fails when `k >= len(scores)`. Always check bounds first, fallback to `np.argsort`.

### OpenAI Client
Don't initialize at module level. Use lazy `_get_client()` pattern (`tools/parse_tools.py:24-28`) to avoid crashes when API key isn't set at import time.

### Python 3.14
ChromaDB is incompatible (pydantic v1). The vector store uses numpy instead. The `langchain_core` pydantic v1 deprecation warning is expected and harmless.

### Admin State
`pending_extractions` and `pending_metadata` in `admin_app.py` are module-level globals - not thread-safe for concurrent admin users. Fine for single-user admin use.

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
1. pdfplumber extracts tables -> maps headers to product fields
2. pypdf extracts text -> GPT-4o-mini structures products (fallback)
3. For user guides: GPT-4o-mini extracts model code decode patterns
4. Admin reviews extracted products, then confirms to index

### Modifying the Matching Pipeline
- Scoring weights: `SCORE_WEIGHTS` in `models.py`
- Confidence threshold: `CONFIDENCE_THRESHOLD` in `models.py`
- Comparison logic: `spec_comparison()` in `storage/product_db.py`
- Graph routing thresholds: `_route_after_lookup()` in `graph.py`
- Fuzzy match thresholds: `identify_competitor_product()` in `tools/lookup_tools.py`

## Dependencies

Core: openai, langchain-openai, langgraph, sentence-transformers, numpy, pydantic, gradio
PDF: pypdf, pdfplumber
Matching: fuzzywuzzy, python-Levenshtein
Data: pandas (admin CSV export)
Config: python-dotenv

## Environment Variables

- `OPENAI_API_KEY` (required) - Used by GPT-4o-mini for query parsing, product extraction, and comparison narratives
