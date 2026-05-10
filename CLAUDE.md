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
# Set ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY) in .env file
# Set LLM_PROVIDER=anthropic (default) or LLM_PROVIDER=openai in .env
python distributor_app.py   # or python app.py for combined
```

## Architecture

### LLM Provider System (`tools/llm_client.py`)

Dual-provider architecture — switch between Anthropic Claude and OpenAI via `LLM_PROVIDER` env var:

| Tier | Anthropic (default) | OpenAI (fallback) | Used For |
|------|--------------------|--------------------|----------|
| `TIER_HIGH` | claude-opus-4-20250514 | gpt-4.1 | Ordering code extraction, vision spool analysis |
| `TIER_MID` | claude-sonnet-4-20250514 | gpt-4.1-mini | Product extraction, spec extraction, chat responses |
| `TIER_LOW` | claude-haiku-4-20250514 | gpt-4o-mini | Query parsing, classification |

Key functions: `call_llm()`, `call_llm_json()`, `call_llm_tool()`, `get_client()`

### Data Flow
```
PDF Upload (admin) -> Parse & Extract -> Ordering Code Generation -> Dedup -> Review -> Store
                              |                    |
                     Table extraction       Combinatorial generation
                     Agent-based extraction from ordering code tables
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

### Specialized Agent Modules (`tools/agents/`)

Each extraction task has its own agent module with focused prompts and the right model tier:

| Agent | File | Tier | Replaces |
|-------|------|------|----------|
| Product extractor | `product_extractor.py` | MID | `extract_products_with_llm()` |
| Spec extractor | `spec_extractor.py` | MID | `extract_model_code_patterns_with_llm()` |
| Ordering code (text+vision) | `ordering_code.py` | HIGH | `extract_ordering_code_with_llm()` + `from_images()` |
| Spool analyzer (text+vision) | `spool_analyzer.py` | HIGH/MID | `analyze_spool_functions()` + vision spool |
| Cross-reference | `cross_reference.py` | MID | `extract_cross_references_with_llm()` |
| Image reader | `image_reader.py` | MID | `extract_text_from_image()` |
| Chat agent | `chat_agent.py` | LOW/MID | LangGraph `graph.py` |

All agents call through `tools/llm_client.py` — never import openai/anthropic directly.

### Chat Agent (`tools/agents/chat_agent.py`)
Simple class-based state machine replacing the LangGraph StateGraph:
```
ChatAgent.search_sync(message, thread_id) -> str

  _parse_query(message)           # TIER_LOW — intent classification
    intent="info"  -> _handle_kb_query()         # TIER_MID — KB Q&A
    intent="match" -> _handle_product_matching()  # TIER_MID — matching + response
```

Reuses all prompts from `prompts.py` and `LookupTools` for DB/vector operations.

### Storage

**SQLite** (`storage/product_db.py`) - 7 tables:
- `products` - 34 spec columns + metadata, indexed on model_code/company/category/coil_voltage/valve_size
- `model_code_patterns` - Decode rules for model code segments (extracted from user guides)
- `confirmed_equivalents` - Manual overrides from admin (bypasses algorithmic matching)
- `feedback` - Distributor thumbs up/down on results
- `synonyms` - Brand name mappings (e.g. "Rexroth" -> "Bosch Rexroth")
- `spool_type_reference` - Per-manufacturer spool dictionary (manufacturer + series_prefix + spool_code → description, center_condition, canonical_pattern). Auto-loaded from `data/spool_seed.json` on every startup (idempotent — `source = 'seed'` rows are skipped if already present unless `force=True`). Drives cross-manufacturer translation in `enrich_competitor_spool()`. See "Spool Matching" section below.
- `series_cross_reference` - Series-level mappings extracted from cross-reference tables in PDFs (competitor_series → my_company_series); used as a hint for vector search

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
  coil_voltage_match:   0.10  (fuzzy string match - "24VDC" = "24 VDC")
  actuator_type_match:  0.08  (fuzzy string match)
  spool_function_match: 0.08  (CANONICAL-PATTERN equality if both sides have extra_specs["canonical_spool_pattern"]; otherwise falls back to fuzzy string match on raw spool_type code)
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

### PDF Text Extraction (`tools/parse_tools.py`)
Primary extractor: **PyMuPDF (fitz)** — significantly better at extracting text from complex layouts, multi-column pages, operating data tables, and technical specifications. Falls back to pypdf if PyMuPDF is not installed.

Supplementary extractor: **pdfplumber** — runs as a second pass to capture any table-adjacent text the primary extractor missed. If pdfplumber finds a page with 100+ more characters than the primary, the additional content is merged.

LLM product extraction now processes the **full document** in 10,000-char batches (previously truncated at 12,000 chars, missing ~55% of a 19-page guide). Results are merged and deduplicated by model code.

### Text Chunking (`ingest.py`)
- Custom `_split_text_for_vectorstore()` using `chunk_text()` from `tools/agents/base.py`: chunk_size=1500, overlap=300

### Guide Indexing Strategy (`ingest.py:index_guide_text`)
Two-pass indexing for maximum retrieval coverage:
1. **Page-level chunks**: Each page is chunked individually with `[Page N of filename.pdf]` prefixes, preserving page context. This makes page-specific content (e.g. "Operating data" on page 6) directly searchable.
2. **Full-document chunks**: The entire document is chunked as one continuous text to capture cross-page information.

### KB Q&A Retrieval (`tools/agents/chat_agent.py:_handle_kb_query`)
- Retrieves 15 results (not 8) for better coverage
- Score threshold: 0.15 (not 0.3) — technical content embeddings often have modest similarity scores
- Multi-pass filter relaxation: tries with filters first, then progressively removes company/model_code filters if no results found

### Numeric Parsing (`parse_tools.py`)
`_parse_numeric_if_possible()` only converts values that ARE pure numbers (or number+unit like "315 bar"). Preserves mixed alpha-numeric values: "24VDC", "G3/8", "FKM", "ISO 4401-03" are all returned as strings, not corrupted.

### Product Categories (`models.py:ProductCategory`)
```
directional_valves, proportional_directional_valves, pressure_valves, flow_valves,
pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
```
Categories must be consistent across: `models.py` enum, `admin_app.py` CATEGORIES, `distributor_app.py` CATEGORIES + CATEGORY_MAP, `tools/agents/chat_agent.py` regex fallback hints, `tools/parse_tools.py` LLM prompts, `prompts.py` query parser prompt.

### Configurable Values (`tools/agents/chat_agent.py`)
- `MY_COMPANY_NAME = "Danfoss"` - Company name used in prompts and responses
- `SALES_CONTACT = "sales@danfoss.com | +44 (0)XXX XXX XXXX"` - Update with real contact number

## File Structure

```
product_matcher/
  app.py                 (~40 lines)  Combined HF Spaces entry point (auth-enabled)
  distributor_app.py    (~270 lines)  Distributor chat UI (Gradio 5/6 compat, dynamic dropdowns)
  admin_app.py          (~605 lines)  Admin console UI (interactive tables, auth, gr.State)
  ingest.py             (~280 lines)  PDF ingestion pipeline (two-pass guide indexing) + ordering code generation
  models.py             (~240 lines)  Pydantic models & constants (incl. OrderingCode*)
  prompts.py            (~100 lines)  LLM system prompts (incl. KB_QA_PROMPT)
  requirements.txt       (32 lines)   Dependencies
  .env                               API keys + auth credentials (gitignored)
  storage/
    product_db.py       (~630 lines)  SQLite CRUD + fuzzy lookup + fuzzy string matching
    vector_store.py     (~400 lines)  Numpy vector store + reranking (atomic saves)
  tools/
    llm_client.py       (~290 lines)  Dual-provider LLM client (Anthropic/OpenAI, tier-based)
    lookup_tools.py     (~250 lines)  Product identification & matching (with DB fallback)
    parse_tools.py      (~620 lines)  PDF extraction (PyMuPDF + pdfplumber + pypdf fallback) + ordering code combinatorial generation (non-LLM functions only)
    agents/
      __init__.py                     Package init
      base.py           (~155 lines)  Shared utilities (chunking, image encoding, vision content builders)
      product_extractor.py (~140 lines) Batched product extraction from text
      spec_extractor.py    (~80 lines)  Model code pattern extraction
      ordering_code.py    (~230 lines)  Ordering code extraction (text + vision) — MOST CRITICAL
      spool_analyzer.py   (~280 lines)  Spool function analysis (text + vision)
      cross_reference.py   (~80 lines)  Cross-reference table extraction
      image_reader.py     (~100 lines)  Camera/photo label reading
      chat_agent.py       (~380 lines)  Distributor chat (replaces LangGraph graph.py)
  data/
    spool_seed.json                  COMMITTED — canonical Danfoss + Bosch Rexroth 4WE6 spool dictionary (41 entries v1.2). Auto-loaded into spool_type_reference table on startup.
    *.db, *_index.json, *_vectors.npz Gitignored runtime data (SQLite + vector store).
  skills/
    hydraulics_engineer.md           COMMITTED — domain-knowledge skill file loaded into LLM agent prompts via tools/agents/base.py:get_skill_context().
```

## Spool Matching & Cross-Manufacturer Translation

The most consequential matching field is `spool_type` (weight 0.08, but it's the
field most likely to drop confidence below 0.75 when wrong). The same physical
spool function has different code letters per manufacturer (Rexroth `E` ≡ Danfoss
`2C` — both BLOCKED). Naive fuzzy string match on raw codes scores ≈ 0.0 on these.
The system solves this via a **canonical-pattern translation table**.

### Translation flow at query time

```
Distributor types e.g. "4WE6E62/EG24N9K4"
  ↓
identify_competitor_product()              # finds product or extracts from code
  ↓
enrich_competitor_spool()                  # looks up Bosch Rexroth/4WE6/E in
  (lookup_tools.py:120)                    # spool_type_reference table
  ↓                                        # → sets product.extra_specs
                                           #   ["canonical_spool_pattern"] =
                                           #   "BLOCKED|PA-BT|PB-AT"
find_my_company_equivalents()              # for each Danfoss candidate
  (lookup_tools.py:202)                    # whose pattern is also "BLOCKED|..."
                                           # spool_function_match = 1.0
  ↓
spec_comparison()                          # weighted sum + category gate
  (product_db.py:1057)
  ↓
≥ 0.75 confidence → return match
< 0.75 confidence → "contact sales rep"
```

### `spool_seed.json` schema (v1.2)

Each entry:
```json
{
  "series_prefix": "4WE6",                 // Manufacturer's series identifier
  "manufacturer": "Bosch Rexroth",         // Must match the company string
                                           // used in admin uploads exactly
  "spool_code": "E",                       // The literal spool code from the
                                           // ordering code position
  "description": "...",                    // Human-readable
  "center_condition": "...",               // De-energised flow paths
  "solenoid_a_function": "...",            // Per spec_comparison consumer
  "solenoid_b_function": "...",
  "canonical_pattern": "BLOCKED|PA-BT|PB-AT",  // FAMILY|sol-a|sol-b — the
                                               // matching key. Equality on
                                               // this string = spool match
  "topology": "4/3 double-solenoid",       // 4/2 single-solenoid /
                                           // 4/3 double-solenoid /
                                           // 4/2 double-solenoid
  "hand_build": "symmetric",               // symmetric / left / right
                                           // (for matching LH ↔ RH builds)
  "is_primary": true                       // Surface in primary spool dropdown
}
```

The seven canonical pattern families:

| Family | De-energised centre | Used by |
|---|---|---|
| `BLOCKED` | All ports blocked | DG4V `2C`/`2A`/`2N`, Rexroth `E`/`E1`/`D` |
| `OPEN` | All four interconnected | DG4V `0C`/`0A`/`10C`/`36`, Rexroth `H`/`C` |
| `TANDEM` | P→T, A&B blocked | DG4V `8C`/`6B`, Rexroth `G` |
| `FLOAT` | P blocked, A→T, B→T (or full float) | DG4V `6C`/`33C`/`H`, Rexroth `J`/`Q`/`W` |
| `REGEN` | P→A, P→B, T blocked | DG4V `4C`/`7C`, Rexroth `M` |
| `SELECTOR` | P→A (or B), other blocked, T isolated | DG4V `22A`/`22AL`, Rexroth `A`/`B` |
| `ASYMMETRIC` | One work port behaves differently | DG4V `31C`/`52C`, Rexroth `R`/`U` |

Plus unique non-Danfoss patterns for Rexroth `F`, `L`, `P`, `T`, `V` — these have
NO equivalent in the Danfoss user guide and intentionally fall through to
"contact sales rep" because no Danfoss entry shares their pattern string.

### Naming conventions encoded in the seed

- **`L` suffix on a Danfoss code** = spring-offset **left-hand build** (NOT lapped
  spool). E.g. `2A` is RH build, `2AL` is LH build (same family, mirrored
  centre). `22A` / `22AL` follow the same rule.
- **`73` suffix on a Rexroth code** = soft-shift / smooth-switching variant.
  Same canonical pattern as the base spool, but the matching layer should drop
  confidence by ~10% on cross-substitution because shift dynamics differ.
  Equivalent to Danfoss `FS` suffix.
- **`E1`, `E2`, `E3`** in Rexroth = transition-behaviour variants of `E`. `E1`
  has explicit P-to-A/B pre-opening with a pressure-intensification warning for
  differential-cylinder applications.
- **Letter + position suffix in Rexroth ordering codes** (e.g. `..EA..`,
  `..E73A..`) — the trailing `A`/`B` is the spool-position-a/b indicator, not
  part of spool functional identity.

### Idempotency caveat for the seed loader

`db.load_seed_spool_data()` skips reload when `spool_type_reference` already
contains rows with `source = 'seed'`. This means **after the seed file is
edited and pushed, an existing deployment won't pick up the changes on a normal
restart**. Two ways to force the new data:

1. Wipe the DB: `rm data/*.db data/*.npz` then restart (loses all uploaded
   products — only safe on a clean instance).
2. Trigger the admin app's force-reload action (`db.load_seed_spool_data(path,
   force=True)` at `admin_app.py:785`).

### Deferred audit items in the seed file

These are known inconsistencies that the v1.2 update did NOT fix (intentionally
deferred to a full audit):

- `0A` and `2A` have descriptions like "Open center" / "Closed center" but their
  `topology` was changed to `4/2 single-solenoid` to align with the verified
  Rexroth `C → 0A` and `D → 2A` cross-references (which are 4/2 spools with
  P→A,B→T centres). The description text doesn't match the topology yet.
- `6CL` was REMOVED in v1.1 because the domain expert confirmed it does not
  exist. If a downstream extraction sees `6CL`, treat as junk.
- `2N` is currently described as `4/3 closed center, no crossover`, but per
  the L-suffix convention it should logically be a separate spring-offset
  variant. Domain-expert review pending.

When extending or modifying, see "Interview-Driven Knowledge Updates" pattern
below.

## Skill File (`skills/hydraulics_engineer.md`)

The skill file is the prose counterpart to `spool_seed.json` — it provides
**extraction-time** domain knowledge to the LLM agents (the seed file provides
**runtime** matching translations).

### How it's loaded

`tools/agents/base.py:get_skill_context(*keys)` parses the file into sections by
`## N. <Name>` headers. Each section's key is the lowercased name with spaces
replaced by underscores. An agent calling `get_skill_context("spool", "unit",
"failure")` receives the concatenation of every section whose key contains any
of those substrings.

Current sections (7) and which agents pull them:

| Section | Pulled by |
|---|---|
| 1. Ordering Code Structure | ordering_code agent (matches "ordering") |
| 2. Spool Type Identification — Canonical Functional Taxonomy | ordering_code, product_extractor, spool_analyzer (all match "spool") |
| 3. Bosch Rexroth 4WE6 Spool Reference | (matches "spool") |
| 4. Spool Cross-References — Rexroth ↔ Danfoss with False Friends | (matches "spool") |
| 5. Key Specification Fields | product_extractor (matches "spec") |
| 6. Unit Normalisation | ordering_code, product_extractor, spool_analyzer (all match "unit") |
| 7. Common Failure Modes and False Friends | ordering_code, spool_analyzer (match "failure") |

### What the skill file authoritatively documents

- The full 20-spool Bosch Rexroth 4WE6 reference table (RE 23178 verified)
- The Rexroth → Danfoss cross-reference, including no-equivalent codes
- **The Rexroth-H ≠ Danfoss-H false friend** (the most expensive error type)
- L = LH build / 73 = soft-shift / E1 = pre-opening / 46 = SO407-OF version
- Topology distinction (4/2 single-solenoid vs 4/3 double-solenoid; 3-box symbols
  can be either)
- Hand-build matching rule (LH vs RH not interchangeable)
- Worked decoding examples for `DG4V-3-2C-M-U-H7-60` and `4WE6E62/EG24N9K4`
- DG4V-3 vs DG4V-5 spec table; ISO 4401 dimensional interchange table
- Voltage code hierarchy (G/H/A/B family + G7/H7 specific)
- Suffix code reference (FS, P, MU, X90, PVG, C/A/D centring)
- DON'T-list covering both extraction-time and matching-time failure modes

### When to update the skill file

- After any domain-expert clarification (e.g. correcting a spool function
  description, adding a manufacturer)
- When `spool_seed.json` changes — the prose narrative must stay aligned
- When a new false-friend or anti-pattern is observed in production matching
- Treat the skill file and the seed file as **twin authorities** — neither is
  complete without the other

## Manufacturer Coverage Status

Current state of `spool_seed.json` per the domain-expert interview:

| Manufacturer | Series | Entries | Status |
|---|---|---|---|
| Danfoss | DG4V | 21 | Cross-reference target. v1.1/v1.2 fixed `6C`, `8C`, `33C`, `2AL` descriptions; added `22A`, `22AL`, `7C`, `31C`, `52C`. |
| Bosch Rexroth | 4WE6 | 20 | Full coverage of A, B, C, D, E, E1, F, G, H, J, L, M, P, Q, R, T, U, V, W, Y. Verified against RE 23178 (2019-01). |
| Parker | D1VW | 0 | **Pending** — domain expert agreed to dictate cross-references; not started. |
| HAWE, Yuken, Atos, Nachi, MOOG | — | 0 | **Pending** — agreed to use canonical-pattern matching only (no per-letter table); spool diagrams will be classified into the 7 families at extraction time. |

The matching pipeline functionally works for any manufacturer once a seed entry
exists with a `canonical_pattern` matching one of the seven families.

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

### LLM Client Initialization
All LLM calls go through `tools/llm_client.py`. The client is lazily initialised via `get_client()` to avoid crashes when API keys aren't set at import time. Agent modules should never import `anthropic` or `openai` directly.

### Anthropic JSON Handling
Anthropic has no native JSON mode. `call_llm_json()` handles this by: appending "Return ONLY valid JSON" to the system prompt, stripping markdown fences from responses, and retrying with a fix-up prompt on parse failure. For guaranteed structured output, use `call_llm_tool()` which leverages Anthropic's tool use feature.

### Python 3.14
ChromaDB is incompatible (pydantic v1). The vector store uses numpy instead.

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
2. PyMuPDF (fitz) extracts text from ALL pages (supplemented by pdfplumber) -> agent structures products (31 spec fields) in batches covering the full document
3. Ordering code agent extracts ordering code breakdown tables -> combinatorial generator creates all product variants (capped at 500)
4. For user guides/datasheets: spec extractor agent also extracts model code decode patterns
5. Deduplication merges products from all sources (keeps richest specs)
6. Admin reviews extracted products (with Source column: table/llm/ordering_code), then confirms to index

### Ordering Code Combinatorial Generation (`parse_tools.py`)
When a PDF contains "Ordering code" / "How to Order" tables:
- `ordering_code.py` agent identifies segment positions, fixed/variable flags, separators, and field mappings
- `generate_products_from_ordering_code()` creates all permutations via `itertools.product()`
- `assemble_model_code()` reconstructs model codes from a template like `{01}{02}{03}-{04}/{05}`
- Empty "no code" options handled (double separators cleaned up)
- `MAX_COMBINATIONS = 500` safety cap
- Data models: `OrderingCodeSegment`, `OrderingCodeDefinition` in `models.py`

### Field Alias Rescue (`ingest.py`)
`_FIELD_ALIASES` maps ~60 common LLM/table output names to canonical HydraulicProduct fields (e.g. "pressure" -> "max_pressure_bar", "voltage" -> "coil_voltage"). Applied before product construction to catch non-standard field names.

### No-Match Diagnostics (`tools/agents/chat_agent.py:_handle_product_matching`)
When `find_equivalents` returns no matches, the response includes diagnostic information:
- How many Danfoss products are in the database
- How many are indexed in the vector store
- The competitor product's category (so the admin knows what to upload)
- Actionable suggestions (upload Danfoss products, re-index, check category coverage)

### Adding a New Product Category
Must be added in **6 places**:
1. `models.py` — `ProductCategory` enum
2. `admin_app.py` — `CATEGORIES` list
3. `distributor_app.py` — `CATEGORIES` list + `CATEGORY_MAP` dict
4. `tools/agents/chat_agent.py` — `category_hints` dict in `_regex_parse_fallback()`
5. `tools/parse_tools.py` — LLM prompt category lists (both extraction and ordering code prompts)
6. `prompts.py` — `QUERY_PARSER_PROMPT` category list

### Modifying the Matching Pipeline
- Scoring weights: `SCORE_WEIGHTS` in `models.py`
- Confidence threshold: `CONFIDENCE_THRESHOLD` in `models.py`
- Comparison logic: `spec_comparison()` in `storage/product_db.py`
- String matching tolerance: `_exact_match()` in `storage/product_db.py`
- Chat routing thresholds: `_route_after_lookup()` in `tools/agents/chat_agent.py`
- Fuzzy match thresholds: `identify_competitor_product()` in `tools/lookup_tools.py`
- DB fallback: `find_my_company_equivalents()` in `tools/lookup_tools.py`

### Interview-Driven Knowledge Updates (Spool Seed + Skill File)

When extending coverage to a new manufacturer or correcting an existing entry,
follow this pattern (proven in the v1.1 / v1.2 work):

1. **Source ground truth from a domain expert**, not public catalogues. Public
   parts websites have been observed to ship cross-reference tables with
   functional definitions swapped (e.g. claiming Vickers Type 6 is closed when
   it's tandem). The expert's user guide is canonical.
2. **For each new spool**, capture: code, topology, hand_build, de-energised
   centre, and which Danfoss code it cross-references to. The Danfoss code
   determines the `canonical_pattern` string to use.
3. **Add seed entries** to `data/spool_seed.json`. Set `canonical_pattern`
   identical to the Danfoss equivalent's pattern so equality matching fires.
   For codes with no Danfoss equivalent, give them a unique pattern string
   (e.g. `HYBRID-OPEN-TANDEM|...`) so they correctly fall through.
4. **Update the skill file** (`skills/hydraulics_engineer.md`) with the new
   manufacturer's reference table and any new false-friend warnings. Keep prose
   and seed in sync.
5. **Increment the seed file's `version`** and append the change to its
   `description` field.
6. **Bump the deployment**: branch protection on `main` blocks direct push, so
   commit to a feature branch, open a PR, merge. See "Deployment & Branch
   Workflow" below.
7. **Force-reload the seed on the deployed instance** if the DB is not wiped —
   the loader is idempotent and will skip otherwise.

### Adding a New Agent
1. Create `tools/agents/my_agent.py`
2. Import `call_llm` / `call_llm_json` / `call_llm_tool` from `tools.llm_client`
3. Choose the right tier (`TIER_HIGH` for critical, `TIER_MID` for standard, `TIER_LOW` for fast)
4. For vision tasks, use `build_image_block()` / `build_vision_content()` from `tools.agents.base`
5. Update `ingest.py` imports if the agent replaces an extraction function

## Dependencies

Core: anthropic (primary), openai (fallback), sentence-transformers, numpy, pydantic, gradio
PDF: pymupdf (primary extractor), pypdf (fallback), pdfplumber (tables + supplementary text)
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
- `chat_agent.py:_parse_query()` wraps the LLM call in try/except with a regex-based fallback parser
- If the LLM provider is unreachable, the regex fallback extracts model codes, competitor names, and categories from user input
- Cross-encoder reranking scores are clamped to `[0.0, 1.0]` to prevent Pydantic validation errors
- `llm_client.py` includes exponential backoff retry (3 attempts) for rate limits

### Configurable Binding
- All apps use env vars for host/port — critical for HF Spaces (`0.0.0.0` binding required)
- `max_file_size` set on Gradio launch to enforce upload limits server-side

## Deployment & Branch Workflow

### Live deployment

The live app is hosted on **Replit** (the user's account; URL not in repo).
Replit pulls from `https://github.com/liamcarter1/product_matcher` on each
deploy. The active branch is `main`. The `.replit` config runs `python app.py`
and maps localPort 7860 → externalPort 80.

The repo also ships:
- HF Spaces frontmatter in `README.md` (`title`, `sdk: gradio`, `app_file: app.py`)
  — could be deployed to a Space; not currently in active use.
- `Dockerfile` + `docker-compose.yml` + `deploy/nginx-match.conf` for VPS
  deployment behind Nginx (WebSocket-aware reverse proxy).

### Branch protection on `main`

`main` is **branch-protected** on GitHub — direct pushes return HTTP 403. All
changes must land via PR + merge. Workflow:

```bash
# Develop on a feature branch
git checkout -b feat/my-change
# ... edit, commit ...
git push -u origin feat/my-change
# Open a PR via the GitHub MCP tools or the GitHub UI, then merge.
# Replit will pick up the new main on its next pull/redeploy.
```

If using the GitHub MCP tools (e.g. when running Claude Code with GitHub
integration), the auto-merge pattern after `mcp__github__create_pull_request` is
`mcp__github__merge_pull_request` with `merge_method: "merge"`.

### After deploying a seed-file change

Replit will pull the updated `spool_seed.json` automatically, but the
`spool_type_reference` table in SQLite **won't be re-loaded** unless either:
- The DB is wiped (`rm data/*.db data/*.npz` in Replit Shell, then restart), or
- The admin app's force-reload action is triggered.

Plan the rollout accordingly — wiping the DB also drops uploaded products and
the vector store, requiring re-upload of all PDFs.

## Setup Gotchas (when onboarding a fresh environment)

These are the friction points encountered when bootstrapping the project on a
new machine or container.

### `_cffi_backend` / cryptography clash on Debian-based systems

`pdfplumber` → `pdfminer` → `cryptography` requires the rust-bindings cffi
backend. If the system already has a Debian-installed `cryptography` package, a
plain `pip install -r requirements.txt` may not override it cleanly, leaving
`pyo3_runtime.PanicException` on import. Fix:

```bash
pip install --upgrade --ignore-installed cffi cryptography
```

### sentence-transformers offline / restricted environments

`storage/vector_store.py` initialises the embedding model
`sentence-transformers/all-MiniLM-L6-v2` and the cross-encoder reranker
`cross-encoder/ms-marco-MiniLM-L-6-v2` lazily on first use. If the host has no
outbound access to `huggingface.co`, the app fails to start with `OSError: We
couldn't connect to 'https://huggingface.co'`. Pre-cache the models on a host
with internet, then mirror the cache to the offline host (`~/.cache/huggingface`),
or set `HF_HOME` to a pre-populated directory.

### `python-dotenv` not always installed by `pip install -r requirements.txt`

If a script importing `dotenv` fails with `ModuleNotFoundError`, install
explicitly: `pip install python-dotenv`. (Observed when system pip silently
skipped it during the requirements install.)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required if LLM_PROVIDER=anthropic) | Anthropic API key for Claude models |
| `OPENAI_API_KEY` | (required if LLM_PROVIDER=openai) | OpenAI API key for GPT models |
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `ADMIN_USERNAME` | (none) | Login username for Gradio auth (optional, but recommended for deployed apps) |
| `ADMIN_PASSWORD` | (none) | Login password for Gradio auth (optional, but recommended for deployed apps) |
| `ADMIN_HOST` | `127.0.0.1` | Bind address for admin_app.py |
| `ADMIN_PORT` | `7861` | Port for admin_app.py |
| `DISTRIBUTOR_HOST` | `0.0.0.0` | Bind address for distributor_app.py |
| `DISTRIBUTOR_PORT` | `7860` | Port for distributor_app.py |
| `PORT` | `7860` | Port for combined app.py (HF Spaces sets this automatically) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size in megabytes |
