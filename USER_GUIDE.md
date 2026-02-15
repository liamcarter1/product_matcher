# ProductMatchPro - User Guide

A comprehensive guide for staff and distributors using the ProductMatchPro hydraulic product cross-reference application.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Admin Console (Back-Office)](#2-admin-console-back-office)
   - [Upload Documents Tab](#21-upload-documents-tab)
   - [Product Database Tab](#22-product-database-tab)
   - [Feedback Review Tab](#23-feedback-review-tab)
   - [Settings Tab](#24-settings-tab)
3. [Distributor App (Product Finder)](#3-distributor-app-product-finder)
4. [Understanding Confidence Scores](#4-understanding-confidence-scores)
5. [Best Practices](#5-best-practices)
6. [Troubleshooting](#6-troubleshooting)
7. [Deploying to Hugging Face Spaces](#7-deploying-to-hugging-face-spaces)

---

## 1. Getting Started

### Prerequisites

- Python 3.10 or later
- An OpenAI API key (used for PDF extraction and query parsing)

### Installation

```bash
cd product_matcher
pip install -r requirements.txt
```

### Setting Up Your API Key

Create a file called `.env` in the `product_matcher/` folder with the following content:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Replace `sk-your-api-key-here` with your actual OpenAI API key. This file is gitignored and will not be committed to version control.

### Setting Up Authentication (Recommended)

To protect the web interface with a login screen, add credentials to your `.env` file:

```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-secure-password
```

When these are set, both the admin and distributor apps will show a login screen before granting access. If these variables are not set, the apps run without authentication (suitable for local development only).

**Important:** Always enable authentication when the app is accessible over a network or deployed to Hugging Face Spaces.

### Launching the Apps

There are three ways to run the application:

| Command | What it does | URL |
|---------|-------------|-----|
| `python admin_app.py` | Admin console only | http://localhost:7861 |
| `python distributor_app.py` | Distributor search only | http://localhost:7860 |
| `python app.py` | Both apps in a tabbed interface | http://localhost:7860 |

For day-to-day use, you will typically run the admin and distributor apps separately. The combined `app.py` is designed for Hugging Face Spaces deployment.

---

## 2. Admin Console (Back-Office)

The admin console is the back-office tool used by staff to manage the product database. It is **not visible to distributors**. Open it at **http://localhost:7861** after running `python admin_app.py`.

The console has four tabs: **Upload Documents**, **Product Database**, **Feedback Review**, and **Settings**.

---

### 2.1 Upload Documents Tab

This is where you add new products to the system by uploading PDF catalogues, user guides, and datasheets.

#### Layout

The left side contains the upload form, and the right side shows the results.

**Left panel (inputs):**
- **Upload PDF(s)** - Drag and drop or click to select one or more PDF files. You can upload multiple files at once and they will all be processed as a batch under the same company name
- **Company Name** - Dropdown to select the manufacturer. Danfoss is the default. Also includes Bosch Rexroth, Parker, ATOS, and any companies you have previously uploaded. You can also type a new company name if needed
- **Document Type** - Dropdown to select the type of document
- **Product Category** - Dropdown to narrow the extraction scope
- **Upload & Process** button

**Right panel (results):**
- **Status** - Shows progress messages and extraction results
- **Extracted Products** table - Review the products before confirming
- **Confirm & Index** button - Saves the products to the database

#### Step-by-Step: Uploading a Product Catalogue

**Example scenario:** You have received a new Bosch Rexroth catalogue PDF containing their directional valve range.

1. **Upload the PDF(s)**
   Click the upload area or drag and drop your PDF files. Only `.pdf` files are accepted. You can select multiple files at once — they will all be processed as a single batch under the same company name, document type, and category.

   **When to use multi-file upload:** When you have several PDFs from the same manufacturer (e.g. separate catalogue pages saved as individual files, or multiple datasheets for the same product range). All files in a batch must belong to the same company.

2. **Select the Company Name**
   Choose the manufacturer from the dropdown. **Danfoss** is selected by default (since you will be uploading your own Danfoss catalogues as the primary reference data).
   - Select `Danfoss` when uploading your own product catalogues, user guides, or datasheets
   - Select a competitor name (e.g. `Bosch Rexroth`, `Parker`, `ATOS`) when uploading their documents
   - If the company is not in the list, simply type a new name — the dropdown accepts custom values
   - The dropdown automatically includes any companies you have previously uploaded
   - Be consistent with naming. "Bosch Rexroth" and "Rexroth" would be treated as different companies unless you set up a synonym (see [Settings Tab](#24-settings-tab))

3. **Select the Document Type**
   Choose from:
   - **catalogue** - Product catalogues with model codes, spec tables, and product listings. This is the most common choice. The system will try to extract product tables using pdfplumber first, then fall back to LLM extraction if tables cannot be parsed. If ordering code tables are found, all variants are generated automatically.
   - **user_guide** - User/installation guides that contain "How to Order" pages and model code breakdowns. These are especially valuable because the system will extract **ordering code breakdown tables** and generate all product variants, plus store model code decode patterns for future use.
   - **datasheet** - Individual product datasheets with detailed specifications. Like user guides, these also trigger ordering code generation and pattern extraction. Best for single product series with full ordering code tables.

4. **Select the Product Category** (optional)
   Choose from: Directional Valves, Proportional Directional Valves, Pressure Valves, Flow Valves, Pumps, Motors, Cylinders, Filters, Accumulators, Hoses & Fittings, or Other.
   - Select **All / Auto-detect** if the PDF covers multiple categories or you want the system to determine the category automatically.
   - Selecting a specific category helps the extraction accuracy, especially for large catalogues with mixed content.

5. **Click "Upload & Process"**
   The system will:
   - Parse the PDF using **PyMuPDF** (for high-quality text from ALL pages), supplemented by **pdfplumber** (for structured tables and any additional text). This dual-extractor approach ensures complex layouts, multi-column text, operating data tables, and technical specifications are all captured
   - If tables are found, map the table columns to product fields using ~90 header patterns (model code, pressure, flow, voltage, seal material, etc.)
   - Send text to GPT-4o-mini for structured product extraction (31 spec fields). **The full document is processed in batches** — not truncated — so products from all pages are captured, not just the first few
   - **Ordering code generation:** Detect "Ordering code" / "How to Order" breakdown tables and generate ALL product variants by combining variable segments (e.g. 4 flow rates x 2 seals x 2 interfaces = 16 products)
   - For user guides/datasheets: extract model code decode patterns for future use
   - **Index guide text using two-pass strategy:** Page-level chunks (with `[Page N]` metadata for direct page retrieval) plus full-document chunks (for cross-page context). Chunk size: 1500 characters with 300-character overlap
   - Deduplicate products from all sources, keeping the variant with the most populated specs

   The Status field will show a message like:
   > Extracted 83 products from 1 file(s). Also indexed 83 text chunks.
   > Sources: 15 from table extraction, 68 generated from ordering code table(s).
   >
   > File results:
   >   Bosch_Rexroth_4WRE_Datasheet.pdf: 83 products, 83 text chunks
   >
   > Review the products below and click 'Confirm & Index' to add them to the database.

   If you uploaded multiple files, you will see a per-file breakdown showing how many products were extracted from each.

6. **Review the Extracted Products**
   The table shows each extracted product with:
   - **Source** - Where the product came from: `table` (PDF table extraction), `llm` (LLM text extraction), or `ordering_code` (generated from ordering code breakdown table)
   - **Model Code** - The product's model number (for ordering code products, this is the fully assembled code e.g. `4WREE6E16-3XV/24A1`)
   - **Product Name** - Human-readable name
   - **Category** - Detected or assigned category
   - **Confidence** - How confident the extraction was (extraction confidence, not matching confidence)
   - **Key specs** - max_pressure_bar, max_flow_lpm, coil_voltage, valve_size, seal_material, actuator_type, port_size, mounting, num_ports

   **Check the results carefully.** If the PDF was poorly formatted or the extraction missed products, you may want to try uploading again with a different Document Type setting. For ordering code products, scroll through to verify the generated combinations look correct.

7. **Click "Confirm & Index"**
   This saves all extracted products to:
   - **SQLite database** - For structured search and spec comparisons
   - **Vector store** - For semantic similarity search

   You will see a confirmation message:
   > "Successfully indexed 47 products into the database and vector store."

#### What Happens with Ordering Code Tables (Combinatorial Generation)

This is the most powerful feature of the system. When you upload any PDF, the system looks for "Ordering code" / "How to Order" breakdown tables.

For example, a Bosch Rexroth datasheet might show an ordering code table like:

```
Position 01: "4"   (fixed)    - 4 main ports
Position 02: "WRE" (fixed)    - Proportional directional valve
Position 03: "E"   (fixed)    - With integrated electronics
Position 04: "6"   (fixed)    - Size 6
Position 06: flow rate (variable) - 04 (4 l/min), 08 (8 l/min), 16 (16 l/min), 32 (32 l/min)
Position 09: seal (variable)  - V (FKM), M (NBR)
Position 10: "24"  (fixed)    - 24V supply voltage
Position 11: interface (variable) - A1 (command ±10V), F1 (command 4-20mA)
Position 12: special (variable)   - (none), -967 (with pressure compensation)
```

The system uses GPT-4o-mini to parse these tables, then **generates ALL valid combinations** as separate product entries. In this example: 4 flow rates x 2 seals x 2 interfaces x 2 specials = **32 unique products**, each with:
- A fully assembled model code (e.g. `4WREE6E16-3XV/24A1`)
- All spec fields populated from the segment mappings (e.g. `max_flow_lpm=16, seal_material=FKM, coil_voltage=24VDC`)

This means that when a distributor later searches for any specific variant like "4WREE6E32-3XM/24F1", the system has that exact product in the database with all its specs, enabling accurate field-by-field comparison.

The system also stores model code decode patterns in the `model_code_patterns` table for future use, so even products that were extracted from a simple table row can have their specs enriched by decoding their model code.

**Generation is capped at 500 products per ordering code table** to prevent combinatorial explosion. If the cap is hit, the admin is warned in the status message.

**Spool type extraction:** The system now includes specialised hydraulic engineering guidance for extracting spool type designations from ordering code tables and model code breakdowns. Spool types define the valve's center condition (e.g. all ports blocked, all ports open to tank, P blocked with A&B to T). The extraction prompt recognises manufacturer-specific codes: Danfoss/Vickers (2A, 33C, 6C, etc.), Bosch Rexroth (D, E, H, J, etc.), Parker (01, 02, 06, etc.), and MOOG. Each spool code is stored with both the code and its functional description (e.g. "2A - All ports open to tank in center"), enabling cross-manufacturer matching based on equivalent valve function.

**Tip:** Upload datasheets and user guides (which contain ordering code tables) BEFORE plain catalogues. This both generates the richest product data and populates decode patterns for future enrichment.

#### Common Upload Scenarios

| Scenario | Document Type | Category | Notes |
|----------|--------------|----------|-------|
| Competitor's full product catalogue | catalogue | All / Auto-detect | Covers multiple product types |
| Competitor's valve-specific catalogue | catalogue | Directional Valves | Narrower = better extraction |
| Competitor's "How to Order" guide | user_guide | All / Auto-detect | Generates all ordering code variants + decode patterns |
| Single product datasheet | datasheet | (select specific) | Best for one product — generates ordering code variants if table found |
| Your own Danfoss catalogue | catalogue | (select specific) | Use "Danfoss" as company name |
| Your own Danfoss datasheet | datasheet | (select specific) | Use "Danfoss" — generates all your product variants for matching |

#### If the Extraction Fails

- **"No products could be extracted"** - Try a different Document Type. Some PDFs work better with `catalogue` (table extraction) vs `user_guide` (LLM extraction).
- **Few products extracted** - The PDF may have complex formatting. Try uploading individual sections as separate PDFs.
- **Incorrect specs** - The table header mapping may not have matched. Review the extracted data carefully before confirming. You can delete incorrect products later from the Product Database tab.

---

### 2.2 Product Database Tab

This tab lets you browse, search, and manage all products in the database.

#### Features

**Search and Filter:**
- **Search** text field - Type a model code or product name to filter
- **Company** dropdown - Filter by company name (populated from uploaded data)
- **Search** button - Applies the filters

The search checks both model codes and product names (case-insensitive). Results are limited to 100 rows.

**Example:** To find all Parker directional valves, type "D1VW" in the search field and select "Parker" from the company dropdown, then click Search.

**Product Counts:**
Below the search results, you will see a summary table showing how many products are stored per company and category. This helps you track what has been uploaded.

**Export CSV:**
Click **Export CSV** to download all products as a CSV file. This exports every product in the database with all spec fields (except raw_text and model_code_decoded which are excluded to keep the file manageable). The file is saved to your system's temp directory and offered as a download.

**Re-index Vector Store:**
Click **Re-index Vector Store** to rebuild the semantic search index from scratch using all products in the SQLite database. Use this if:
- You manually edited the database outside the app
- The vector store files appear corrupted
- Search results seem wrong or incomplete

You will see a message like: "Re-indexed 234 products in the vector store."

**Delete Product:**
To remove a product:
1. Find its ID in the search results table (shown as the first 8 characters)
2. Type those characters into the "Delete Product" field
3. Click **Delete**

The product will be removed from both SQLite and the vector store.

**Example:** If the table shows ID `a1b2c3d4` for a product you want to remove, type `a1b2c3d4` and click Delete. You will see: "Deleted product 4WE6D6X/EG24N9K4 (a1b2c3d4)".

---

### 2.3 Feedback Review Tab

This tab shows distributor feedback and lets you manually confirm product equivalents.

#### Distributor Feedback Table

When distributors use the Product Finder app and click the thumbs up or thumbs down buttons, their feedback is stored here. The table shows:
- **Date** - When the feedback was given
- **Query** - What the distributor searched for
- **Competitor** - The competitor model code they were looking up
- **Our Product** - The equivalent that was suggested
- **Confidence** - The confidence score of the match
- **Thumbs Up** - Yes or No

Click **Refresh** to load the latest feedback.

**How to use this data:** Look for patterns in thumbs-down feedback. If distributors consistently reject a particular match, it may indicate that the automated matching is wrong for that product pair. You can fix this by creating a confirmed equivalent (see below).

#### Manually Confirm an Equivalent

This is a critical feature for improving accuracy. When you know that a specific competitor product maps to a specific product of yours, you can create a **confirmed equivalent** that overrides the algorithmic matching.

**Fields:**
- **Competitor Model Code** - The competitor's model code (e.g. `4WE6D6X/EG24N9K4`)
- **Competitor Company** - The competitor's name (e.g. `Bosch Rexroth`)
- **Danfoss Model Code** - Your Danfoss equivalent product code

**Example:**
1. From the feedback table, you notice that `4WE6D6X/EG24N9K4` keeps getting matched with the wrong Danfoss product
2. You know the correct Danfoss equivalent is `DHZO-AE-073-L5`
3. Fill in:
   - Competitor Model Code: `4WE6D6X/EG24N9K4`
   - Competitor Company: `Bosch Rexroth`
   - Danfoss Model Code: `DHZO-AE-073-L5`
4. Click **Confirm Equivalent**

From now on, whenever a distributor searches for that competitor code (or a fuzzy match to it), the system will immediately return the confirmed equivalent with a "Confirmed Equivalent (verified by our team)" badge, bypassing the algorithmic matching entirely.

---

### 2.4 Settings Tab

This tab provides system configuration and monitoring.

#### Brand Synonyms

Hydraulic brands have many names. A single company might be known by its current name, a legacy name, or an abbreviation. The synonym mapping ensures all variations are treated as the same company.

**Fields:**
- **Term** - The alternative name (e.g. `Rexroth`)
- **Canonical Name** - The "official" name as stored in your database (e.g. `Bosch Rexroth`)

**Example mappings:**

| Term | Canonical Name | Why |
|------|---------------|-----|
| Rexroth | Bosch Rexroth | Common abbreviation |
| Denison | Parker | Denison was acquired by Parker |
| Atos SpA | ATOS | Variant naming |
| Daikin | Daikin Industries | Variant naming |

When a distributor types "Rexroth 4WE6", the system resolves "Rexroth" to "Bosch Rexroth" before searching, ensuring it finds the right products.

Click **Add Synonym** to save a new mapping.

#### Vector Store Status

Shows the current count of indexed items in each vector store collection:
- **Danfoss Products** - Number of your Danfoss products indexed for semantic search
- **Competitor Products** - Number of competitor products indexed
- **Guide Chunks** - Number of text chunks from uploaded user guides/datasheets

Click **Refresh** to update the counts.

**What to look for:** If these numbers seem low compared to what you have uploaded, try clicking "Re-index Vector Store" on the Product Database tab.

#### Configuration Reference

The Settings tab also displays the current system configuration:
- **Confidence Threshold:** 75% - Matches below this are flagged as uncertain (changeable in `models.py` by editing `CONFIDENCE_THRESHOLD`)
- **Sales Contact:** The contact details shown to distributors when confidence is below threshold (changeable in `graph.py` by editing `SALES_CONTACT`)
- **Model Code Patterns:** Automatically extracted from user guide uploads

---

## 3. Distributor App (Product Finder)

The distributor-facing app is a clean chat interface where distributors search for equivalent products. Open it at **http://localhost:7860** after running `python distributor_app.py`.

### How to Search

**Option A: Enter a model code**
Type a full or partial competitor model code directly into the search box and press Enter or click Search.

Examples:
- Full code: `4WE6D6X/EG24N9K4`
- Partial code: `4WE6D6X`
- Series only: `4WE6`

**Option B: Use brand + code**
Type the competitor brand name followed by the model code.

Examples:
- `Parker D1VW020BN`
- `Bosch Rexroth 4WE6`
- `ATOS DHI-0631`

**Option C: Describe what you need**
Type a natural language description of the product specifications.

Examples:
- `24V solenoid directional valve CETOP 5 315 bar`
- `hydraulic pump 100cc displacement`

**Option D: Click an example**
At the bottom of the page, several example searches are provided. Click any of them to use it as your search query.

### Using the Filters

Below the search box, two optional dropdown filters help narrow results:

- **Category** - Limits the search to a specific product type (Directional Valves, Proportional Directional Valves, Pressure Valves, Flow Valves, Pumps, Motors, Cylinders, Filters, Accumulators, Hoses & Fittings)
- **Competitor** - Limits the search to a specific competitor brand. The dropdown is populated from the database and refreshes automatically when new companies are uploaded. You can also type any competitor name directly — even if it is not yet in the list.

Leave both set to "All" for the broadest search.

### Understanding the Results

The system will return one of four types of response:

**1. Match Found (confidence >= 75%)**
You will see:
- A confidence score with a visual bar (e.g. "Match Found! Confidence: 87% ████████░░")
- The competitor product details
- The recommended equivalent product
- A spec comparison table showing how each specification compares (with checkmarks for matches and crosses for differences)
- Up to 2 alternative options with their confidence scores
- An AI-generated analysis explaining the match

**2. Partial Match (confidence < 75%)**
When the best match is below the 75% threshold:
- The system explains that it cannot confidently recommend a drop-in replacement
- It lists the closest partial matches for reference (with their confidence scores)
- It provides your sales representative's contact information

**3. Ambiguous Match**
If the search matches multiple products with similar scores:
- The system lists up to 5 options with their match percentages
- It asks you to clarify by typing a number, the full model code, or more details

**4. No Match Found**
If the product cannot be found at all:
- Suggestions to check spelling or try a shorter code
- Your sales representative's contact details

**5. Competitor Found but No Equivalent**
If the competitor product is identified but no equivalent exists in the database:
- An explanation of **why** no match was found (e.g. no Danfoss products uploaded, search index empty, no products in the same category)
- The number of Danfoss products in the database and search index
- Guidance for what the administrator needs to upload
- Your sales representative's contact details

### Giving Feedback

After receiving a result, click:
- **Thumbs Up** if the suggested equivalent is correct
- **Thumbs Down** if the match is wrong or unhelpful

This feedback is stored and visible to admin staff in the Feedback Review tab, helping improve the system over time.

### Starting a New Session

Click **New Search Session** to clear the conversation history and start fresh. This creates a new conversation thread - useful if the previous search context is no longer relevant.

---

## 4. Understanding Confidence Scores

The confidence score (0-100%) represents how closely a candidate product matches the competitor product across 12 weighted dimensions:

| Dimension | Weight | Type | What it Compares |
|-----------|--------|------|------------------|
| Semantic Similarity | 15% | Continuous | Overall description similarity from AI embeddings |
| Category Match | 10% | Gate | Same product category (if mismatched, total capped at 30%) |
| Max Pressure | 10% | Numerical | How close the pressure ratings are |
| Max Flow | 10% | Numerical | How close the flow ratings are |
| Valve Size | 10% | Fuzzy | CETOP/NG size (tolerates formatting differences like "CETOP 5" vs "CETOP 05") |
| Coil Voltage | 10% | Fuzzy | "24VDC" matches "24 VDC"; "110VAC" matches "110 VAC" |
| Spool Function | 8% | Fuzzy | Valve spool type/function code and center condition (e.g. "2A - All ports open to tank") |
| Actuator Type | 8% | Fuzzy | "proportional_solenoid" matches "proportional solenoid" |
| Mounting | 8% | Fuzzy | Mounting pattern (e.g. "ISO 4401-05" matches "ISO4401-05") |
| Port Size | 6% | Fuzzy | Port size and thread type |
| Seal Material | 3% | Fuzzy | NBR, FKM/Viton, etc. (different materials score 0.0) |
| Temp Range | 2% | Range | Does the candidate cover the operating temperature range? |

**Important rules:**
- If a spec is missing from **both** products, it is excluded from scoring (does not penalise)
- If a spec is missing from **one** product only, it scores 0.5 (neutral - not penalised fully, not rewarded)
- If the **category** does not match, the total score is capped at 30% regardless of other matches
- The **75% threshold** means: scores at or above 75% are presented as confident matches; scores below 75% trigger a "contact sales rep" message

---

## 5. Best Practices

### For Admin Staff

**Building the Database:**
1. Start by uploading your own company's datasheets and user guides first - these contain ordering code tables that generate all your product variants with full specs
2. Then upload your own catalogues - these add any products not covered by the datasheets
3. Next, upload competitor datasheets and user guides (generates their product variants + decode patterns)
4. Finally, upload competitor catalogues for any remaining products

This order ensures the richest possible product data for accurate matching. Datasheets with ordering code tables produce the best results because every product variant gets its own database entry with fully populated specs.

**Maintaining Accuracy:**
- Check the Feedback Review tab regularly for thumbs-down feedback
- Create confirmed equivalents for your most commonly requested products
- Add brand synonyms for any competitor naming variations your distributors use
- After bulk uploads, spot-check a few products in the Product Database tab to verify extraction quality
- Re-index the vector store if search results seem stale

**Naming Conventions:**
- Use consistent company names when uploading (e.g. always "Bosch Rexroth", not sometimes "Bosch" and sometimes "Rexroth")
- If you must use variations, set up synonyms in the Settings tab
- Categories should match the predefined list (directional_valves, proportional_directional_valves, pressure_valves, flow_valves, pumps, motors, cylinders, filters, accumulators, hoses_fittings, other)

### For Distributors

- **Start specific:** Enter as much of the model code as you know. Longer codes give more accurate results.
- **Use filters:** If you know the product type or competitor, use the dropdown filters to narrow the search.
- **Check the spec table:** Even when confidence is high, review the spec comparison table to confirm critical specs (coil voltage, mounting pattern) match your needs.
- **Give feedback:** Your thumbs up/down helps improve the system for everyone.
- **Ask follow-up questions:** The system remembers your conversation, so you can ask things like "what about a 110VAC version?" after an initial search.

---

## 6. Troubleshooting

### Common Issues

**"No products could be extracted from this PDF"**
- The PDF may have an unusual format. Try selecting a different Document Type (e.g. switch from "catalogue" to "user_guide")
- Scanned PDFs without embedded text will not work - the text must be selectable in the PDF. The system uses PyMuPDF and pdfplumber for text extraction, not OCR
- Very large PDFs may time out. Try splitting into smaller sections
- Check the terminal logs for per-page extraction details — the system now logs how many pages each extractor processed

**"Error processing PDF" or "An error occurred during processing"**
- Check that your OpenAI API key is set correctly in the `.env` file
- Check your internet connection (the LLM extraction requires API access)
- Check the terminal/server logs for detailed error messages (the web UI shows sanitised errors for security)
- If you see "API configuration error", your OpenAI API key is missing or invalid

**Search returns no results or "couldn't find a matching equivalent"**
- The system now explains why no match was found (no Danfoss products, empty index, or category mismatch) — read the response carefully for guidance
- Verify Danfoss products exist in the database (check the Product Database tab)
- Try a shorter or more general search term
- Check if the vector store has been indexed (see the Settings tab for collection counts)
- Try the "Re-index Vector Store" button on the Product Database tab
- You can select and copy any text from the admin tables for diagnostic purposes

**Knowledge base questions return "couldn't find specific information"**
- Ensure user guides have been uploaded AND processed (the "Confirm & Index" step must be completed)
- Check the Guide Chunks count in Settings tab — if zero, the text wasn't indexed
- The system now indexes guide text per-page with page number metadata, so specific content (e.g. "operating data on page 6") is directly searchable
- Try more specific queries — include technical terms that would appear in the document (e.g. "power supply voltage 24V" rather than just "electrical specs")
- If guides were uploaded before this update, re-upload them to benefit from the improved extraction and two-pass indexing

**Incorrect matches being returned**
- Create a confirmed equivalent in the Feedback Review tab to override the algorithmic match
- Upload more detailed user guides to improve model code decoding
- Check if there are duplicate or incorrectly extracted products in the database

**Apps fail to start**
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the `.env` file exists with a valid `OPENAI_API_KEY`
- Check no other application is using port 7860 or 7861

**Login screen appears unexpectedly**
- This means `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set in your environment or `.env` file
- Enter the credentials you configured, or remove these variables from `.env` to disable authentication

**"OPENAI_API_KEY not set" errors**
- Create a `.env` file in the `product_matcher/` directory
- Ensure the file contains: `OPENAI_API_KEY=sk-your-actual-key`
- Restart the application after editing `.env`

**PDF upload rejected ("not a valid PDF file" or "too large")**
- The system validates that uploaded files are genuine PDFs (checks the file header)
- Maximum upload size defaults to 50MB (configurable via `MAX_UPLOAD_SIZE_MB` env var)
- Ensure you are uploading actual PDF files, not renamed Word documents or images

---

---

## 7. Deploying to Hugging Face Spaces

You can deploy ProductMatchPro to Hugging Face Spaces for easy sharing with colleagues and trial use. The combined `app.py` provides both the Product Finder and Admin Console in a single tabbed interface behind a login screen.

### Step-by-Step Deployment

1. **Create a new Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Gradio** as the SDK
   - Set the Space name (e.g. `ProductMatchPro`)
   - Choose visibility (Private recommended for internal use)

2. **Push the code**
   Push the contents of the `product_matcher/` directory to the Space repository. The key file is `app.py` — Hugging Face will detect it automatically.

3. **Set Space Secrets**
   In the Space settings, add the following secrets:
   - `OPENAI_API_KEY` — Your OpenAI API key
   - `ADMIN_USERNAME` — Login username (e.g. `admin`)
   - `ADMIN_PASSWORD` — Login password

   These are set in the Space's **Settings > Repository secrets** section. They work exactly like environment variables but are kept secure.

4. **Wait for the build**
   HF Spaces will install dependencies from `requirements.txt` and launch `app.py`. The build typically takes 2-3 minutes.

5. **Log in and use**
   Once the Space is running, you will see a login screen. Enter the username and password you set in the secrets. After logging in, you will see two tabs:
   - **Product Finder** — The distributor-facing search interface
   - **Admin Console** — PDF upload, product management, feedback review, and settings

### Workflow on HF Spaces

Since the database starts empty on a fresh deployment, you need to populate it first:

1. **Go to the Admin Console tab**
2. **Upload your Danfoss catalogues and user guides** (see [Section 2.1](#21-upload-documents-tab))
3. **Upload competitor catalogues and user guides**
4. **Switch to the Product Finder tab** and test searches

**Note:** The SQLite database and vector store persist within the Space's file system. However, if the Space is rebuilt (e.g. after a code push), the data will be lost. For persistent storage, consider using Hugging Face Datasets or an external database.

### Limitations on HF Spaces

- **Cold starts:** Free-tier Spaces may sleep after inactivity and take 30-60 seconds to wake up
- **Storage:** Data files persist during the Space's lifetime but are lost on rebuild
- **Resources:** Free-tier Spaces have limited CPU/RAM. The sentence-transformers models may be slow on first load
- **File uploads:** PDFs are uploaded through Gradio's file handling, which works the same as locally

---

## 8. Deploying with Docker (VPS)

ProductMatchPro can be deployed to a VPS using Docker. A `Dockerfile`, `docker-compose.yml`, and Nginx config are included.

### Prerequisites

- A VPS with Docker and Nginx installed
- SSH access to the VPS
- A domain/subdomain pointed at the VPS IP

### Deployment Steps

1. **Zip and transfer the project** from your local machine:
   ```powershell
   # On Windows (PowerShell)
   Compress-Archive -Path * -DestinationPath productmatchpro.zip -Force
   scp productmatchpro.zip root@YOUR_VPS_IP:/opt/productmatchpro/
   ```

2. **Extract and build on the VPS:**
   ```bash
   cd /opt/productmatchpro
   unzip -o productmatchpro.zip
   docker compose up -d --build
   ```

3. **Configure Nginx** (copy from `deploy/nginx-match.conf`):
   ```bash
   cp deploy/nginx-match.conf /etc/nginx/sites-available/match.conf
   ln -sf /etc/nginx/sites-available/match.conf /etc/nginx/sites-enabled/
   nginx -t && systemctl reload nginx
   ```

4. **Get an SSL certificate:**
   ```bash
   certbot --nginx -d your-subdomain.example.com
   ```

### Redeployment Workflow

After making local code changes:

```bash
# 1. Zip the project
Compress-Archive -Path * -DestinationPath productmatchpro.zip -Force

# 2. Transfer to VPS
scp productmatchpro.zip root@YOUR_VPS_IP:/tmp/

# 3. SSH in and rebuild
ssh root@YOUR_VPS_IP
cd /opt/productmatchpro
unzip -o /tmp/productmatchpro.zip
docker compose up -d --build
```

The Docker build caches pip install layers, so rebuilds after code-only changes take seconds.

### Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers 156 tests across ingestion (type coercion, field aliases, deduplication), PDF parsing (ordering code generation, model code assembly, LLM prompt validation), and storage (CRUD, spec comparison, fuzzy lookup, model code decoding).

---

*ProductMatchPro - Built for accurate hydraulic product cross-referencing.*
