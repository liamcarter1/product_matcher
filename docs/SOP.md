# Danfoss RAG Chatbot - Standard Operating Procedure (SOP)

## Overview

This document explains how to add documents to your RAG chatbot, how the system processes them, and how to deploy the chatbot on your website for a trade show demo or production use.

---

## PART 1: Adding Documents to the Vector Database

### Step 1: Prepare Your Documents

**Supported File Types:**
| Format | Extensions | Best For |
|--------|------------|----------|
| PDF | `.pdf` | User manuals, product guides, technical documentation |
| Excel | `.xlsx`, `.xls` | Parts cross-reference tables, product specifications |
| CSV | `.csv` | Parts data, specification lists |

**Where to Place Files:**
```
danfoss_rag/
└── data/                    <-- PUT YOUR FILES HERE
    ├── product_manual.pdf
    ├── parts_crossref.xlsx
    └── specifications.csv
```

### Step 2: Run the Ingestion Script

**Option A: Ingest All Files in the Data Folder**
```bash
cd danfoss_rag
python scripts/ingest_documents.py --dir ./data
```

**Option B: Ingest a Single File**
```bash
python scripts/ingest_documents.py --file ./data/my_document.pdf
```

**Option C: Clear Old Data and Re-ingest (for updates)**
```bash
python scripts/ingest_documents.py --dir ./data --clear-existing
```

**Option D: Preview What Will Be Ingested (Dry Run)**
```bash
python scripts/ingest_documents.py --dir ./data --dry-run
```

---

## PART 2: How the System Processes Your Documents

### The Data Flow Pipeline

```
YOUR DOCUMENT
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DOCUMENT LOADING                                   │
│  File: backend/app/services/document_loader.py              │
│                                                             │
│  • Detects file type (PDF, Excel, CSV)                     │
│  • Extracts text from PDFs (all pages)                     │
│  • Reads rows from Excel/CSV files                         │
│  • Auto-detects column types (part numbers, specs, etc.)   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TEXT CHUNKING                                      │
│  File: backend/app/services/document_loader.py              │
│                                                             │
│  • Splits long documents into chunks (1000 characters)     │
│  • Maintains 200-character overlap between chunks          │
│  • Preserves context by keeping sentences together         │
│  • Adds metadata: source file, page number, chunk index    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING CREATION                                 │
│  File: backend/app/services/pinecone_service.py             │
│                                                             │
│  • Sends text to OpenAI API (text-embedding-3-small)       │
│  • Converts text into 1536-dimension vectors               │
│  • Processes in batches of 100 for efficiency              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR STORAGE (Pinecone)                         │
│  File: backend/app/services/pinecone_service.py             │
│                                                             │
│  • Stores vectors with unique IDs (MD5 hash)               │
│  • Organizes into namespaces:                              │
│    - "products" = parts cross-reference data               │
│    - "guides" = general documentation                      │
│  • Stores metadata for retrieval (source, text preview)    │
└─────────────────────────────────────────────────────────────┘
```

### Key Python Files Summary

| File | Location | Purpose |
|------|----------|---------|
| `ingest_documents.py` | `scripts/` | CLI script to run the ingestion |
| `document_loader.py` | `backend/app/services/` | Loads & chunks documents |
| `pinecone_service.py` | `backend/app/services/` | Handles vector DB operations |
| `rag_service.py` | `backend/app/services/` | Processes user queries |
| `confidence.py` | `backend/app/services/` | Calculates answer confidence |
| `main.py` | `backend/app/` | FastAPI application entry point |
| `chat.py` | `backend/app/routers/` | Chat API endpoint |
| `ingest.py` | `backend/app/routers/` | Document upload API endpoint |
| `config.py` | `backend/app/` | Environment configuration |

---

## PART 3: Setting Up for Trade Show Prototype

### Step 1: Configure Environment Variables

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your real API keys:
   ```
   # OpenAI (for embeddings and chat)
   OPENAI_API_KEY=sk-your-real-key-here
   OPENAI_MODEL=gpt-4o
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   # Pinecone (vector database)
   PINECONE_API_KEY=pc-your-real-key-here
   PINECONE_INDEX=danfoss-products
   PINECONE_ENVIRONMENT=us-east-1

   # Security
   JWT_SECRET=change-this-to-random-string

   # Allow your demo laptop
   CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
   ```

### Step 2: Install Dependencies

```bash
cd danfoss_rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r backend/requirements.txt
```

### Step 3: Ingest Your Documents

```bash
# Add your company documents to the data folder first
python scripts/ingest_documents.py --dir ./data --verbose
```

### Step 4: Start the Backend Server

```bash
cd backend
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

The API is now running at: `http://localhost:8000`

### Step 5: Test the Demo Page

Open in browser: `http://localhost:8000/static/demo.html`

You should see the product catalog page with a chat widget in the bottom-right corner.

---

## PART 4: Adding the Chatbot to Your Website

### Option A: Simple Embed (Recommended for Trade Show)

Add this single line before the closing `</body>` tag on any HTML page:

```html
<script
    src="http://localhost:8000/static/danfoss-chat-widget.js"
    data-api-url="http://localhost:8000"
    data-primary-color="#E2000F"
    data-title="Product Assistant">
</script>
```

**That's it!** The widget will appear as a floating button in the bottom-right corner.

### Option B: Full HTML Page Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Company - Products</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        h1 { color: #E2000F; }
    </style>
</head>
<body>
    <h1>Welcome to Our Products</h1>
    <p>Browse our catalog and use the chat assistant for help finding parts.</p>

    <!-- Your existing website content here -->

    <!-- ADD THIS SCRIPT TO ENABLE THE CHATBOT -->
    <script
        src="http://localhost:8000/static/danfoss-chat-widget.js"
        data-api-url="http://localhost:8000"
        data-primary-color="#E2000F"
        data-title="Product Assistant">
    </script>
</body>
</html>
```

### Widget Configuration Options

| Attribute | Default | Description |
|-----------|---------|-------------|
| `data-api-url` | `http://localhost:8000` | Your backend API URL |
| `data-primary-color` | `#E2000F` | Button/header color (use your brand color) |
| `data-title` | `Danfoss Product Assistant` | Title shown in chat header |

---

## PART 5: Deploying to Company Systems (Production)

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Website  │────▶│   Backend API   │────▶│   Pinecone DB   │
│   (HTML + JS)   │     │   (FastAPI)     │     │   (Cloud)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   OpenAI API    │
                        │   (Embeddings   │
                        │    + Chat)      │
                        └─────────────────┘
```

### Deployment Options

**Option 1: Docker (Recommended)**
```bash
cd danfoss_rag
docker-compose up -d
```

This starts the backend on port 8000.

**Option 2: Cloud Hosting (Azure, AWS, etc.)**

1. Deploy the `backend/` folder as a Python web app
2. Set environment variables in your cloud provider
3. Update `CORS_ORIGINS` to include your website domain
4. Update the widget script `data-api-url` to your cloud URL

### Production Checklist

- [ ] Replace all API keys with production keys
- [ ] Set `DEBUG=false` in `.env`
- [ ] Change `JWT_SECRET` to a long random string
- [ ] Update `CORS_ORIGINS` with your actual website domain(s)
- [ ] Use HTTPS for the backend URL
- [ ] Test document ingestion with real company data
- [ ] Verify chat responses are accurate

### Update Widget Script for Production

```html
<script
    src="https://your-api-server.com/static/danfoss-chat-widget.js"
    data-api-url="https://your-api-server.com"
    data-primary-color="#E2000F"
    data-title="Product Assistant">
</script>
```

---

## PART 6: Quick Reference Commands

### Document Management
```bash
# Ingest all documents
python scripts/ingest_documents.py --dir ./data

# Ingest single file
python scripts/ingest_documents.py --file ./data/manual.pdf

# Clear and re-ingest
python scripts/ingest_documents.py --dir ./data --clear-existing

# Check what would be ingested
python scripts/ingest_documents.py --dir ./data --dry-run
```

### Server Management
```bash
# Start development server
cd backend
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Start with Docker
docker-compose up -d

# View logs
docker-compose logs -f
```

### Testing
```bash
# Health check
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you have?"}'
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Connection refused" | Make sure the backend server is running |
| "CORS error" | Add your website URL to `CORS_ORIGINS` in `.env` |
| "No results found" | Run the ingestion script to add documents |
| Chat widget not appearing | Check browser console for JavaScript errors |
| Low confidence answers | Add more relevant documents to the `data/` folder |
