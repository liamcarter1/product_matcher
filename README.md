# Danfoss RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) system for Danfoss distributors to look up Danfoss part numbers from competitor part numbers and query product performance characteristics.

## Features

- **Part Cross-Reference Lookup**: Find Danfoss equivalent parts for competitor products
- **Fuzzy Part Number Matching**: Handles variations in part number formatting
- **Product Specifications Query**: Get technical details for any product
- **Confidence Scoring**: Each response includes a confidence indicator
- **Embeddable Chat Widget**: Easy integration into any website
- **Multi-format Document Support**: PDF, Excel, and CSV ingestion
- **Conversation Memory**: Maintains context across messages

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Distributor Website                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  <script src="danfoss-chat-widget.js"></script>                  │    │
│  │  Popup Chatbot (Danfoss Red #E2000F)                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ POST /api/chat │  │ POST /api/auth │  │ POST /api/ingest           │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                    │                                    │
        ┌───────────┴───────────┐                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐
│   Pinecone   │      │   OpenAI     │      │  In-Memory           │
│   Vector DB  │      │   GPT-4o     │      │  Session Store       │
└──────────────┘      └──────────────┘      └──────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Pinecone API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd danfoss_rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start the backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

6. **Open the demo page**
   Open `frontend/demo.html` in your browser

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### Chat

```bash
POST /api/chat
Content-Type: application/json

{
    "message": "What Danfoss part replaces Siemens 6SL3210?",
    "session_id": "optional-session-id"
}

Response:
{
    "response": "Danfoss part FC-302P7K5...",
    "confidence": 85.5,
    "confidence_level": "high",
    "session_id": "abc123",
    "sources": [{"file": "parts.xlsx", "type": "excel"}]
}
```

#### Document Ingestion

```bash
POST /api/ingest
Content-Type: multipart/form-data

file: <your-document.xlsx>

Response:
{
    "status": "success",
    "documents_added": 150,
    "file": "parts_crossref.xlsx",
    "file_type": "xlsx"
}
```

## Document Formats

### Excel/CSV Part Cross-Reference

The system auto-detects columns. Recommended format:

| danfoss_part | competitor_brand | competitor_part | description | voltage | current |
|--------------|------------------|-----------------|-------------|---------|---------|
| ABC-123      | Siemens          | 6SL3210-XXX     | VFD 7.5kW   | 400V    | 15A     |
| DEF-456      | ABB              | ACS580-XXX      | VFD 11kW    | 480V    | 22A     |

### PDF Documents

Upload product guides, manuals, and specification sheets. The system will:
- Extract text from all pages
- Chunk content for efficient retrieval
- Preserve metadata for source attribution

## Widget Integration

Add the chat widget to any website:

```html
<script
    src="https://your-api-domain.com/static/danfoss-chat-widget.js"
    data-api-url="https://your-api-domain.com"
    data-primary-color="#E2000F"
    data-title="Danfoss Product Assistant">
</script>
```

### Configuration Options

| Attribute | Description | Default |
|-----------|-------------|---------|
| `data-api-url` | Backend API URL | `http://localhost:8000` |
| `data-primary-color` | Widget accent color | `#E2000F` (Danfoss Red) |
| `data-title` | Chat header title | `Danfoss Product Assistant` |

## CLI Tools

### Bulk Document Ingestion

```bash
python scripts/ingest_documents.py --dir ./data
python scripts/ingest_documents.py --file sample_parts.xlsx
python scripts/ingest_documents.py --dir ./data --clear-existing
```

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_INDEX` | Pinecone index name | No (default: `danfoss-products`) |
| `JWT_SECRET` | Secret for JWT signing | No (default provided) |
| `CORS_ORIGINS` | Allowed origins | No |

## Confidence Scoring

The system provides confidence scores with each response:

| Level | Score | Indicator |
|-------|-------|-----------|
| High | 80-100% | 🟢 Green |
| Medium | 50-80% | 🟡 Yellow |
| Low | 0-50% | 🔴 Red |

Low confidence responses include a disclaimer suggesting verification with Danfoss technical support.

## Project Structure

```
danfoss_rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── routers/             # API endpoints
│   │   │   ├── chat.py          # Chat endpoint
│   │   │   ├── auth.py          # Authentication
│   │   │   └── ingest.py        # Document ingestion
│   │   ├── services/            # Business logic
│   │   │   ├── rag_service.py   # LangGraph RAG chain
│   │   │   ├── pinecone_service.py
│   │   │   ├── document_loader.py
│   │   │   ├── vision_service.py # GPT-4o vision for nameplate/valve images
│   │   │   ├── skill_loader.py  # Loads hydraulics domain knowledge
│   │   │   └── confidence.py
│   │   └── models/
│   │       └── schemas.py       # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
├── skills/
│   └── hydraulics_engineer.md   # Domain knowledge for valve extraction
├── frontend/
│   ├── danfoss-chat-widget.js   # Embeddable widget
│   ├── danfoss-chat-widget.css  # Widget styles
│   └── demo.html                # Demo page
├── scripts/
│   └── ingest_documents.py      # CLI ingestion tool
├── docker-compose.yml
├── .env.example
└── README.md
```

## Security Considerations

- Store API keys securely (never commit `.env` files)
- Use strong JWT secrets in production
- Configure CORS appropriately for your domains
- Consider rate limiting for production deployments
- Enable HTTPS in production

## Support

For issues with this RAG system, please open a GitHub issue.

For Danfoss product support, contact Danfoss technical support directly.

## License

Proprietary - For authorized Danfoss distributors only.
