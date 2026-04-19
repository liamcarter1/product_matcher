# CLAUDE.md - Project Guidelines for RAG Systems

## Project Overview

This is a **Retrieval-Augmented Generation (RAG)** chatbot built with:
- **Backend**: FastAPI + Python 3.11+
- **Vector Database**: Pinecone
- **LLM Provider**: OpenAI (GPT-4o)
- **Embeddings**: OpenAI text-embedding-3-small
- **Frontend**: Vanilla JavaScript widget (embeddable)

---

## Architecture Patterns

### RAG Pipeline Flow
```
User Query → Embedding → Vector Search → Context Retrieval → LLM Prompt → Response
```

### Key Services
| Service | Responsibility |
|---------|----------------|
| `document_loader.py` | Load and chunk documents (PDF, Excel, CSV) |
| `pinecone_service.py` | Vector storage and similarity search |
| `rag_service.py` | Orchestrate retrieval and LLM calls |
| `vision_service.py` | GPT-4o vision for nameplate and valve image extraction |
| `skill_loader.py` | Load hydraulics domain knowledge from `skills/` into prompts |
| `confidence.py` | Score answer confidence based on retrieval |

### Domain Knowledge
- `skills/hydraulics_engineer.md` contains hydraulic valve domain knowledge (spool mappings, ordering code structure, unit normalisation)
- Loaded once at startup by `skill_loader.py`, split into sections
- Injected into system prompts conditionally by query type to minimise token cost
- `part_lookup` queries get spool mapping + units + failure modes (~1000 tokens)
- `specification_query` queries get spec fields + units (~600 tokens)
- `general_question` queries get no domain injection (0 extra tokens)

### Chunking Strategy
- **Chunk size**: 1000 characters
- **Overlap**: 200 characters
- **Separator**: Sentence boundaries preferred
- **Metadata**: Always include source file, page number, chunk index

---

## Security Requirements

### API Security
- [ ] Never expose API keys in client-side code
- [ ] Use environment variables for all secrets (`.env` file)
- [ ] Implement rate limiting on chat endpoints
- [ ] Validate and sanitize all user inputs
- [ ] Use HTTPS in production

### Environment Variables (Required)
```
OPENAI_API_KEY=sk-...        # Never commit this
PINECONE_API_KEY=pc-...      # Never commit this
JWT_SECRET=<random-string>   # For session management
CORS_ORIGINS=["https://..."] # Restrict to known domains
```

### Input Validation
- Sanitize user messages before processing
- Limit message length (recommended: 2000 characters max)
- Strip HTML/script tags from inputs
- Validate file uploads (type, size, content)

### Data Protection
- Do not log full user queries in production
- Do not store PII in vector metadata
- Implement data retention policies
- Consider encryption at rest for sensitive documents

---

## LangChain Best Practices

### Document Loading
```python
# Always use appropriate loader for file type
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# Add metadata during loading
loader = PyPDFLoader(file_path)
docs = loader.load()
for doc in docs:
    doc.metadata["source"] = file_path
    doc.metadata["ingested_at"] = datetime.utcnow().isoformat()
```

### Text Splitting
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### Embedding Best Practices
- Batch embeddings (100 at a time) to avoid rate limits
- Cache embeddings when possible
- Use consistent embedding model across ingestion and query
- Handle API failures with exponential backoff

### Retrieval Configuration
```python
# Recommended retrieval settings
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,  # Number of chunks to retrieve
    }
)
```

---

## Python/FastAPI Standards

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app factory
│   ├── config.py         # Pydantic settings
│   ├── routers/          # API endpoints
│   │   ├── chat.py
│   │   └── ingest.py
│   └── services/         # Business logic
│       ├── document_loader.py
│       ├── pinecone_service.py
│       └── rag_service.py
├── requirements.txt
└── tests/
```

### Dependency Management
- Pin all dependencies with versions in `requirements.txt`
- Use virtual environments (`venv` or `conda`)
- Separate dev dependencies from production

### Error Handling
```python
# Always handle external API failures gracefully
try:
    response = await openai_client.chat.completions.create(...)
except OpenAIError as e:
    logger.error(f"OpenAI API error: {e}")
    raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
```

### Async Patterns
- Use `async/await` for I/O-bound operations
- Use connection pooling for database connections
- Implement proper timeout handling

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Log levels:
# DEBUG - Detailed debugging (not in production)
# INFO - General operational events
# WARNING - Unexpected but handled situations
# ERROR - Failures requiring attention
```

---

## Vector Database (Pinecone) Guidelines

### Index Configuration
- **Dimensions**: 1536 (for text-embedding-3-small)
- **Metric**: Cosine similarity
- **Pod type**: Starter (dev) or s1 (production)

### Namespace Strategy
| Namespace | Content |
|-----------|---------|
| `products` | Parts cross-reference, product specs |
| `guides` | User manuals, technical docs |
| `support` | FAQ, troubleshooting guides |

### Vector ID Generation
```python
import hashlib

def generate_vector_id(text: str, source: str) -> str:
    """Generate deterministic ID for deduplication."""
    content = f"{source}:{text}"
    return hashlib.md5(content.encode()).hexdigest()
```

### Metadata Best Practices
- Keep metadata small (< 40KB per vector)
- Include: source, page, chunk_index, text_preview
- Avoid: full text content, binary data, PII

---

## Testing Requirements

### Unit Tests
- Test document loaders with sample files
- Test chunking logic with edge cases
- Mock external APIs (OpenAI, Pinecone)

### Integration Tests
- Test full RAG pipeline with test documents
- Verify CORS configuration
- Test error handling paths

### Test Commands
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_rag_service.py -v
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] CORS origins restricted to production domains
- [ ] Rate limiting enabled
- [ ] Logging configured (no sensitive data)
- [ ] Health check endpoint working

### Docker
```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as builder
# ... build dependencies

FROM python:3.11-slim
# ... runtime only
```

### Monitoring
- Monitor API latency (target: < 3s for chat)
- Track token usage for cost management
- Alert on error rate spikes
- Monitor vector database query latency

---

## Common Pitfalls to Avoid

1. **Don't** hardcode API keys or secrets
2. **Don't** skip input validation on user messages
3. **Don't** use synchronous calls for API requests
4. **Don't** ignore rate limits on embedding APIs
5. **Don't** store full document text in vector metadata
6. **Don't** deploy without HTTPS
7. **Don't** log user queries containing sensitive information
8. **Don't** use `*` for CORS origins in production

---

## Quick Reference

### Start Development Server
```bash
cd backend
uvicorn app.main:create_app --factory --reload --port 8000
```

### Ingest Documents
```bash
python scripts/ingest_documents.py --dir ./data --verbose
```

### Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you have?"}'
```
