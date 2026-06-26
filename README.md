# 🤖 RAG Document Assistant

**IR INFOTECH — Round 2 Assignment | AI/ML Intern**

An AI-powered Document Assistant that lets you upload PDF documents and ask questions about their content in natural language. Built with **FastAPI**, **FAISS**, **HuggingFace Sentence Transformers**, and **Groq LLM** — with production-grade features including streaming responses, API key authentication, multi-document support, and Docker deployment.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Bonus Features](#bonus-features)
- [Design Decisions](#design-decisions)

---

## ✅ Features

### Core (Required)
| Feature | Implementation |
|---|---|
| **PDF Upload API** | `POST /documents/upload` — accepts one or more PDFs |
| **Text Extraction** | PyMuPDF (`fitz`) — page-aware extraction with `[Page N]` markers |
| **Document Chunking** | LangChain `RecursiveCharacterTextSplitter` (size=500, overlap=50) |
| **Embedding Generation** | HuggingFace `all-MiniLM-L6-v2` via `sentence-transformers` |
| **FAISS Integration** | `IndexFlatIP` with L2-normalised vectors (cosine similarity) |
| **Question Answering** | Groq LLM (`llama-3.3-70b-versatile`) with retrieved chunk context |
| **Source References** | Every answer includes `document_id`, `filename`, `chunk_index`, and content preview |
| **Chat History API** | In-memory session store — get and clear history per session |

### Bonus
| Feature | Implementation |
|---|---|
| **Multi-document Upload** | `POST /documents/upload` accepts `List[UploadFile]`; each file processed independently; failed files reported without stopping others |
| **Dockerization** | `Dockerfile` + `docker-compose.yml` with volume mounts for persistence |
| **API Key Authentication** | `X-API-Key` header enforced on all `/documents` and `/chat` routes via FastAPI dependency |
| **Streaming Responses** | `POST /chat/ask/stream` — Server-Sent Events (SSE) with word-level buffering for smooth output |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI 0.111 + Uvicorn |
| **PDF Parsing** | PyMuPDF 1.24 |
| **Text Chunking** | LangChain Text Splitters 0.2 |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim, HuggingFace) |
| **Vector Store** | FAISS `IndexFlatIP` (persisted to disk) |
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **Validation** | Pydantic v2 + pydantic-settings |
| **Container** | Docker + Docker Compose |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT  (Swagger / curl / frontend)          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  HTTP  (X-API-Key header)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                        │
│                                                                     │
│   ┌─────────────────────┐        ┌──────────────────────────────┐   │
│   │  /documents router  │        │       /chat router           │   │
│   │                     │        │                              │   │
│   │  POST /upload       │        │  POST /ask          (sync)   │   │
│   │  GET  /             │        │  POST /ask/stream   (SSE)    │   │
│   │  DELETE /{id}       │        │  GET  /history/{id}          │   │
│   └────────┬────────────┘        │  DELETE /history/{id}        │   │
│            │                     └────────────┬─────────────────┘   │
└────────────┼────────────────────────────────────────────────────────┘
             │                                  │
     ┌───────▼───────┐                 ┌────────▼────────┐
     │  pdf_service  │                 │  rag_service    │
     │               │                 │                 │
     │ • PyMuPDF     │                 │ • FAISS index   │
     │ • LangChain   │──── chunks ────>│ • embed_query() │
     │   Splitter    │                 │ • Groq LLM call │
     └───────────────┘                 └────────┬────────┘
                                                │
                              ┌─────────────────┴───────────────┐
                              │                                 │
                    ┌─────────▼────────┐             ┌──────────▼────────┐
                    │ embedding_service│             │   Groq API        │
                    │                  │             │                   │
                    │ all-MiniLM-L6-v2 │             │ llama-3.3-70b-    │
                    │ (384-dim vectors)│             │   versatile       │
                    └──────────────────┘             └───────────────────┘
                              │
                    ┌─────────▼────────┐
                    │  FAISS on disk   │
                    │                  │
                    │  index.faiss     │
                    │  metadata.json   │
                    └──────────────────┘
```

### RAG Pipeline Flow

```
PDF Upload
  │
  ├─ PyMuPDF → raw text (page-aware)
  ├─ LangChain splitter → chunks (500 chars, 50 overlap)
  ├─ all-MiniLM-L6-v2 → 384-dim embeddings
  ├─ L2 normalise → cosine similarity via inner product
  └─ FAISS IndexFlatIP → stored + persisted to disk

Question
  │
  ├─ embed_query() → 384-dim vector (normalised)
  ├─ FAISS.search(top_k=4) → nearest chunk indices
  ├─ Metadata lookup → {content, filename, chunk_index}
  ├─ Build context string (multi-source)
  └─ Groq LLM → natural language answer + source refs
```

---

## 📁 Project Structure

```
rag_document_assistant/
│
├── app/
│   ├── main.py                  # FastAPI app, CORS, middleware, routers
│   ├── config.py                # pydantic-settings (reads .env)
│   │
│   ├── routers/
│   │   ├── documents.py         # Upload, list, delete endpoints
│   │   └── chat.py              # Ask, stream, history endpoints
│   │
│   ├── services/
│   │   ├── rag_service.py       # FAISS index + Groq LLM integration
│   │   ├── pdf_service.py       # PyMuPDF extraction + LangChain chunking
│   │   ├── embedding_service.py # HuggingFace sentence-transformers
│   │   └── chat_service.py      # In-memory session store
│   │
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   │
│   └── utils/
│       ├── auth.py              # X-API-Key dependency
│       └── logger.py            # Structured logger
│
├── data/
│   └── faiss_index/
│       ├── index.faiss          # Persisted FAISS index
│       └── metadata.json        # Chunk metadata (content, doc_id, filename)
│
├── uploads/                     # Uploaded PDF files
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd rag_document_assistant

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
GROQ_API_KEY=your_groq_api_key_here
API_KEY=your_secret_api_key_here
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8004
```

Open **http://127.0.0.1:8004/docs** — the interactive Swagger UI.

> **Authorize first:** Click the 🔒 **Authorize** button in Swagger and enter your `API_KEY` value.

---

### 🐳 Docker

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

The API will be available at **http://localhost:8000**.

PDF uploads and the FAISS index are persisted via Docker volumes — data survives container restarts.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `API_KEY` | *(required)* | Your custom API key for endpoint auth |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | `./data/faiss_index/index.faiss` | FAISS index file path |
| `FAISS_META_PATH` | `./data/faiss_index/metadata.json` | Chunk metadata file path |
| `UPLOAD_DIR` | `./uploads` | Directory to save uploaded PDFs |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of chunks retrieved per query |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 📡 API Reference

All `/documents` and `/chat` endpoints require the header:
```
X-API-Key: <your_api_key>
```

Public endpoints (no auth): `GET /`, `GET /health`

---

### Documents

#### `POST /documents/upload`
Upload one or more PDF files.

```bash
curl -X POST http://localhost:8004/documents/upload \
  -H "X-API-Key: my-secret-api-key" \
  -F "files=@report.pdf" \
  -F "files=@notes.pdf"
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "3fa85f64-...",
      "filename": "report.pdf",
      "total_chunks": 24,
      "status": "success",
      "message": "Uploaded and indexed successfully with 24 chunks."
    }
  ],
  "total_uploaded": 1,
  "total_failed": 0
}
```

---

#### `GET /documents/`
List all indexed documents.

```bash
curl http://localhost:8004/documents/ \
  -H "X-API-Key: my-secret-api-key"
```

---

#### `DELETE /documents/{document_id}`
Delete a document and all its chunks from the FAISS index.

```bash
curl -X DELETE http://localhost:8004/documents/3fa85f64-... \
  -H "X-API-Key: my-secret-api-key"
```

---

### Chat

#### `POST /chat/ask`
Ask a question and get a complete answer.

```bash
curl -X POST http://localhost:8004/chat/ask \
  -H "X-API-Key: my-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain how LSTM works",
    "session_id": "user-001"
  }'
```

**Response:**
```json
{
  "answer": "LSTM (Long Short-Term Memory) is a type of recurrent neural network...",
  "sources": [
    {
      "document_id": "3fa85f64-...",
      "filename": "RNNpart2.pdf",
      "chunk_index": 4,
      "content_preview": "The LSTM architecture introduces three gates..."
    }
  ],
  "session_id": "user-001",
  "question": "Explain how LSTM works"
}
```

---

#### `POST /chat/ask/stream`
Stream the answer token-by-token via Server-Sent Events (SSE).

```bash
curl -X POST http://localhost:8004/chat/ask/stream \
  -H "X-API-Key: my-secret-api-key" \
  -H "Content-Type: application/json" \
  -N \
  -d '{"question": "Explain how LSTM works", "session_id": "user-001"}'
```

**SSE stream format:**
```
data: LSTM

data: networks

data: work

data: by

...

data: [DONE]
```

Words are buffered so each `data:` event is a complete word, not a raw subword token.

---

#### `GET /chat/history/{session_id}`
Retrieve full conversation history for a session.

```bash
curl http://localhost:8004/chat/history/user-001 \
  -H "X-API-Key: my-secret-api-key"
```

---

#### `DELETE /chat/history/{session_id}`
Clear conversation history for a session.

```bash
curl -X DELETE http://localhost:8004/chat/history/user-001 \
  -H "X-API-Key: my-secret-api-key"
```

---

#### `GET /health`
Health check — public, no auth needed.

```bash
curl http://localhost:8004/health
# {"status": "ok", "service": "RAG Document Assistant", "version": "1.0.0"}
```

---

## 🌟 Bonus Features

### 1. Multi-Document Upload
`POST /documents/upload` accepts a `files` field with multiple PDFs. Each is processed independently. If one fails (e.g. scanned/image-only PDF), it is marked `"status": "failed"` in the response without blocking the others.

### 2. Dockerization
- **`Dockerfile`** — `python:3.11-slim` base, layer-cached pip install, non-root friendly.
- **`docker-compose.yml`** — single command `docker compose up --build`.
- Volumes `./uploads` and `./data` are mounted so uploaded PDFs and the FAISS index survive container restarts.

### 3. API Key Authentication
Every protected endpoint requires `X-API-Key` in the request header. Implemented using FastAPI's `APIKeyHeader` security scheme — the key appears as a proper **🔒 Authorize** button in the Swagger UI. Missing or wrong keys return `HTTP 401`.

### 4. Streaming Responses
`POST /chat/ask/stream` uses Groq's streaming API (`stream=True`) and wraps the output in a `StreamingResponse` (SSE format). Tokens are word-buffered — yielded only at space/newline boundaries — so the browser receives complete words, not subword fragments.

---

## 💡 Design Decisions

| Decision | Rationale |
|---|---|
| **FAISS over ChromaDB** | No external server needed; `IndexFlatIP` with L2-normalised vectors gives exact cosine similarity; index persisted as flat files |
| **`all-MiniLM-L6-v2`** | 384-dim, fast to embed, strong semantic quality for a 80MB model |
| **`llama-3.3-70b-versatile` on Groq** | Free tier, sub-2s latency, strong instruction following |
| **LangChain `RecursiveCharacterTextSplitter`** | Respects paragraph → sentence → word hierarchy; better context preservation than fixed-size splits |
| **Router-level auth** | `dependencies=[Depends(verify_api_key)]` on `APIRouter` protects all child routes automatically without touching each endpoint |
| **Word-buffered SSE** | Raw Groq tokens are sub-word pieces; buffering to word boundaries makes streaming readable in the browser |
| **`lru_cache` on heavy objects** | Embedding model and Groq client are expensive to init — cached across requests, loaded once per process |
