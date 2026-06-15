# 🧠 Context Aware Chat Assistant

> A production-style **Offline Retrieval-Augmented Generation (RAG) Chatbot** with hybrid search, multi-level caching, GPU acceleration, and conversational memory — built to simulate real-world enterprise AI architectures.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [File-by-File Explanation](#file-by-file-explanation)
- [Caching Strategy](#caching-strategy)
- [Latency Optimization](#latency-optimization)
- [Code Quality & Design](#code-quality--design)
- [System Design Flow](#system-design-flow)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Supported File Types](#supported-file-types)
- [Example Query Flow](#example-query-flow)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

---

## Overview

This chatbot allows users to upload documents, ask natural language questions, and receive fast, context-aware, document-grounded answers — running **fully offline with GPU acceleration**.

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Caching | Redis |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Ollama — Mistral |
| GPU Support | CUDA (PyTorch) |

**Supported document types:** `.pdf` · `.txt` · `.md`

---

## Features

| Feature | Description |
|---|---|
| **Hybrid Search** | FAISS semantic search + BM25 keyword ranking |
| **Multi-Level Caching** | Exact, semantic, and document hash caching |
| **Adaptive Context Filtering** | Dynamic similarity threshold removes low-relevance chunks |
| **Dynamic Chunking** | Chunk size adapts to document length |
| **Conversational Memory** | Stores history for contextual follow-up queries |
| **GPU Embedding Support** | CUDA-accelerated embedding generation |
| **Document Deduplication** | File hash prevents re-ingesting the same document |
| **Summary Injection** | Global document summary added to every prompt |
| **Processing Time Display** | Real-time latency shown in the UI |
| **Streaming-ready Architecture** | Designed for token streaming output |

---

## System Architecture

```
┌──────────────────────────────────────────┐
│         User Layer (Streamlit UI)        │
│  Upload docs · Chat · View latency       │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           API Layer (FastAPI)            │
│  /ingest  ·  /query  ·  Async handlers   │
└───────────┬──────────────────────────────┘
            │
            ├──▶ Exact Cache (Redis)
            │
            ├──▶ Semantic Cache (FAISS)
            │
            ├──▶ Hybrid Retrieval
            │       ├── FAISS  (semantic similarity)
            │       └── BM25   (keyword ranking)
            │
            ├──▶ Adaptive Context Filtering
            ├──▶ Memory Injection
            └──▶ Summary Injection
                    │
┌───────────────────▼──────────────────────┐
│        LLM Generation (Ollama/Mistral)   │
│  GPU accelerated  ·  Streaming-ready     │
└───────────────────┬──────────────────────┘
                    │
              Response to User
```

---

## Project Structure

```
RAG-App/
│
├── app/
│   ├── main.py               # FastAPI entrypoint & API endpoints
│   ├── ingest.py             # Document ingestion pipeline
│   ├── query.py              # Hybrid retrieval & response generation
│   ├── chunker.py            # Dynamic text chunking
│   ├── cache.py              # Multi-level caching (Exact + Semantic)
│   ├── vector_store.py       # FAISS index management
│   ├── bm25_store.py         # BM25 keyword index
│   ├── model_loader.py       # Embedding model + GPU configuration
│   ├── document_manager.py   # File hash deduplication
│   ├── memory.py             # Conversational memory
│   ├── logger.py             # Structured logging & performance timing
│   └── utils.py              # Text extraction (PDF, TXT, MD)
│
├── data/
│   └── summary.txt           # Generated document summary
│
├── models/
│   ├── faiss_index.bin       # FAISS vector index
│   ├── docs.pkl              # Stored document chunks
│   ├── cache_index.bin       # Semantic cache index
│   └── cache_data.pkl        # Semantic cache responses
│
├── logs/
│   └── rag_logs.txt          # Query & performance logs
│
├── streamlit_app.py          # Streamlit chat UI
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## File-by-File Explanation

### Backend — `app/`

| File | Responsibility |
|---|---|
| `main.py` | Exposes `/ingest` and `/query` API endpoints |
| `ingest.py` | Orchestrates chunking, embedding, indexing, and summary generation |
| `query.py` | Runs hybrid retrieval, context filtering, prompt building, and LLM calls |
| `chunker.py` | Dynamically sizes chunks based on document length |
| `cache.py` | Implements exact (Redis) and semantic (FAISS) caching |
| `vector_store.py` | Creates, updates, and queries the FAISS vector index |
| `bm25_store.py` | Builds and queries the BM25 keyword ranking index |
| `model_loader.py` | Loads the embedding model and configures GPU (CUDA) acceleration |
| `document_manager.py` | Hashes uploaded files to prevent duplicate ingestion |
| `memory.py` | Stores recent conversation turns for contextual continuity |
| `logger.py` | Logs queries, cache hits, retrieval times, and LLM latency |
| `utils.py` | Extracts raw text from `.pdf`, `.txt`, and `.md` files |

### Frontend

| File | Responsibility |
|---|---|
| `streamlit_app.py` | Chat UI with file upload, query input, and real-time processing stats |

### Config

| File | Responsibility |
|---|---|
| `requirements.txt` | All Python dependencies |
| `.gitignore` | Excludes build artifacts, models, and cache files from version control |
| `.env.example` | Environment variable templates |

---

## Caching Strategy

The system implements **three levels of caching** to minimize redundant computation.

### Level 1 — Exact Cache (Redis)

| Property | Detail |
|---|---|
| Key | Raw query string |
| Trigger | Identical repeated query |
| Result | Instant O(1) response — zero LLM computation |

### Level 2 — Semantic Cache (FAISS)

| Property | Detail |
|---|---|
| Key | Query embedding vector |
| Trigger | Cosine similarity ≥ 0.85 to a cached query |
| Result | Cached response reused without retrieval or LLM call |

**Example — both queries hit the same cache entry:**
```
"what is supervised learning?"
"what is the meaning of supervised learning?"
```

### Level 3 — Document Hash Cache

| Property | Detail |
|---|---|
| Key | SHA hash of uploaded file |
| Trigger | Same file uploaded again |
| Result | Skips re-ingestion — saves chunking, embedding, and indexing time |

```
TTL = 24 hours   (auto-eviction on all Redis entries)
```

---

## Latency Optimization

### Hybrid Retrieval

Combining FAISS and BM25 gives better recall than either alone:

| Retriever | Strength |
|---|---|
| FAISS | Catches semantically similar content |
| BM25 | Catches exact keyword matches |
| Combined | Higher precision, fewer missed chunks |

### Adaptive Context Filtering

Before sending chunks to the LLM, a dynamic similarity threshold filters out low-relevance content:

- ✅ Removes irrelevant chunks
- ✅ Reduces prompt size
- ✅ Faster LLM response

### Dynamic Chunking

| Document Size | Chunk Size |
|---|---|
| Small | 300 characters |
| Medium | 500 characters |
| Large | 800 characters |

### GPU Acceleration

The embedding model runs on CUDA when available, significantly reducing ingestion time for large documents.

### Summary Injection

A document-level summary is generated once during ingestion and injected into every prompt, providing global context without requiring additional retrieval.

### Prompt Structure

```
┌─────────────────────────┐
│   Conversation History  │  ← Contextual continuity
├─────────────────────────┤
│   Document Summary      │  ← Global document context
├─────────────────────────┤
│   Retrieved Chunks      │  ← Filtered relevant content
├─────────────────────────┤
│   User Query            │  ← Current question
└─────────────────────────┘
```

> **Fallback:** If no relevant context is retrieved, the LLM answers independently with the disclaimer: *"NOTE: This response is generated entirely by AI without document context."*

---

## Code Quality & Design

The codebase follows industry-standard software engineering principles:

| Principle | Implementation |
|---|---|
| **Single Responsibility** | Each module handles exactly one concern |
| **Separation of Concerns** | Ingestion, retrieval, caching, and generation are fully decoupled |
| **Modular Architecture** | Components can be swapped or upgraded independently |
| **Reusable Components** | Shared utilities across ingestion and query pipelines |
| **Structured Logging** | Every query logs timing and cache status |
| **Error Handling** | Graceful fallbacks at every pipeline stage |

**Benefits:** Easy debugging · Easy scaling · Maintainable codebase · Interview-friendly structure

---

## System Design Flow

```
 1.  User uploads document
 2.  Text extracted (PDF / TXT / MD)
 3.  Document hash checked → skip if duplicate
 4.  Text chunked dynamically
 5.  Embeddings generated (GPU if available)
 6.  FAISS vector index built / updated
 7.  BM25 keyword index built
 8.  Document summary generated (once)
 9.  User sends query
 10. Exact cache checked → return if hit
 11. Semantic cache checked → return if hit
 12. Hybrid retrieval runs (FAISS + BM25)
 13. Adaptive context filtering applied
 14. Prompt assembled (history + summary + context + query)
 15. LLM generates response (Ollama / Mistral)
 16. Response cached (L1 + L2)
 17. Response streamed to user
```

**Why this design works:**

| Quality | How it's achieved |
|---|---|
| High accuracy | Hybrid retrieval + context filtering |
| Low latency | Three-tier caching + GPU embeddings |
| Scalable | Modular, decoupled components |
| Production-ready | Structured logging, error handling, deduplication |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI (async) |
| **Frontend** | Streamlit |
| **Vector Database** | FAISS |
| **Keyword Search** | BM25 |
| **Cache** | Redis |
| **Embedding Model** | all-MiniLM-L6-v2 (SentenceTransformers) |
| **LLM** | Ollama — Mistral |
| **GPU Acceleration** | PyTorch (CUDA) |
| **Language** | Python 3.11 |
| **Utilities** | NumPy |

---

## Getting Started

> Follow these steps **in order**.

### Prerequisites

- Python 3.11
- Docker Desktop (for Redis)
- [Ollama](https://ollama.com) installed locally
- NVIDIA GPU (optional, recommended)

### Step 1 — Clone Repository

```bash
git clone https://github.com/your-repo/rag-app.git
cd rag-app
```

### Step 2 — Create Virtual Environment

```bash
py -3.11 -m venv venv_gpu
venv_gpu\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Step 4 — Start Redis (Docker)

Start Docker Desktop, then run:

```bash
docker start vibrant_fermi
```

Optional — flush existing cache:

```bash
docker exec -it vibrant_fermi redis-cli FLUSHALL
```

Verify:

```bash
docker ps
```

### Step 5 — Start Ollama

```bash
ollama serve
```

Ensure Mistral is available:

```bash
ollama list
```

If missing:

```bash
ollama pull mistral
```

### Step 6 — Start the Backend

```bash
venv_gpu\Scripts\activate
uvicorn app.main:app --reload
```

API available at: `http://127.0.0.1:8000`

### Step 7 — Start the Chat UI

Open a new terminal:

```bash
venv_gpu\Scripts\activate
streamlit run streamlit_app.py
```

---

## Supported File Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Plain Text | `.txt` |
| Markdown | `.md` |

---

## Example Query Flow

```
User: "What is supervised learning?"
        │
        ▼
  Check Exact Cache ──── HIT ──▶ Return instantly
        │ MISS
        ▼
  Check Semantic Cache ── HIT ──▶ Return cached response
        │ MISS
        ▼
  Hybrid Retrieval (FAISS + BM25)
        │
        ▼
  Adaptive Context Filtering
        │
        ▼
  LLM Generation (Ollama / Mistral)
        │
        ▼
  Cache Response → Stream to User
```

**Sample log entry:**

```
Query     = 'supervised learning'
Cache     = None
Retrieval = 0.18s
LLM       = 1.92s
Total     = 2.10s
```

---

## Current Limitations

| Limitation | Impact |
|---|---|
| Single-document focus | Cannot query across multiple files simultaneously |
| No multi-user support | Shared state across all sessions |
| No cloud deployment | Runs locally only |
| No cross-machine persistence | Vector DB not portable between machines |
| Semantic cache grows unbounded | Memory usage increases without eviction |

---

## Future Enhancements

- [ ] Multi-document support with cross-file retrieval
- [ ] Query token streaming
- [ ] Metadata-based filtering
- [ ] RAG evaluation metrics (e.g. RAGAS)
- [ ] Docker Compose deployment
- [ ] Cloud-ready architecture
- [ ] Multi-user authentication
- [ ] Persistent cross-session chat memory
- [ ] Cross-Encoder reranker for improved precision
- [ ] Vector DB persistence across machines

---

## Conclusion

This project delivers a **production-style offline RAG system** demonstrating:

| Skill | Demonstrated By |
|---|---|
| RAG Architecture | End-to-end document Q&A pipeline |
| Hybrid Retrieval | FAISS + BM25 score fusion |
| Multi-Level Caching | Three-tier Redis + FAISS cache |
| GPU Acceleration | CUDA-based embedding inference |
| Adaptive Filtering | Dynamic similarity thresholds |
| Modular Design | Decoupled, single-responsibility modules |

**Designed for:**
- ✅ AI Engineering Interviews
- ✅ Production Readiness
- ✅ Scalable AI Systems

---

*Built for speed. Designed for scale. Runs offline.*
