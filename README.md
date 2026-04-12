# 🚀 Offline Hybrid RAG Chat Assistant

> A **low-latency offline Retrieval-Augmented Generation (RAG) chatbot** built using **FastAPI, FAISS, BM25, Redis, and Ollama (Mistral)** — featuring hybrid retrieval, multi-level caching, conversational memory, and GPU acceleration.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Upload & Query Flow](#upload--query-flow)
- [Logging](#logging)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Why This Project Matters](#why-this-project-matters)
- [Author](#author)

---

## Overview

This chatbot allows users to upload PDF documents, ask natural language questions, and receive context-aware, document-grounded answers — all **running fully offline with GPU acceleration**.

| Capability | Detail |
|---|---|
| Document ingestion | PDF upload → chunk → embed → index |
| Retrieval | Hybrid search (FAISS vector + BM25 keyword) |
| Caching | Exact cache + semantic cache via Redis |
| Memory | Conversation history for follow-up queries |
| Generation | Local LLM via Ollama (Mistral) |
| Interface | Streamlit chat UI with streaming output |

The system simulates **real-world enterprise RAG architectures**.

---

## System Architecture

```
┌──────────────────────────────────────────┐
│         User Layer (Streamlit UI)        │
│  Upload PDF · Ask questions · Chat       │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           API Layer (FastAPI)            │
│  /ingest  ·  /query  ·  Async endpoints  │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│             Cache Layer (Redis)          │
│  Exact Cache  ·  Semantic Cache (FAISS)  │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│        Hybrid Retrieval Layer            │
│  FAISS (semantic)  ·  BM25 (keyword)     │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│        LLM Generation Layer              │
│  Ollama (Mistral)  ·  GPU accelerated    │
└──────────────────────────────────────────┘
```

### Processing Flow

| Step | Action |
|---|---|
| 1 | Upload PDF document |
| 2 | Extract text from PDF |
| 3 | Chunk text dynamically |
| 4 | Generate embeddings (all-MiniLM-L6-v2) |
| 5 | Store vectors in FAISS |
| 6 | Build BM25 keyword index |
| 7 | Generate document summary (once) |
| 8 | Receive user query |
| 9 | Check exact + semantic cache |
| 10 | Retrieve relevant chunks (hybrid) |
| 11 | Build prompt with context + memory |
| 12 | Generate and stream LLM response |
| 13 | Cache response for future reuse |

---

## Project Structure

```
RAG-App/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entrypoint
│   ├── query.py              # Query processing pipeline
│   ├── ingest.py             # Document ingestion pipeline
│   ├── cache.py              # Redis cache (exact + semantic)
│   ├── memory.py             # Conversation memory
│   ├── vector_store.py       # FAISS vector database
│   ├── bm25_store.py         # BM25 keyword index
│   ├── chunker.py            # Dynamic text chunking
│   ├── logger.py             # Structured logging
│   └── document_manager.py   # Document hash tracking
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
│   └── rag_logs.txt          # Performance + query logs
│
├── streamlit_app.py          # Streamlit chat UI
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## Key Features

### 📄 Document Processing

- PDF text extraction
- Dynamic chunk sizing based on document length
- Overlapping chunk generation for context continuity
- Document hashing (single ingestion per file)
- Automatic document summary generation

### 🔍 Hybrid Retrieval (FAISS + BM25)

| Retriever | Method | Strength |
|---|---|---|
| **FAISS** | Embedding-based vector search | Semantic similarity |
| **BM25** | Token-based TF-IDF ranking | Exact keyword matching |
| **Combined** | Score fusion | Higher precision, fewer hallucinations |

### ⚡ Multi-Level Caching

| Level | Key | Trigger | Result |
|---|---|---|---|
| **L1 — Exact Cache** | Raw query string | Identical repeated query | Instant O(1) response |
| **L2 — Semantic Cache** | Query embedding | Similar query (≥ 0.85 cosine) | Fast cached response |

```
TTL = 24 hours   (auto-eviction to prevent memory overflow)
```

### 🧠 Conversational Memory

Stores recent conversation history to enable:

- **Query expansion** — contextualises follow-ups automatically
- **Context continuity** — answers build on prior exchanges
- **Follow-up understanding** — no need to repeat context

**Example:**
```
User:   "What is supervised learning?"
User:   "Give examples."

System internally expands to:
        "Give examples of supervised learning."
```

### 📊 Dynamic Chunking

Chunk size adjusts automatically based on document length:

| Document Size | Chunk Size |
|---|---|
| Small | 300 characters |
| Medium | 500 characters |
| Large | 800 characters |

### 🧾 Prompt Design

Every prompt is assembled from four layers for maximum accuracy:

```
┌─────────────────────────┐
│   Conversation History  │  ← Maintains continuity
├─────────────────────────┤
│   Document Summary      │  ← Global document context
├─────────────────────────┤
│   Retrieved Chunks      │  ← Specific relevant content
├─────────────────────────┤
│   User Query            │  ← Current question
└─────────────────────────┘
```

> **Fallback:** If no relevant context is found, the LLM answers independently and appends: *"NOTE: This response is generated entirely by AI without document context."*

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
- NVIDIA GPU (optional but recommended)

---

### Step 1 — Clone Repository

```bash
git clone <your-repo-url>
cd RAG-App
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

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output:
```
NVIDIA GeForce GTX 1650  (or your available GPU)
```

### Step 4 — Start Redis (Docker)

Start Docker Desktop first, then run:

```bash
docker start vibrant_fermi
```

Optional — clear existing cache:

```bash
docker exec -it vibrant_fermi redis-cli FLUSHALL
```

Verify the container is running:

```bash
docker ps
```

### Step 5 — Start Ollama

```bash
ollama serve
```

Verify Mistral is available:

```bash
ollama list
```

If `mistral:latest` is missing:

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

Open a new terminal, then:

```bash
venv_gpu\Scripts\activate
streamlit run streamlit_app.py
```

The UI will open automatically in your browser.

---

## Upload & Query Flow

```
1. Upload PDF via Streamlit UI
        │
        ▼
2. Wait for background processing (chunking + indexing)
        │
        ▼
3. Ask a natural language question
        │
        ▼
4. Continue the conversation with follow-up questions
        │
        ▼
5. Repeated or similar queries return instantly from cache
```

---

## Logging

All query performance metrics are logged to `logs/rag_logs.txt`.

**Logged fields:**
- Query text
- Cache hit/miss status
- Retrieval time
- LLM generation time
- Total response time

**Example log entry:**

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
| Single-document processing | Cannot query across multiple PDFs |
| No reranker | Retrieval precision could be higher |
| Semantic cache grows indefinitely | Memory usage increases over time |
| Large PDFs increase ingestion time | Slower startup for heavy documents |

---

## Future Enhancements

- [ ] Cross-Encoder reranker for improved retrieval precision
- [ ] Multi-document support
- [ ] Async ingestion pipeline
- [ ] Streaming token output
- [ ] Semantic cache eviction policy
- [ ] Persistent cross-session chat memory
- [ ] Multi-user support
- [ ] Docker Compose deployment
- [ ] Cloud-ready architecture

---

## Why This Project Matters

This project demonstrates production-relevant skills used across enterprise AI:

| Skill | Application |
|---|---|
| Real-world RAG architecture | End-to-end document Q&A pipeline |
| Hybrid retrieval systems | FAISS + BM25 score fusion |
| Memory-aware AI workflows | Conversational context continuity |
| Production-style caching | Multi-level Redis + FAISS cache |
| GPU acceleration pipelines | PyTorch CUDA for fast inference |

**Used in industries like:**
- Enterprise AI assistants
- Knowledge retrieval systems
- AI copilots
- Intelligent search engines

---

## Author

**Mohd Afrid**  
*AI / ML Engineer*

Focused on building scalable AI systems and real-world ML applications.

---

*Built for speed. Designed for scale. Runs offline.*
