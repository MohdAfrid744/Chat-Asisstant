# 🚀 Offline RAG Chat Assistant

> A production-style **Retrieval-Augmented Generation (RAG)** chatbot built for **low latency, high accuracy, and scalable AI-powered document understanding** — featuring hybrid retrieval, multi-level caching, local LLM inference, and a streaming chat UI.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Document Ingestion Pipeline](#document-ingestion-pipeline)
- [Query Processing Pipeline](#query-processing-pipeline)
- [Prompt Strategy](#prompt-strategy)
- [Caching Strategy](#caching-strategy)
- [Performance Optimizations](#performance-optimizations)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Performance Highlights](#performance-highlights)
- [Learning Outcomes](#learning-outcomes)
- [Example Usage](#example-usage)

---

## Overview

This project implements an **offline document-aware AI assistant** that allows users to upload documents, ask natural language questions, retrieve relevant content, generate intelligent responses, and maintain conversation memory.

**Key Objectives:**

| Goal | Approach |
|---|---|
| Fast response generation | Multi-level Redis caching (Exact + Semantic) |
| Reduced redundant LLM calls | Cache-first query resolution |
| Support for similar queries | Embedding-based semantic matching |
| High retrieval accuracy | Hybrid search (FAISS + BM25) |
| Scalable, modular design | Async FastAPI backend |

---

## System Architecture

```
┌──────────────────────────────────────────┐
│           User Layer (Streamlit UI)      │
│  Upload documents · Send queries         │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           API Layer (FastAPI)            │
│  /ingest  ·  /query  ·  Async endpoints  │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│             Cache Layer (Redis)          │
│  Exact Cache  ·  Semantic Cache  ·  TTL  │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│        Hybrid Retrieval Layer            │
│  FAISS (vector)  ·  BM25 (keyword)       │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│        LLM Generation Layer              │
│  Ollama (Mistral)  ·  Streaming output   │
└──────────────────────────────────────────┘
```

---

## Project Structure

```
RAG-App/
│
├── app/
│   ├── chunker.py            # Text chunking logic
│   ├── vector_store.py       # FAISS vector database
│   ├── bm25_store.py         # BM25 keyword index
│   ├── cache.py              # Redis cache (exact + semantic)
│   ├── memory.py             # Conversation memory
│   ├── ingest.py             # Document ingestion pipeline
│   ├── query.py              # Query processing pipeline
│   ├── main.py               # FastAPI app entrypoint
│   ├── document_manager.py   # Document hash tracking
│   └── logger.py             # Structured logging
│
├── data/                     # Uploaded documents
├── models/                   # Local model files
│
├── streamlit_app.py          # Streamlit chat UI
├── requirements.txt
├── test_redis.py
└── README.md
```

---

## Document Ingestion Pipeline

Documents are processed and indexed via the `/ingest` endpoint.

```
Upload Document (/ingest)
        │
        ▼
  Text Extraction (PDF)
        │
        ▼
  Chunking
  ├── Chunk size:  500 characters
  └── Overlap:     50 characters
        │
        ▼
  Embedding Generation (all-MiniLM-L6-v2)
        │
        ▼
  Store in FAISS Vector DB
        │
        ▼
  Build BM25 Keyword Index
        │
        ▼
  Generate Document Summary (optional)
```

---

## Query Processing Pipeline

Incoming queries are resolved through a layered cache-first lookup before hitting the LLM.

```
User Query (/query)
        │
        ▼
  Level 1: Exact Cache ──── HIT ──▶ Return Response
        │ MISS
        ▼
  Level 2: Semantic Cache ── HIT ──▶ Return Response
        │ MISS
        ▼
  Hybrid Retrieval (FAISS + BM25)  ← top_k = 3
        │
        ▼
  Prompt Builder
  ├── Query
  ├── Retrieved context chunks
  └── Conversation history
        │
        ▼
  LLM Generation (Ollama — Mistral)
        │
        ▼
  Cache Response (L1 + L2)
        │
        ▼
  Stream Answer to User
```

---

## Prompt Strategy

The system builds context-aware prompts with a built-in fallback:

| Condition | Behaviour |
|---|---|
| **Relevant context found** | Answer grounded in document content |
| **No context found** | LLM generates independently with disclaimer |

> **Fallback disclaimer:** *"NOTE: The output is generated entirely with AI."*  
> This reduces hallucination confusion and keeps users informed.

---

## Caching Strategy

Multi-level caching is implemented using **Redis** to minimize latency and avoid redundant LLM calls.

### Level 1 — Exact Cache

| Property | Detail |
|---|---|
| Key | Raw query string |
| Value | Cached response |
| Best for | Identical repeated queries |
| Speed | Fastest (O(1) hash lookup) |

### Level 2 — Semantic Cache

| Property | Detail |
|---|---|
| Key | Query embedding vector |
| Value | Cached response |
| Match condition | Cosine similarity ≥ **0.85** |
| Best for | Paraphrased or semantically similar queries |

### Cache Eviction (TTL)

All cache entries automatically expire to prevent memory overflow:

```
TTL = 24 hours
```

---

## Performance Optimizations

| Technique | Benefit |
|---|---|
| Exact Cache (Redis) | Zero-cost repeat queries |
| Semantic Cache (Embedding-based) | Handles paraphrased queries |
| Hybrid Retrieval (FAISS + BM25) | Better recall vs. vector-only search |
| `top_k = 3` retrieval | Small, focused prompts |
| Background document processing | Non-blocking ingestion |
| Streaming response output | Lower perceived latency |
| Memory-based context handling | Coherent multi-turn conversations |
| Document hash detection | Avoids re-indexing unchanged files |
| Async FastAPI endpoints | High concurrency under load |
| Lightweight embedding model | Fast, CPU-friendly inference |

### Trade-offs

| Area | Advantage | Limitation |
|---|---|---|
| **top_k = 3** | Faster response, smaller prompt | Some useful context may be missed |
| **Semantic threshold 0.85** | High accuracy cache reuse | Requires tuning per dataset |
| **Fallback responses** | Better user experience | Risk of hallucination without grounding |
| **CPU-based embeddings** | No GPU required | Slower than GPU inference |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI (async) |
| **Frontend** | Streamlit |
| **Vector Database** | FAISS |
| **Keyword Search** | BM25 |
| **Cache** | Redis |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **LLM** | Ollama (Mistral) |
| **Language** | Python 3.9+ |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for Redis)
- [Ollama](https://ollama.com) installed locally

### Step 1 — Clone Repository

```bash
git clone <repo_url>
cd RAG-App
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start Redis

```bash
docker run -d -p 6379:6379 redis
```

Verify it's running:

```bash
docker ps
```

### Step 5 — Pull and Start Ollama (Mistral)

```bash
ollama run mistral
```

Then exit the interactive session (`Ctrl+D`). Ollama will continue serving in the background.

### Step 6 — Start the Backend

```bash
uvicorn app.main:app --reload
```

### Step 7 — Start the Chat UI

```bash
streamlit run streamlit_app.py
```

---

## Key Features

| Feature | Status |
|---|---|
| Hybrid Retrieval (FAISS + BM25) | ✅ |
| Exact Cache (Redis) | ✅ |
| Semantic Cache (Embedding-based) | ✅ |
| TTL Cache Eviction | ✅ |
| Conversation Memory | ✅ |
| Background Document Ingestion | ✅ |
| Document Hash Tracking | ✅ |
| Streaming Responses | ✅ |
| Local LLM Processing (Offline) | ✅ |
| Structured Prompt Strategy | ✅ |
| Fallback AI Generation | ✅ |
| Document-Level Summaries | ✅ Optional |

---

## Current Limitations

- Supports **PDF files only** (no TXT / Markdown yet)
- **Single-document** session (no multi-doc retrieval)
- **CPU-based embeddings** (GPU acceleration optional)
- **No authentication** system
- Semantic cache search may slow at very large scale

---

## Future Enhancements

- [ ] Multi-document support
- [ ] GPU-accelerated embeddings
- [ ] Vector DB optimization (e.g. HNSW indexing)
- [ ] Query ranking improvements (hybrid reranking models)
- [ ] UI document status tracking
- [ ] Docker Compose deployment
- [ ] Cloud-ready architecture
- [ ] Persistent cross-session chat memory

---

## Performance Highlights

| Stage | Optimization |
|---|---|
| Retrieval | Hybrid Search (FAISS + BM25) |
| Caching | Dual Layer (Exact + Semantic) |
| Generation | Streaming output |
| Ingestion | Background processing |

---

## Learning Outcomes

This project demonstrates:

- ✅ System Design Thinking
- ✅ RAG Architecture
- ✅ Cache Optimization
- ✅ Latency Reduction Techniques
- ✅ Hybrid Search Implementation
- ✅ Production-style AI Engineering

---

## Example Usage

**1. Upload a document:**
```
Machine Learning Notes.pdf
```

**2. Ask a question:**
```
What is unsupervised learning?
```

**3. System retrieves:**
```
Relevant chunk from the uploaded document
```

**4. Generates:**
```
A context-aware, grounded answer using the retrieved content
```

---

> When a user uploads a document, it is processed through chunking and embedding, then indexed in both FAISS and BM25. When a query is received, the system first checks exact and semantic caches. If no match is found, hybrid retrieval fetches the top-k chunks, which are combined with the query and conversation history into a prompt. The LLM generates a streaming response that is cached and returned to the user.

---

*Built for speed. Designed for scale. Runs offline.*
