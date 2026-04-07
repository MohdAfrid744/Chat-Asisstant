# Low-Latency Cache-Optimized GenAI Assistant

> A Retrieval-Augmented Generation (RAG) system built for speed — leveraging multi-level caching, semantic search, and async processing to deliver fast, context-aware responses from your documents.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Document Ingestion Pipeline](#document-ingestion-pipeline)
- [Query Processing Pipeline](#query-processing-pipeline)
- [Caching Strategy](#caching-strategy)
- [Latency Optimization](#latency-optimization)
- [Trade-offs & Limitations](#trade-offs--limitations)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Key Features](#key-features)

---

## Overview

This project implements a low-latency GenAI assistant that answers user queries from uploaded documents (PDF, TXT, Markdown) using Retrieval-Augmented Generation (RAG).

**Key Objectives:**

| Goal | Approach |
|---|---|
| Fast response generation | Multi-level Redis caching |
| Reduced redundant LLM calls | Exact + semantic cache lookup |
| Support for similar queries | Embedding-based semantic matching |
| Scalable architecture | Modular async FastAPI design |

---

## System Architecture

The system is organized into five modular layers:

```
┌──────────────────────────────────────────┐
│              User Layer                  │
│  Upload documents · Send queries         │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│              API Layer                   │
│  FastAPI  ·  /ingest  ·  /query          │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│             Cache Layer                  │
│  Redis  ·  Exact Cache  ·  Semantic Cache│
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           Retrieval Layer                │
│  FAISS  ·  Similarity Search  ·  top_k=3 │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│           Generation Layer               │
│  LLM  ·  Prompt Building  ·  Response    │
└──────────────────────────────────────────┘
```

---

## Document Ingestion Pipeline

Documents are processed and indexed via the `/ingest` endpoint.

```
Upload Document (/ingest)
        │
        ▼
  Extract Text
        │
        ▼
  Chunk Document
  ├── Chunk size:  500–800 characters
  └── Overlap:     50–100 characters
        │
        ▼
  Generate Embeddings (all-MiniLM-L6-v2)
        │
        ▼
  Store in FAISS Vector Database
```

> **Optional Enhancement:** A short document-level summary is generated during ingestion to provide global context during response generation.

---

## Query Processing Pipeline

Incoming queries are resolved through a layered cache-first lookup before hitting the LLM.

```
Receive Query (/query)
        │
        ▼
  Level 1: Exact Cache ──── HIT ──▶ Return Response
        │ MISS
        ▼
  Level 2: Semantic Cache ── HIT ──▶ Return Response
        │ MISS
        ▼
  Retrieve top_k=3 Chunks from FAISS
        │
        ▼
  Build Prompt
  ├── Query
  ├── Retrieved chunks
  └── Optional document summary
        │
        ▼
  Send to LLM → Generate Response
        │
        ▼
  Store in Caches (L1 + L2)
        │
        ▼
  Return Response to User
```

> **Optional Enhancement — Fallback Response:** If no relevant context is found, the LLM generates a fallback response accompanied by the disclaimer:
> *"This response is generated without supporting document context."*

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

To prevent memory overflow, all cache entries automatically expire:

```
TTL = 24 hours
```

---

## Latency Optimization

### Core Strategies

- ✅ Multi-level caching (Exact + Semantic)
- ✅ `top_k = 3` chunk retrieval (small, focused prompts)
- ✅ Async FastAPI endpoints
- ✅ Lightweight embedding model (`all-MiniLM-L6-v2`)
- ✅ Precomputed document embeddings at ingestion time
- ✅ Redis in-memory caching

### Optional Enhancement — Request Deduplication

When identical queries arrive simultaneously, only **one** LLM call is executed and all concurrent requests share the same response.

**Benefits:** Reduced compute cost · Faster responses · Improved throughput

---

## Trade-offs & Limitations

| Area | Advantage | Limitation |
|---|---|---|
| **top_k = 3** | Faster response, smaller prompt | Some useful context may be missed |
| **Semantic threshold 0.85** | High accuracy cache reuse | Requires tuning per dataset |
| **Fallback responses** | Better user experience | Risk of hallucination without grounding |

---

## Tech Stack

| Component | Technology |
|---|---|
| **API Framework** | FastAPI (async) |
| **Vector Database** | FAISS |
| **Cache** | Redis |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **LLM** | OpenAI / Gemini / Mistral / HuggingFace |
| **Retrieval Framework** | LangChain |
| **Optional UI** | Streamlit |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Redis server
- API key for your chosen LLM provider

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Start Redis

```bash
redis-server
```

### Step 3 — Start the API Server

```bash
uvicorn main:app --reload
```

### Step 4 — Ingest a Document

```bash
POST /ingest
```

Upload your PDF, TXT, or Markdown file via this endpoint.

### Step 5 — Query the System

```bash
POST /query
```

Submit your question. The system will resolve it through the cache or generate a response via the LLM.

---

## Key Features

| Feature | Status |
|---|---|
| Retrieval-Augmented Generation | ✅ |
| Multi-level Caching (L1 + L2) | ✅ |
| Semantic Similarity Search | ✅ |
| Low-Latency Optimization | ✅ |
| Async Processing | ✅ |
| TTL Cache Eviction | ✅ |
| Document-Level Summaries | ✅ Optional |
| Fallback Response Handling | ✅ Optional |
| Request Deduplication | ✅ Optional |

---

## How It Works — Summary

> When a user uploads a document, it is processed through chunking and embedding, then stored in a vector database. When a query is received, the system first checks exact and semantic caches. If no match is found, relevant chunks are retrieved from the vector database and passed to the LLM along with the query. The generated response is cached for future reuse.

---

*Built for speed. Designed for scale.*
