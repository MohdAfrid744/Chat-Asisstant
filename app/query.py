# app/query.py

import time
import requests
import numpy as np

from app.vector_store import retrieve
from app.bm25_store import bm25_search

from app.cache import (
    check_exact_cache,
    check_semantic_cache,
    store_cache
)

from app.memory import (
    add_to_memory,
    get_memory
)

from app.logger import log_event
from app.model_loader import embedding_model


# =========================
# CONFIG
# =========================

MIN_CONTEXT_REQUIRED = 1

MIN_THRESHOLD = 0.20
MAX_THRESHOLD = 0.40


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return np.dot(a, b)


# =========================
# ADAPTIVE THRESHOLD
# =========================

def compute_dynamic_threshold(similarities):

    if not similarities:
        return 0.25

    chunk_count = len(similarities)

    avg_sim = np.mean(similarities)

    if chunk_count < 10:

        threshold = 0.20

    else:

        threshold = avg_sim * 0.6

    threshold = max(
        MIN_THRESHOLD,
        min(MAX_THRESHOLD, threshold)
    )

    print(f"Average similarity: {avg_sim:.3f}")
    print(f"Dynamic threshold: {threshold:.3f}")

    return threshold


# =========================
# CONTEXT FILTER
# =========================

def adaptive_filter(query, chunks):

    if not chunks:
        return []

    query_emb = embedding_model.encode(query)

    chunk_embeddings = embedding_model.encode(
        chunks
    )

    similarities = []

    for chunk_emb in chunk_embeddings:

        sim = cosine_similarity(
            query_emb,
            chunk_emb
        )

        similarities.append(sim)

    threshold = compute_dynamic_threshold(
        similarities
    )

    filtered = []

    for i, sim in enumerate(similarities):

        if sim >= threshold:

            filtered.append(
                chunks[i]
            )

    if len(filtered) < MIN_CONTEXT_REQUIRED:

        print("Fallback activated.")

        return chunks[:3]

    print(
        f"Filtered {len(filtered)} relevant chunks"
    )

    return filtered[:3]


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query):

    vector_results = retrieve(
        query,
        top_k=5
    )

    bm25_results = bm25_search(
        query,
        top_k=5
    )

    combined = list(
        dict.fromkeys(
            vector_results +
            bm25_results
        )
    )

    return adaptive_filter(
        query,
        combined
    )


# =========================
# LOAD SUMMARY
# =========================

def load_summary():

    try:

        with open(
            "data/summary.txt",
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return f.read()

    except:

        return ""


# =========================
# RESPONSE GENERATION
# =========================

def generate_response(query, context):

    history = get_memory()

    history_text = ""

    for item in history:

        history_text += (
            f"User: {item['query']}\n"
            f"Assistant: {item['response']}\n\n"
        )

    context_text = "\n\n".join(context)

    summary_text = load_summary()

    has_context = len(context) > 0


    prompt = f"""
You are an intelligent AI assistant.

Retrieved Context:
{context_text}

Document Summary:
{summary_text}

Conversation History:
{history_text}

User Question:
{query}

RULES:

1. If context exists — answer from context.
2. Maintain conversation continuity.
3. If context does NOT exist — answer using general knowledge.
4. Do NOT mention context in response.
5. Keep answers concise.

Answer:
"""


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    answer = result.get(
        "response",
        "No response generated."
    )


    # Only add NOTE if context missing

    if not has_context:

        answer += (
            "\n\nNOTE: "
            "The output is generated entirely with AI."
        )


    return answer


# =========================
# MAIN QUERY
# =========================

def process_query(query):

    start_time = time.time()


    cached = check_exact_cache(query)

    if cached:

        return cached, "⚡ Exact Cache Hit"


    semantic = check_semantic_cache(query)

    if semantic:

        return semantic, "⚡ Semantic Cache Hit"


    context = hybrid_retrieve(query)


    print("\n========== RETRIEVED CHUNKS ==========")

    for i, c in enumerate(context):

        print(f"\nChunk {i+1}:\n{c}")

    print("\n======================================")


    response = generate_response(
        query,
        context
    )


    add_to_memory(
        query,
        response
    )


    store_cache(
        query,
        response
    )


    total_time = time.time() - start_time

    log_event(
        f"Query='{query}' | "
        f"Total={total_time:.2f}s"
    )


    return response, "🧠 Fresh Response"
