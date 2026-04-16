# app/query.py

import time
import requests

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


# =========================
# CONFIG
# =========================

TOP_K_CONTEXT = 3


# =========================
# QUERY NORMALIZATION
# =========================

def normalize_query(query):

    """
    Ensures consistent matching
    for cache lookups
    """

    return query.strip().lower()


# =========================
# QUERY COMPRESSION
# =========================

def compress_query(query):

    """
    Fix short queries like:

    'objective'
    'summary'
    """

    words = query.split()

    if len(words) <= 2:

        query = query + " of this report"

    return query


# =========================
# QUERY EXPANSION
# =========================

def expand_query(query):

    """
    Adds semantic keywords
    improves retrieval accuracy
    """

    expanded = query

    q = query.lower()

    if "objective" in q:

        expanded += " purpose goal intent aim"

    if "summary" in q:

        expanded += " overview conclusion highlights"

    if "introduction" in q:

        expanded += " background overview start"

    if "method" in q:

        expanded += " process approach workflow"

    if "result" in q:

        expanded += " outcome findings output"

    if "conclusion" in q:

        expanded += " summary final remarks"

    print("Expanded query:", expanded)

    return expanded


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query):

    # Step 1 — Compression

    query = compress_query(query)

    # Step 2 — Expansion

    query = expand_query(query)

    # Step 3 — Vector Retrieval

    vector_results = retrieve(
        query,
        top_k=5
    )

    # Step 4 — BM25 Retrieval

    bm25_results = bm25_search(
        query,
        top_k=5
    )

    # Step 5 — Merge

    combined = list(

        dict.fromkeys(

            vector_results +
            bm25_results

        )

    )

    return combined[:TOP_K_CONTEXT]


# =========================
# LOAD SUMMARY
# =========================

def load_summary():

    try:

        with open(
            "data/summary.txt",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except:

        return ""


# =========================
# BUILD PROMPT
# =========================

def build_prompt(
    query,
    context,
    summary,
    history
):

    history_text = ""

    for item in history[-3:]:

        history_text += (

            f"User: {item['query']}\n"
            f"Assistant: {item['response']}\n\n"

        )

    context_text = "\n\n".join(context)


    prompt = f"""
You are an intelligent assistant.

Use the retrieved context to answer.

Retrieved Context:
{context_text}

Summary:
{summary}

Conversation:
{history_text}

User Question:
{query}

Rules:

1. Use retrieved context first
2. Be precise and factual
3. If context missing → answer using knowledge

Answer:
"""

    return prompt


# =========================
# GENERATE RESPONSE
# =========================

def generate_response(
    query,
    context
):

    history = get_memory()

    summary = load_summary()

    prompt = build_prompt(

        query,
        context,
        summary,
        history

    )


    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": "mistral",

                "prompt": prompt,

                "stream": False,

                # ⚡ Speed Control

                "options": {

                    "num_predict": 200

                }

            },

            timeout=120

        )

        result = response.json()

        answer = result.get(

            "response",
            "No response generated."

        )

    except:

        answer = "⚠️ Model error."


    # =========================
    # AI NOTE LOGIC (FIXED)
    # =========================

    if len(context) == 0:

        answer += (

            "\n\nNOTE: "
            "The output is generated entirely with AI."

        )

    return answer


# =========================
# MAIN QUERY FUNCTION
# =========================

def process_query(query):

    start_time = time.time()


    # =========================
    # NORMALIZE QUERY
    # =========================

    query = normalize_query(query)


    # =========================
    # EXACT CACHE
    # =========================

    cached = check_exact_cache(query)

    if cached:

        return cached, "⚡ Exact Cache Hit"


    # =========================
    # SEMANTIC CACHE
    # =========================

    semantic = check_semantic_cache(query)

    if semantic:

        return semantic, "⚡ Semantic Cache Hit"


    # =========================
    # RETRIEVE CONTEXT
    # =========================

    context = hybrid_retrieve(query)


    print(
        "\n========== RETRIEVED CHUNKS =========="
    )

    for i, c in enumerate(context):

        print(f"\nChunk {i+1}:\n{c}")

    print(
        "\n======================================"
    )


    # =========================
    # GENERATE RESPONSE
    # =========================

    response = generate_response(

        query,
        context

    )


    # =========================
    # STORE MEMORY
    # =========================

    add_to_memory(

        query,
        response

    )


    # =========================
    # STORE CACHE
    # =========================

    store_cache(

        query,
        response

    )


    # =========================
    # LOGGING
    # =========================

    total_time = time.time() - start_time

    log_event(

        f"Query='{query}' | "
        f"Total={total_time:.2f}s"

    )


    return response, "🧠 Fresh Response"
