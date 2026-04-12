import time
import requests
import os

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
# LOAD SUMMARY
# =========================

def load_summary():

    if not os.path.exists("data/summary.txt"):

        return ""

    try:

        with open(
            "data/summary.txt",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except:

        with open(
            "data/summary.txt",
            "r",
            encoding="latin-1"
        ) as f:

            return f.read()


# =========================
# QUERY EXPANSION
# (CRITICAL FIX)
# =========================

def expand_query_using_memory(query):

    history = get_memory()

    if not history:

        return query


    last_query = history[-1]["query"]


    # If query too short → expand

    if len(query.split()) <= 3:

        expanded_query = (

            last_query +

            " " +

            query

        )

        print(
            "Expanded Query:",
            expanded_query
        )

        return expanded_query


    return query


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query):

    vector_results = retrieve(
        query,
        top_k=3
    )

    bm25_results = bm25_search(
        query,
        top_k=3
    )

    combined = list(
        dict.fromkeys(
            vector_results +
            bm25_results
        )
    )

    results = combined[:3]


    print(
        "\n========== RETRIEVED CHUNKS =========="
    )

    for i, r in enumerate(results):

        print(
            f"\nChunk {i+1}:\n{r[:200]}"
        )

    print(
        "\n======================================"
    )

    return results


# =========================
# GENERATE RESPONSE
# =========================

def generate_response(query, context):

    history = get_memory()

    summary_text = load_summary()

    context_text = "\n\n".join(context)


    # Build history

    history_text = ""

    for item in history:

        history_text += (

            f"User: {item['query']}\n"

            f"Assistant: {item['response']}\n\n"

        )


    no_context = False

    if not context_text.strip():

        no_context = True


    prompt = f"""
You are an intelligent AI assistant.

Instructions:

- Use retrieved context when available
- Use document summary when available
- Maintain conversation continuity
- Answer only from context when possible
- If no relevant context exists,
  generate answer using your own knowledge

Conversation History:
{history_text}

Document Summary:
{summary_text}

Retrieved Context:
{context_text}

User Question:
{query}

Answer clearly.
"""


    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={

                "model": "mistral",

                "prompt": prompt,

                "stream": False

            },
            timeout=120
        )

        result = response.json()

        answer = result.get(
            "response",
            "No response generated."
        )

    except Exception as e:

        print("LLM Error:", e)

        answer = "Error generating response."


    if no_context:

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


    # Exact Cache

    cached = check_exact_cache(query)

    if cached:

        log_event(
            f"Query='{query}' | Cache=Exact"
        )

        return cached, "⚡ Exact Cache Hit"


    # Semantic Cache

    semantic_cached = check_semantic_cache(query)

    if semantic_cached:

        log_event(
            f"Query='{query}' | Cache=Semantic"
        )

        return semantic_cached, "⚡ Semantic Cache Hit"


    # Expand Query (KEY FIX)

    expanded_query = expand_query_using_memory(
        query
    )


    # Retrieval

    retrieval_start = time.time()

    context = hybrid_retrieve(
        expanded_query
    )

    retrieval_time = (

        time.time() -

        retrieval_start

    )


    # LLM

    llm_start = time.time()

    response = generate_response(
        query,
        context
    )

    llm_time = (

        time.time() -

        llm_start

    )


    # Store memory

    add_to_memory(
        query,
        response
    )


    # Store cache

    store_cache(
        query,
        response
    )


    total_time = (

        time.time() -

        start_time

    )


    log_event(

        f"Query='{query}' | "

        f"Retrieval={retrieval_time:.2f}s | "

        f"LLM={llm_time:.2f}s | "

        f"Total={total_time:.2f}s"

    )


    return response, "🧠 Fresh Response"
