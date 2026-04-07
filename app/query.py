import time
import requests

from app.vector_store import retrieve
from app.bm25_store import bm25_retrieve

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
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query):

    vector_results = retrieve(
        query,
        top_k=3
    )

    bm25_results = bm25_retrieve(
        query,
        top_k=3
    )

    combined = list(
        dict.fromkeys(
            vector_results +
            bm25_results
        )
    )

    return combined[:3]


# =========================
# LLM RESPONSE
# =========================

def generate_response(query, context):

    history = get_memory()


    # Clean context

    context = [
        c for c in context
        if c and c.strip()
    ]


    context_text = "\n\n".join(
        context
    )


    history_text = "\n".join(
        [
            f"{h['role']}: {h['content']}"
            for h in history[-6:]
        ]
    )


    # Detect empty context properly

    no_context = len(context) == 0


    prompt = f"""
You are an intelligent AI assistant.

Instructions:

1. If relevant context is available:
   - Answer using the provided context
   - Keep answers clear, structured, and concise

2. If NO relevant context is found:
   - Generate the best possible answer using your own knowledge
   - At the end of the answer, add this note exactly:

   NOTE: The output is generated entirely with AI.

Conversation History:
{history_text}

Context:
{context_text}

User Question:
{query}

Answer:
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

        answer = (
            "Error generating response."
        )


    # Add disclaimer if no context

    if no_context:

        answer += (
            "\n\n⚠️ Note: "
            "This response was generated "
            "entirely using AI knowledge."
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


    # Retrieval

    retrieval_start = time.time()

    context = hybrid_retrieve(query)

    retrieval_time = (
        time.time() - retrieval_start
    )


    # LLM Generation

    llm_start = time.time()

    response = generate_response(
        query,
        context
    )

    llm_time = (
        time.time() - llm_start
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
        time.time() - start_time
    )


    log_event(
        f"Query='{query}' | "
        f"Cache=None | "
        f"Retrieval={retrieval_time:.2f}s | "
        f"LLM={llm_time:.2f}s | "
        f"Total={total_time:.2f}s"
    )


    return response, "🧠 Fresh Response"