# app/cache.py

import redis
import numpy as np
import pickle
import os
import faiss

from app.model_loader import embedding_model


# =========================
# CONFIG
# =========================

SIMILARITY_THRESHOLD = 0.48

CACHE_INDEX_PATH = "models/cache_index.bin"
CACHE_DATA_PATH = "models/cache_data.pkl"


# =========================
# REDIS CONNECTION
# =========================

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


# =========================
# NORMALIZE QUERY
# =========================

def normalize_query(query):

    return query.lower().strip()


# =========================
# NORMALIZE EMBEDDING
# =========================

def normalize_embedding(vec):

    vec = np.array(vec).astype("float32")

    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


# =========================
# LOAD CACHE INDEX
# =========================

dimension = embedding_model.get_sentence_embedding_dimension()

if os.path.exists(CACHE_INDEX_PATH):

    cache_index = faiss.read_index(
        CACHE_INDEX_PATH
    )

    with open(
        CACHE_DATA_PATH,
        "rb"
    ) as f:

        cache_data = pickle.load(f)

    print("Loaded semantic cache.")

else:

    cache_index = faiss.IndexFlatIP(
        dimension
    )

    cache_data = []

    print("Created new semantic cache index.")


# =========================
# EXACT CACHE CHECK
# =========================

def check_exact_cache(query):

    query = normalize_query(query)

    response = redis_client.get(query)

    if response:

        print("⚡ Exact Cache Hit")

        return response

    return None


# =========================
# SEMANTIC CACHE CHECK
# =========================

def check_semantic_cache(query):

    if len(cache_data) == 0:
        return None

    query = normalize_query(query)

    embedding = embedding_model.encode(
        query
    )

    embedding = normalize_embedding(
        embedding
    )

    embedding = np.array([embedding])

    distances, indices = cache_index.search(
        embedding,
        1
    )

    similarity = distances[0][0]

    print(
        f"Semantic similarity: {similarity}"
    )

    if similarity >= SIMILARITY_THRESHOLD:

        idx = indices[0][0]

        print("⚡ Semantic Cache Hit")

        return cache_data[idx]["response"]

    return None


# =========================
# STORE CACHE
# =========================

def store_cache(query, response):

    query = normalize_query(query)

    # Exact Cache

    redis_client.set(
        query,
        response,
        ex=86400
    )

    # Semantic Cache

    embedding = embedding_model.encode(
        query
    )

    embedding = normalize_embedding(
        embedding
    )

    embedding = np.array([embedding])

    cache_index.add(
        embedding
    )

    cache_data.append({

        "query": query,
        "response": response

    })

    os.makedirs(
        "models",
        exist_ok=True
    )

    faiss.write_index(
        cache_index,
        CACHE_INDEX_PATH
    )

    with open(
        CACHE_DATA_PATH,
        "wb"
    ) as f:

        pickle.dump(
            cache_data,
            f
        )

    print("💾 Cached in FAISS.")


# =========================
# CLEAR CACHE
# =========================

def clear_cache():

    redis_client.flushall()

    global cache_index
    global cache_data

    cache_index = faiss.IndexFlatIP(
        dimension
    )

    cache_data = []

    if os.path.exists(CACHE_INDEX_PATH):
        os.remove(CACHE_INDEX_PATH)

    if os.path.exists(CACHE_DATA_PATH):
        os.remove(CACHE_DATA_PATH)

    print("🧹 Cache cleared.")
