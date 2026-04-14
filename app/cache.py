import redis
import numpy as np
import pickle
import os
import faiss

from app.model_loader import (
    embedding_model,
    embedding_dimension
)

SIMILARITY_THRESHOLD = 0.65

CACHE_INDEX_PATH = "models/cache_index.bin"
CACHE_DATA_PATH = "models/cache_data.pkl"

# Redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# =========================
# LOAD CACHE
# =========================

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
        embedding_dimension
    )

    cache_data = []

    print("Created new semantic cache index.")


# =========================
# NORMALIZATION
# =========================

def normalize_embedding(vec):

    vec = np.array(vec).astype("float32")

    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


# =========================
# EXACT CACHE
# =========================

def check_exact_cache(query):

    response = redis_client.get(query)

    if response:

        print("⚡ Exact Cache Hit")

        return response

    return None


# =========================
# SEMANTIC CACHE
# =========================

def check_semantic_cache(query):

    if len(cache_data) == 0:
        return None

    embedding = embedding_model.encode(query)

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

    redis_client.set(
        query,
        response,
        ex=86400
    )

    embedding = embedding_model.encode(query)

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
