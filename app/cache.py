import redis
import json
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer


# =========================
# REDIS
# =========================

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


# =========================
# MODEL (GPU ENABLED)
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# SETTINGS
# =========================

SIMILARITY_THRESHOLD = 0.65

CACHE_INDEX_PATH = "models/cache_index.bin"
CACHE_DATA_PATH = "models/cache_data.json"

dimension = 384


# =========================
# LOAD / CREATE FAISS
# =========================

if os.path.exists(CACHE_INDEX_PATH):

    cache_index = faiss.read_index(
        CACHE_INDEX_PATH
    )

    print("Loaded semantic cache index.")

else:

    cache_index = faiss.IndexFlatIP(
        dimension
    )

    print("Created new semantic cache index.")


# =========================
# LOAD CACHE DATA
# =========================

if os.path.exists(CACHE_DATA_PATH):

    with open(
        CACHE_DATA_PATH,
        "r"
    ) as f:

        cache_data = json.load(f)

else:

    cache_data = []


# =========================
# NORMALIZE EMBEDDING
# =========================

def normalize(vec):

    vec = np.array(vec).astype("float32")

    faiss.normalize_L2(vec)

    return vec


# =========================
# EXACT CACHE
# =========================

def check_exact_cache(query):

    response = redis_client.get(query)

    if response:

        return response

    return None


# =========================
# SEMANTIC CACHE
# =========================

def check_semantic_cache(query):

    if cache_index.ntotal == 0:

        return None


    query_embedding = model.encode([query])

    query_embedding = normalize(
        query_embedding
    )


    distances, indices = cache_index.search(
        query_embedding,
        1
    )


    similarity = distances[0][0]

    print(
        "Semantic similarity:",
        similarity
    )


    if similarity >= SIMILARITY_THRESHOLD:

        idx = indices[0][0]

        if idx < len(cache_data):

            print("⚡ Semantic Cache Hit")

            return cache_data[idx]["response"]


    return None


# =========================
# STORE CACHE
# =========================

def store_cache(query, response):

    # Exact cache

    redis_client.set(
        query,
        response,
        ex=86400
    )


    embedding = model.encode([query])

    embedding = normalize(
        embedding
    )


    cache_index.add(embedding)


    cache_data.append({

        "query": query,

        "response": response

    })


    faiss.write_index(

        cache_index,

        CACHE_INDEX_PATH

    )


    with open(

        CACHE_DATA_PATH,

        "w"

    ) as f:

        json.dump(

            cache_data,

            f

        )


    print("💾 Cached in FAISS.")
