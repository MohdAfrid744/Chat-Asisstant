import redis
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================
# REDIS CONNECTION
# =========================

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


# =========================
# LOAD MODEL ONCE
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


SIMILARITY_THRESHOLD = 0.85


# =========================
# EXACT CACHE
# =========================

def check_exact_cache(query):

    response = redis_client.get(query)

    if response:

        return response

    return None


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(a, b):

    return np.dot(a, b) / (

        np.linalg.norm(a)

        * np.linalg.norm(b)

    )


# =========================
# SEMANTIC CACHE
# =========================

def check_semantic_cache(query):

    query_embedding = model.encode(
        [query]
    )[0]


    # Use scan_iter (FAST)

    for key in redis_client.scan_iter(
        "embed:*"
    ):

        stored_embedding = np.array(

            json.loads(

                redis_client.get(key)

            )

        )


        similarity = cosine_similarity(

            query_embedding,

            stored_embedding

        )


        if similarity >= SIMILARITY_THRESHOLD:

            response_key = key.replace(

                "embed:",

                "response:"

            )


            return redis_client.get(

                response_key

            )


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


    # Semantic cache

    embedding = model.encode(

        [query]

    )[0]


    redis_client.set(

        f"embed:{query}",

        json.dumps(

            embedding.tolist()

        ),

        ex=86400

    )


    redis_client.set(

        f"response:{query}",

        response,

        ex=86400

    )