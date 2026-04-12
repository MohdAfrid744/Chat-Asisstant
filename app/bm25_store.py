import pickle
from rank_bm25 import BM25Okapi
import os


# =========================
# BUILD BM25 INDEX
# =========================

def build_bm25_index(chunks):

    tokenized_chunks = [
        chunk.split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    os.makedirs("data", exist_ok=True)

    with open(
        "data/bm25.pkl",
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )

    # Save chunks too

    with open(
        "data/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


# =========================
# BM25 SEARCH
# =========================

def bm25_search(query, top_k=3):

    if not os.path.exists("data/bm25.pkl"):

        return []

    with open(
        "data/bm25.pkl",
        "rb"
    ) as f:

        bm25 = pickle.load(f)

    with open(
        "data/chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)


    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )


    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]


    results = []

    for idx in ranked_indices:

        if idx < len(chunks):

            results.append(
                chunks[idx]
            )

    return results
