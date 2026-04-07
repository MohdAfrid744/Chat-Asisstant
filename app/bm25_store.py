import pickle
import os

from rank_bm25 import BM25Okapi


# =========================
# FILE PATHS
# =========================

BM25_PATH = "models/bm25.pkl"
DOCS_PATH = "models/docs.pkl"


# =========================
# GLOBAL BM25 OBJECT
# =========================

bm25 = None
chunks = []


# =========================
# BUILD BM25 INDEX
# =========================

def build_bm25_index(chunks_list):

    global bm25
    global chunks

    if not chunks_list:

        return


    chunks = chunks_list


    tokenized_chunks = [

        chunk.split()

        for chunk in chunks
    ]


    bm25 = BM25Okapi(
        tokenized_chunks
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    with open(
        BM25_PATH,
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )


# =========================
# LOAD EXISTING BM25
# =========================

if os.path.exists(BM25_PATH):

    with open(
        BM25_PATH,
        "rb"
    ) as f:

        bm25 = pickle.load(f)


if os.path.exists(DOCS_PATH):

    with open(
        DOCS_PATH,
        "rb"
    ) as f:

        chunks = pickle.load(f)


# =========================
# RETRIEVE FUNCTION ⭐
# =========================

def bm25_retrieve(query, top_k=3):

    global bm25
    global chunks

    if bm25 is None:

        return []


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