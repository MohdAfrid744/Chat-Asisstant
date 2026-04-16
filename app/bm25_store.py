import pickle
import os

from rank_bm25 import BM25Okapi


BM25_PATH = "data/bm25.pkl"
CHUNK_PATH = "data/chunks.pkl"


def build_bm25_index(chunks):

    tokenized = [

        chunk.split()

        for chunk in chunks

    ]

    bm25 = BM25Okapi(tokenized)


    os.makedirs("data", exist_ok=True)


    with open(
        BM25_PATH,
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )


    with open(
        CHUNK_PATH,
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


def bm25_search(query, top_k=5):

    if not os.path.exists(BM25_PATH):

        return []


    with open(
        BM25_PATH,
        "rb"
    ) as f:

        bm25 = pickle.load(f)


    with open(
        CHUNK_PATH,
        "rb"
    ) as f:

        chunks = pickle.load(f)


    scores = bm25.get_scores(
        query.split()
    )


    ranked = sorted(

        range(len(scores)),

        key=lambda i: scores[i],

        reverse=True

    )[:top_k]


    return [

        chunks[i]

        for i in ranked

    ]
