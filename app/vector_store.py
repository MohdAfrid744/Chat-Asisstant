# app/vector_store.py

import os
import faiss
import pickle

from app.model_loader import embedding_model


INDEX_PATH = "models/faiss_index.bin"
DOC_PATH = "models/docs.pkl"


dimension = embedding_model.get_sentence_embedding_dimension()


# =========================
# LOAD INDEX
# =========================

if os.path.exists(INDEX_PATH):

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        DOC_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

else:

    print("Creating new FAISS index...")

    index = faiss.IndexFlatIP(
        dimension
    )

    documents = []


# =========================
# ADD DOCUMENTS
# =========================

def add_documents(chunks):

    embeddings = embedding_model.encode(
        chunks,
        batch_size=16
    )

    index.add(embeddings)

    documents.extend(chunks)

    save_index()


# =========================
# HEADING BOOST
# =========================

def heading_boost(query, chunk):

    boost = 0

    keywords = [

        "objective",
        "purpose",
        "goal",
        "introduction",
        "summary"

    ]

    for word in keywords:

        if word in query.lower():

            if word in chunk.lower():

                boost += 0.25

    return boost


# =========================
# RETRIEVE
# =========================

def retrieve(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query]
    )

    D, I = index.search(
        query_embedding,
        top_k * 2
    )

    scored = []

    for score, idx in zip(D[0], I[0]):

        if idx < len(documents):

            chunk = documents[idx]

            boost = heading_boost(
                query,
                chunk
            )

            final_score = score + boost

            scored.append(

                (final_score, chunk)

            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = [

        chunk
        for _, chunk in scored[:top_k]

    ]

    return results


# =========================
# SAVE INDEX
# =========================

def save_index():

    os.makedirs(
        "models",
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        DOC_PATH,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )
