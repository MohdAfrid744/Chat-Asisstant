# app/vector_store.py

import faiss
import numpy as np
import pickle
import os

from app.model_loader import embedding_model


dimension = embedding_model.get_embedding_dimension()


if os.path.exists("models/faiss_index.bin"):

    index = faiss.read_index(
        "models/faiss_index.bin"
    )

    with open(
        "models/docs.pkl",
        "rb"
    ) as f:

        documents = pickle.load(f)

    print("Loading existing FAISS index...")

else:

    index = faiss.IndexFlatL2(dimension)

    documents = []

    print("Creating new FAISS index...")


# =========================
# ADD DOCUMENTS
# =========================

def add_documents(chunks):

    global documents

    print(
        f"Generating embeddings for {len(chunks)} chunks..."
    )

    embeddings = embedding_model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True
    )

    index.add(
        np.array(embeddings)
    )

    documents.extend(chunks)

    os.makedirs(
        "models",
        exist_ok=True
    )

    faiss.write_index(
        index,
        "models/faiss_index.bin"
    )

    with open(
        "models/docs.pkl",
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )

    print("FAISS index updated.")


# =========================
# RETRIEVE
# =========================

def retrieve(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(documents):

            results.append(
                documents[idx]
            )

    return results
