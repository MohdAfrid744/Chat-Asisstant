import faiss
import numpy as np
import pickle
import os
import torch

from sentence_transformers import SentenceTransformer


# =========================
# DEVICE SETUP (GPU/CPU)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Embedding device: {device}")


# =========================
# MODEL LOAD
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)


# =========================
# FILE PATHS
# =========================

INDEX_PATH = "models/faiss_index.bin"
DOCS_PATH = "models/docs.pkl"


# =========================
# LOAD OR CREATE INDEX
# =========================

dimension = 384

if os.path.exists(INDEX_PATH):

    print("Loading existing FAISS index...")

    index = faiss.read_index(
        INDEX_PATH
    )

else:

    print("Creating new FAISS index...")

    index = faiss.IndexFlatL2(
        dimension
    )


# =========================
# LOAD DOCUMENTS
# =========================

if os.path.exists(DOCS_PATH):

    with open(
        DOCS_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

else:

    documents = []


# =========================
# ADD DOCUMENTS
# =========================

def add_documents(chunks):

    global documents

    if not chunks:

        return


    print(
        f"Generating embeddings for {len(chunks)} chunks..."
    )


    embeddings = model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True
    )


    index.add(
        np.array(embeddings)
    )


    documents.extend(
        chunks
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    # Save FAISS index

    faiss.write_index(
        index,
        INDEX_PATH
    )


    # Save documents

    with open(
        DOCS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )


    print("FAISS index updated.")


# =========================
# RETRIEVE FUNCTION
# =========================

def retrieve(query, top_k=3):

    if index.ntotal == 0:

        return []


    query_embedding = model.encode(
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
