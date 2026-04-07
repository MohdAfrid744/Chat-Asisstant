from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


# =========================
# LOAD MODEL (ONCE)
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# FILE PATHS
# =========================

INDEX_PATH = "models/faiss_index.bin"
DOCS_PATH = "models/docs.pkl"


# =========================
# LOAD EXISTING INDEX
# =========================

dimension = 384

if os.path.exists(INDEX_PATH):

    index = faiss.read_index(
        INDEX_PATH
    )

else:

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
    global index

    if not chunks:

        return


    embeddings = model.encode(
        chunks
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
        INDEX_PATH
    )


    with open(
        DOCS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )


# =========================
# RETRIEVE FUNCTION ⭐
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