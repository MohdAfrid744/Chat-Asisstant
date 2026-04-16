from sentence_transformers import SentenceTransformer
import torch


def load_embedding_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Embedding device:", device)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    return model


embedding_model = load_embedding_model()

# REQUIRED — fixes FAISS dimension mismatch

embedding_dimension = (
    embedding_model.get_embedding_dimension()
)
