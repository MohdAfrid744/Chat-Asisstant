# app/model_loader.py

from sentence_transformers import SentenceTransformer
import torch

print("Loading embedding model...")

# GPU detection

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Embedding device: {device}")

# Load model ONLY ONCE

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)

# Get dimension dynamically

embedding_dimension = (
    embedding_model.get_embedding_dimension()
)