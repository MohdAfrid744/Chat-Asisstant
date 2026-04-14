# app/ingest.py

import os
import json
import time
import requests

from pypdf import PdfReader

from app.chunker import chunk_text
from app.vector_store import add_documents
from app.bm25_store import build_bm25_index


# =========================
# TEXT EXTRACTION
# =========================

def extract_text(file_path):

    ext = os.path.splitext(file_path)[1]

    text = ""

    if ext == ".pdf":

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

    elif ext in [".txt", ".md"]:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            text = f.read()

    else:

        raise ValueError(
            "Unsupported file format."
        )

    return text


# =========================
# DYNAMIC CHUNKING
# =========================

def dynamic_chunking(text):

    text_length = len(text)

    if text_length < 2000:

        chunk_size = 300
        overlap = 50

    elif text_length < 10000:

        chunk_size = 500
        overlap = 80

    else:

        chunk_size = 800
        overlap = 120

    print(
        f"Dynamic chunk_size={chunk_size}, overlap={overlap}"
    )

    return chunk_text(
        text,
        chunk_size,
        overlap
    )


# =========================
# SUMMARY GENERATION
# =========================

def generate_summary(text):

    print("\nGenerating summary...")

    if len(text) < 1500:

        print("Short document — skipping summary.")

        return "Short document — summary skipped."


    short_text = text[:2000]

    prompt = f"""
Summarize the following document
in 5–7 clear sentences.

Document:
{short_text}
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        result = response.json()

        return result.get(
            "response",
            "Summary generation failed."
        )

    except Exception as e:

        print("Summary Error:", e)

        return "Summary generation failed."


# =========================
# MAIN INGEST
# =========================

def ingest_document(file_path):

    start_time = time.time()

    print("\n========== INGEST START ==========")


    # Step 1 — Extract text

    print("\nStep 1: Extracting text...")

    text = extract_text(file_path)

    if not text:

        raise ValueError(
            "No text extracted."
        )

    print("Text length:", len(text))


    # Step 2 — Chunking

    print("\nStep 2: Chunking text...")

    chunks = dynamic_chunking(text)

    print("Chunks created:", len(chunks))


    # Save metadata

    metadata = {

        "chunk_count": len(chunks),
        "text_length": len(text)

    }

    os.makedirs("data", exist_ok=True)

    with open(
        "data/doc_metadata.json",
        "w"
    ) as f:

        json.dump(metadata, f)


    # Step 3 — Embeddings

    print("\nStep 3: Creating embeddings...")

    add_documents(chunks)

    print("Embeddings created.")


    # Step 4 — BM25

    print("\nStep 4: Building BM25 index...")

    build_bm25_index(chunks)

    print("BM25 built.")


    # Step 5 — Summary

    print("\nStep 5: Generating summary...")

    summary = generate_summary(text)


    with open(
        "data/summary.txt",
        "w",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        f.write(summary)


    total_time = time.time() - start_time

    print(
        f"\n✅ INGEST COMPLETE ({total_time:.2f}s)"
    )

    return total_time
