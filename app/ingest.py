import os
import requests

from pypdf import PdfReader

from app.chunker import chunk_text
from app.vector_store import add_documents
from app.bm25_store import build_bm25_index


# =========================
# TEXT EXTRACTION
# =========================

def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text

    return text


# =========================
# SUMMARY GENERATION
# =========================

def generate_summary(text):

    print("\nGenerating summary...")

    # Skip summary for small docs

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
            }
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
# MAIN INGEST FUNCTION
# =========================

def ingest_document(file_path):

    print("\n========== INGEST START ==========")


    # Step 1 — Extract text

    print("\nStep 1: Extracting text...")

    text = extract_text_from_pdf(
        file_path
    )

    if not text:

        raise ValueError(
            "No text extracted from document."
        )

    print("Text length:", len(text))


    # Step 2 — Chunk text

    print("\nStep 2: Chunking text...")

    chunks = chunk_text(text)

    print("Chunks created:", len(chunks))


    if not chunks:

        raise ValueError(
            "No chunks created."
        )


    # Step 3 — Create embeddings

    print("\nStep 3: Creating embeddings...")

    add_documents(chunks)

    print("Embeddings created successfully.")


    # Step 4 — Build BM25

    print("\nStep 4: Building BM25 index...")

    build_bm25_index(chunks)

    print("BM25 index built.")


    # Step 5 — Generate summary

    print("\nStep 5: Generating summary...")

    summary = generate_summary(text)

    print("Summary created.")


    # Step 6 — Save summary

    print("\nStep 6: Saving summary...")

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        "data/summary.txt",
        "w"
    ) as f:

        f.write(summary)


    print("\n✅ INGEST COMPLETE")