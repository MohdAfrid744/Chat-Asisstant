import os
import time
import hashlib
import json
import requests

from pypdf import PdfReader

from app.chunker import chunk_text
from app.vector_store import add_documents
from app.bm25_store import build_bm25_index


HASH_FILE = "data/file_hash.json"


# =========================
# FILE HASH
# =========================

def compute_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def is_duplicate(file_hash):

    if not os.path.exists(HASH_FILE):

        return False

    with open(HASH_FILE, "r") as f:

        data = json.load(f)

    return file_hash in data


def store_hash(file_hash):

    os.makedirs("data", exist_ok=True)

    if os.path.exists(HASH_FILE):

        with open(HASH_FILE, "r") as f:

            data = json.load(f)

    else:

        data = []

    data.append(file_hash)

    with open(HASH_FILE, "w") as f:

        json.dump(data, f)


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
# SUMMARY GENERATION
# =========================

def generate_summary(text):

    if len(text) < 1500:

        return "Short document."

    short_text = text[:3000]

    prompt = f"""
Summarize this document briefly.

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
            "Summary failed."
        )

    except:

        return "Summary generation failed."


# =========================
# MAIN INGEST FUNCTION
# =========================

def ingest_document(file_path):

    start_time = time.time()

    print("\n========== INGEST START ==========")


    # HASH CHECK

    file_hash = compute_file_hash(file_path)

    if is_duplicate(file_hash):

        print("Duplicate file detected.")

        return {
            "status": "duplicate",
            "time": 0
        }


    # TEXT EXTRACTION

    print("Step 1: Extracting text...")

    text = extract_text(file_path)

    print("Text length:", len(text))


    # CHUNKING

    print("Step 2: Chunking text...")

    chunks = chunk_text(text)

    print("Chunks created:", len(chunks))


    # VECTOR STORE

    print("Step 3: Creating embeddings...")

    add_documents(chunks)


    # BM25

    print("Step 4: Building BM25 index...")

    build_bm25_index(chunks)


    # SUMMARY

    print("Step 5: Generating summary...")

    summary = generate_summary(text)

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        "data/summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)


    store_hash(file_hash)


    total_time = time.time() - start_time


    print(
        f"\n✅ INGEST COMPLETE ({total_time:.2f}s)"
    )


    return {
        "status": "processed",
        "time": round(total_time, 2)
    }
