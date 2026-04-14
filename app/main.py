# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import shutil
import os
import time

from app.ingest import ingest_document
from app.query import process_query
from app.document_manager import is_new_document


# =========================
# FASTAPI INIT
# =========================

app = FastAPI(
    title="Offline RAG Assistant",
    version="6.0"
)


# =========================
# CONFIG
# =========================

UPLOAD_DIR = "data"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".md"
]


# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "RAG System Ready"
    }


# =========================
# INGEST DOCUMENT
# =========================

@app.post("/ingest")
async def upload_document(
    file: UploadFile = File(...)
):

    try:

        start_time = time.time()

        filename = file.filename

        file_extension = os.path.splitext(
            filename
        )[1].lower()


        # Validate file type

        if file_extension not in SUPPORTED_EXTENSIONS:

            return JSONResponse(
                status_code=400,
                content={
                    "message":
                    "Unsupported file type.",
                    "processing_time": 0
                }
            )


        file_path = os.path.join(
            UPLOAD_DIR,
            filename
        )


        # Save file

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Check if new document

        if is_new_document(file_path):

            processing_time = ingest_document(
                file_path
            )

            return JSONResponse(
                content={
                    "message":
                    "Document processed successfully.",
                    "processing_time":
                    round(processing_time, 2)
                }
            )


        else:

            return JSONResponse(
                content={
                    "message":
                    "Document already processed.",
                    "processing_time": 0
                }
            )


    except Exception as e:

        print("INGEST ERROR:", e)

        return JSONResponse(
            status_code=500,
            content={
                "message":
                "Document processing failed.",
                "processing_time": 0
            }
        )


# =========================
# QUERY SYSTEM
# =========================

@app.get("/query")
async def query_system(query: str):

    try:

        response, source = process_query(
            query
        )

        return JSONResponse(
            content={
                "response": response,
                "source": source
            }
        )

    except Exception as e:

        print("QUERY ERROR:", e)

        return JSONResponse(
            status_code=500,
            content={
                "response":
                "⚠️ Error generating response.",
                "source":
                "❌ System Error"
            }
        )


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }
