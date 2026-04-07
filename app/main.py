from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

import shutil
import os

from app.ingest import ingest_document
from app.query import process_query
from app.document_manager import is_new_document
from app.memory import clear_memory


app = FastAPI(
    title="Offline RAG Assistant",
    version="5.0"
)


UPLOAD_DIR = "data"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "message": "RAG System Running"
    }


# =========================
# INGEST (BACKGROUND)
# =========================

@app.post("/ingest")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    if is_new_document(file_path):

        # Clear old memory

        clear_memory()

        # Run ingestion in background

        background_tasks.add_task(
            ingest_document,
            file_path
        )

        message = (
            "Document uploaded. "
            "Processing in background..."
        )

    else:

        message = (
            "Document already processed."
        )


    return JSONResponse(
        content={
            "message": message
        }
    )


# =========================
# QUERY
# =========================

@app.get("/query")
async def query_system(query: str):

    def stream_response():

        response, source = process_query(
            query
        )

        yield (
            f"{source}\n\n"
            f"{response}"
        )


    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )