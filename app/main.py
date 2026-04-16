from fastapi import FastAPI
from pydantic import BaseModel

from app.ingest import ingest_document
from app.query import process_query

from app.cache import clear_cache
from app.memory import clear_memory


app = FastAPI()


# =========================
# REQUEST MODELS
# =========================

class IngestRequest(BaseModel):

    file_path: str


# =========================
# INGEST ENDPOINT
# =========================

@app.post("/ingest")

def ingest(req: IngestRequest):

    result = ingest_document(
        req.file_path
    )

    return result


# =========================
# QUERY ENDPOINT
# =========================

@app.get("/query")

def query(query: str):

    response, source = process_query(query)

    return {

        "response": response,

        "source": source

    }


# =========================
# CLEAR ENDPOINT
# =========================

@app.post("/clear")

def clear():

    clear_cache()

    clear_memory()

    return {

        "status": "cleared"

    }
