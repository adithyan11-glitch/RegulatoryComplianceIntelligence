import os
import shutil
from app.ingestion.ingestion import ingest_pdf, delete_pdf

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def upload_and_ingest(filename: str, file_bytes: bytes) -> dict:
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    try:
        chunks = ingest_pdf(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(f"Ingestion failed: {e}")
    return {"filename": filename, "saved_path": save_path, "chunks": chunks}


def delete_document(filename: str):
    save_path = os.path.join(UPLOAD_DIR, filename)
    delete_pdf(save_path)          # remove from postgres
    if os.path.exists(save_path):
        os.remove(save_path)       # remove from disk


def list_documents() -> list[str]:
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
