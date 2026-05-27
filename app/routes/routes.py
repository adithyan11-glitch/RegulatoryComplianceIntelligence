import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.service.service import upload_and_ingest, delete_document, list_documents

router = APIRouter()


@router.post("/upload-and-ingest")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        result = upload_and_ingest(file.filename, await file.read())
        return {"message": "Ingested successfully.", **result}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{filename}")
async def delete(filename: str):
    try:
        delete_document(filename)
        return {"message": f"{filename} deleted from disk and postgres."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def documents():
    return {"files": list_documents()}
