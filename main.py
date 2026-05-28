from fastapi import FastAPI
from app.routes.routes import router

app = FastAPI(title="RAG Ingestion API")
app.include_router(router)
