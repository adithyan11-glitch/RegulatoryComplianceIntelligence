import os
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.db import get_embeddings, get_vector_store

load_dotenv()

COLLECTION_NAME = "regulatory_compliance_system"


def load_pdf(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    docs = []
    last_updated = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": file_path,
                "document_extension": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
                "category": COLLECTION_NAME,
                "last_updated": last_updated,
            }
        ))
    return docs


def ingest_pdf(file_path: str) -> int:
    """Ingest PDF into pgvector. Returns number of chunks stored."""
    docs = load_pdf(file_path)
    if not docs:
        raise ValueError("No extractable text found. Is this a scanned PDF?")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    ids = [
        hashlib.md5(
            f"{chunk.metadata['source']}-{chunk.metadata['page']}-{i}".encode()
        ).hexdigest()
        for i, chunk in enumerate(chunks)
    ]

    vector_store = get_vector_store(COLLECTION_NAME)
    for chunk, doc_id in zip(chunks, ids):
        vector_store.add_documents([chunk], ids=[doc_id])

    return len(chunks)


def delete_pdf(file_path: str):
    """Delete all pgvector chunks associated with a file path."""
    vector_store = get_vector_store(COLLECTION_NAME)
    # PGVector supports filtering by metadata
    vector_store.delete(filter={"source": file_path})
