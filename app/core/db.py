import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDINGS_MODEL", "models/gemini-embedding-2"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=1536,
        task_type="RETRIEVAL_DOCUMENT",
    )


def get_vector_store(collection_name: str = "regulatory_compliance_system", pre_delete_collection: bool = False):
   return PGVector(
       collection_name=collection_name,
       connection=PG_CONNECTION,
       embeddings=get_embeddings(),
       use_jsonb=True,
       pre_delete_collection=pre_delete_collection
   )

