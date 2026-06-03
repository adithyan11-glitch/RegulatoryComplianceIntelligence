import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from app.core.db import get_vector_store

load_dotenv()

_raw_conn_string = os.getenv("PG_CONNECTION_STRING", "")
_raw_conn = _raw_conn_string.replace("+psycopg2", "")

COLLECTION_NAME = "regulatory_compliance_system"


def fts_search(query: str, k: int = 6) -> list[dict]:
    """Full-text search using PostgreSQL tsvector."""
    sql = """
       SELECT
           e.document                                               AS content,
           e.cmetadata                                              AS metadata,
           ts_rank(
               to_tsvector('english', e.document),
               plainto_tsquery('english', %(query)s)
           )                                                        AS fts_rank
       FROM  langchain_pg_embedding  e
       JOIN  langchain_pg_collection c ON c.uuid = e.collection_id
       WHERE c.name = %(collection)s
         AND to_tsvector('english', e.document)
             @@ plainto_tsquery('english', %(query)s)
       ORDER BY fts_rank DESC
       LIMIT %(k)s;
    """
    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "collection": COLLECTION_NAME, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
            "vector_rank": 0.0,
        }
        for row in rows
    ]


def vector_search(query: str, k: int = 6) -> list[dict]:
    """Semantic vector similarity search."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "fts_rank": 0.0,
            "vector_rank": round(float(score), 4),
        }
        for doc, score in docs
    ]


def hybrid_search(query: str, k: int = 6, fts_weight: float = 0.3, vector_weight: float = 0.7) -> list[dict]:
    """
    Hybrid search: combines FTS + vector results using Reciprocal Rank Fusion (RRF).
    Simple, effective, no extra dependencies.
    """
    fts_results = fts_search(query, k=k)
    vec_results = vector_search(query, k=k)

    # RRF scoring: score(d) = sum(1 / (rank + 60))
    RRF_K = 60
    scores: dict[str, dict] = {}

    for rank, result in enumerate(fts_results):
        key = result["content"][:100]  # dedup key
        if key not in scores:
            scores[key] = {"result": result, "score": 0.0}
        scores[key]["score"] += fts_weight * (1 / (rank + 1 + RRF_K))

    for rank, result in enumerate(vec_results):
        key = result["content"][:100]
        if key not in scores:
            scores[key] = {"result": result, "score": 0.0}
        scores[key]["score"] += vector_weight * (1 / (rank + 1 + RRF_K))

    # Sort by combined RRF score, return top-k
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["result"] for item in ranked[:k]]