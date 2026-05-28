import re
from app.core.db import get_vector_store
import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Use this single source of truth for all connections in this file
_raw_conn_string = os.getenv("PG_CONNECTION_STRING", "")

_raw_conn = _raw_conn_string.replace("+psycopg2", "")


@tool
def fts_search(query: str, k: int = 6, collection_name: str = "regulatory_compliance_system")-> list[dict]:
    """Search for documents containing the exact words or phrases in your query. Best for precise terms, codes, or legal/technical language.
    use when:
    when query includes specific keywords, IDs, or technical terms
    Ideal for filtering before deeper semantic search
    """
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
    print("fts-searching.....")
    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query,
                            "collection": collection_name,
                            "k":k})
            rows = cur.fetchall()

    return [
    {
        "content": row["content"],
        "metadata": row["metadata"],
        "fts_rank": round(float(row["fts_rank"]),4),
    }
    for row in rows
   ]


@tool
def vector_search(query: str, k: int = 6)-> list[dict]:
    """search based on meaning, not just keywords. Perfect for natural language questions, paraphrased queries, or contextual understanding.
    use when:
    Your query is in natural language or vague
    You want semantic matches rather than exact words
    """ 
    print("vector-searching.....")   
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


agent = create_agent(
    model = "google_genai:gemini-3.1-pro-preview",#brain
    tools = [fts_search, vector_search],#register the tool with the agent
    system_prompt = """You are an expert AI compliance assistant designed to support bank compliance officers. Your primary objective is to provide accurate, high-integrity, and fully cited responses to regulatory queries based on tools response.

### Operational Protocol

To ensure the highest level of accuracy and regulatory compliance, you must strictly adhere to the following workflow for every user query:

1. **Search Strategy:** Analyze the user's query to identify the specific regulatory domain (RBI, SEBI, or Basel III). Execute a precise search using the provided search tool to retrieve the most relevant, up-to-date documentation chunks.
2. **Synthesis & Citation:** Draft the response with a **temperature setting of 0** to ensure deterministic, factual, and non-hallucinated output. Every claim, interpretation, or regulatory requirement mentioned must be accompanied by a clear, specific citation (e.g., "[Document Name, Section/Clause Number]").

### Response Guidelines

**Tone:** Maintain a professional, objective, and authoritative tone suitable for legal and compliance environments.
**Accuracy:** If the retrieved documents do not contain sufficient information to answer the query, explicitly state that the information is unavailable in the current regulatory database rather than attempting to infer or guess.
**Structure:** Use clear headings and bullet points to make complex regulatory requirements easy to navigate.
**Integrity:** Always prioritize the explicit text of the regulations. If there is ambiguity in a regulation, reflect that ambiguity rather than providing a definitive interpretation that could lead to non-compliance.

---

**System Constraint:** You are operating in a zero-hallucination mode. All responses must be grounded exclusively in the provided search results. If a user asks for advice, provide the relevant regulatory facts and clauses, but remind the user that this does not constitute formal legal counsel.""",#roles and goals
)