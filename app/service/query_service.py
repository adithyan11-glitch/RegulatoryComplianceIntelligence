import os
import re
from dotenv import load_dotenv
from app.retrieval.retrieval import hybrid_search

load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
except Exception:
    _llm = None

DISCLAIMER = (
    "This response is generated from regulatory documents for informational purposes only. "
    "It does not constitute legal or compliance advice. Always consult a qualified compliance "
    "professional or legal counsel before making regulatory decisions."
)

SYSTEM_PROMPT = """You are an AI compliance assistant for banking and financial regulations. Focus primarily on RBI, SEBI, Basel III, AML/KYC, risk management, fraud controls, audit, and regulatory reporting.

You should prioritize compliance-related conversations and gently steer discussions toward these domains. You may engage in brief normal conversation, but avoid providing detailed responses outside financial compliance topics.

For unrelated queries, respond politely:
"I can best assist with banking and financial compliance topics such as RBI, SEBI, AML/KYC, Basel III, and risk governance."

Guidelines:

Use a professional and concise tone.
Prefer grounded, regulation-based responses.
Do not fabricate rules, citations, or interpretations.
If information is unavailable, clearly say so.
For compliance answers, include references when available.

Respond in JSON format:
{
"answer": "<response>",
"citations": [{"source": "<filename>", "page": <int>, "excerpt": "<short quote>"}],
"rule_summary": {},
"confidence_score": 0.0
}
"""
""



def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", "?")
        parts.append(f"[Chunk {i+1} | {source} | Page {page}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_rule_summary(answer: str) -> dict:
    """Simple regex extraction of key rules from the answer text."""
    rules = {}
    patterns = [
        (r"LTV[^\d](\d+(?:\.\d+)?)\s%", "LTV"),
        (r"CAR[^\d](\d+(?:\.\d+)?)\s%", "CAR"),
        (r"CRR[^\d](\d+(?:\.\d+)?)\s%", "CRR"),
        (r"SLR[^\d](\d+(?:\.\d+)?)\s%", "SLR"),
        (r"repo rate[^\d](\d+(?:\.\d+)?)\s%", "Repo Rate"),
        (r"(\d+(?:\.\d+)?)\s*%.*?(?:limit|ratio|cap|floor)", "Regulatory Limit"),
    ]
    for pattern, name in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match and name not in rules:
            rules[name] = f"{match.group(1)}%"
    return rules


def _confidence_from_chunks(chunks: list[dict]) -> float:
    """Estimate confidence based on retrieval quality."""
    if not chunks:
        return 0.1
    if len(chunks) >= 5:
        return 0.85
    if len(chunks) >= 3:
        return 0.72
    return 0.55


async def handle_query(query: str) -> dict:
    chunks = hybrid_search(query, k=6)
    context = _build_context(chunks)

    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        citations.append({
            "source": os.path.basename(meta.get("source", "unknown")),
            "page": meta.get("page", None),
            "excerpt": chunk["content"][:200].strip(),
        })

    if _llm is None:
        return {
            "query": query,
            "answer": "LLM not configured. Set GEMINI_API_KEY in your .env file.",
            "citations": citations,
            "rule_summary": {},
            "confidence_score": 0.0,
            "disclaimer": DISCLAIMER,
        }

    user_message = f"Regulatory document excerpts:\n\n{context}\n\nUser query: {query}"

    import json
    try:
        response = await _llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*$", "", raw)
        parsed = json.loads(raw)

        return {
            "query": query,
            "answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", citations),
            "rule_summary": parsed.get("rule_summary", _extract_rule_summary(parsed.get("answer", ""))),
            "confidence_score": float(parsed.get("confidence_score", _confidence_from_chunks(chunks))),
            "disclaimer": DISCLAIMER,
        }
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return raw answer without structured parse
        answer_text = response.content if hasattr(response, "content") else str(e)
        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "rule_summary": _extract_rule_summary(answer_text),
            "confidence_score": _confidence_from_chunks(chunks),
            "disclaimer": DISCLAIMER,
        }