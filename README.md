# Regulatory Compliance Intelligence System

A Retrieval-Augmented Generation (RAG) based compliance assistant that helps users ask questions about banking and financial regulations from uploaded PDF documents.  
The system extracts content from regulatory PDFs, stores searchable chunks in PostgreSQL with PGVector, retrieves the most relevant information using hybrid search, and generates answers with citations using a Gemini LLM.

---

## 1. Project Objective

The main goal of this project is to make regulatory document search faster, easier, and more reliable.

Instead of manually reading long RBI, SEBI, Basel III, AML/KYC, or compliance documents, users can upload a PDF and ask questions in simple English.

Example questions:

- What is the maximum LTV ratio for gold loans?
- When is a term loan classified as NPA?
- What is the minimum CET1 ratio under Basel III?
- What are the key AML/KYC requirements?

The system returns:

- A clear answer
- Source citations
- Page number
- Rule summary
- Confidence score
- Compliance disclaimer

---

## 2. High-Level Architecture

```text
User
  |
  v
Streamlit UI (app.py)
  |
  v
FastAPI Backend (main.py)
  |
  v
Routes Layer (app/routes/routes.py)
  |
  +--------------------------+
  |                          |
  v                          v
Ingestion Pipeline        Query Pipeline
  |                          |
  v                          v
PDF Processing            Hybrid Retrieval
  |                          |
  v                          v
Chunking + Embedding      Context Building
  |                          |
  v                          v
PostgreSQL + PGVector     Gemini LLM
  |                          |
  +-------------> Answer + Citations
```

---

## 3. Folder Structure

```text
RegulatoryComplianceIntelligence/
│
├── app.py
├── main.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .env
│
├── app/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── db.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── ingestion.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retrieval.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   └── service/
│       ├── __init__.py
│       ├── service.py
│       └── query_service.py
│
└── data/
    └── Capstone_Project_1_Regulatory_Compliance_System_FAQ.pdf
```

---

## 4. Main Components

### 4.1 Streamlit UI - `app.py`

This file provides the user interface.

Main responsibilities:

- Upload PDF files
- Show uploaded documents
- Delete documents
- Open chatbot screen
- Send user questions to the backend API
- Display answer, citations, confidence score, and disclaimer

The UI communicates with FastAPI using HTTP requests.

Important backend calls from Streamlit:

```python
POST /upload-and-ingest
POST /api/v1/query
GET /documents
DELETE /delete/{filename}
```

---

### 4.2 FastAPI Entry Point - `main.py`

This is the backend application entry point.

```python
from fastapi import FastAPI
from app.routes.routes import router

app = FastAPI(title="RAG Ingestion API")
app.include_router(router)
```

Purpose:

- Creates the FastAPI app
- Registers all routes from `app/routes/routes.py`
- Runs the backend API server

---

### 4.3 Routes Layer - `app/routes/routes.py`

This file defines the API endpoints.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload-and-ingest` | Upload a PDF and ingest it into vector DB |
| POST | `/api/v1/query` | Ask a compliance question |
| GET | `/documents` | List uploaded PDF documents |
| DELETE | `/delete/{filename}` | Delete PDF and related vector chunks |

The routes layer receives requests and calls the service layer.

---

### 4.4 Service Layer - `app/service/service.py`

This file handles document-level operations.

Main functions:

```python
upload_and_ingest(filename, file_bytes)
delete_document(filename)
list_documents()
```

Responsibilities:

- Save uploaded PDF into the `data/` folder
- Call the ingestion pipeline
- Delete PDF from local storage
- Delete related chunks from PostgreSQL / PGVector
- List available PDF files

---

### 4.5 Database Configuration - `app/core/db.py`

This file manages embeddings and vector database connection.

Main responsibilities:

- Load environment variables from `.env`
- Create Google Generative AI embeddings
- Connect to PostgreSQL + PGVector
- Return a reusable vector store object

Important functions:

```python
get_embeddings()
get_vector_store()
```

The vector store is used by both ingestion and retrieval.

---

### 4.6 Ingestion Pipeline - `app/ingestion/ingestion.py`

This file converts uploaded PDFs into searchable chunks.

Main steps:

1. Load PDF using `PdfReader`
2. Extract text from each page
3. Add metadata such as source, page number, total pages, and last updated time
4. Split text into smaller chunks using `RecursiveCharacterTextSplitter`
5. Generate unique chunk IDs using MD5 hashing
6. Store chunks and embeddings in PGVector

Important functions:

```python
load_pdf(file_path)
ingest_pdf(file_path)
delete_pdf(file_path)
```

Why chunking is needed:

Large documents cannot be sent fully to the LLM.  
So the PDF is divided into smaller chunks, and only the most relevant chunks are retrieved during question answering.

---

### 4.7 Retrieval Pipeline - `app/retrieval/retrieval.py`

This file searches for relevant document chunks.

It uses hybrid retrieval:

1. **FTS Search**  
   Full Text Search is useful for exact keyword matches.

2. **Vector Search**  
   Vector search is useful for semantic meaning-based matches.

3. **Hybrid Search / RRF Ranking**  
   Results from FTS and vector search are combined using Reciprocal Rank Fusion.

Important functions:

```python
fts_search(query, k)
vector_search(query, k)
hybrid_search(query, k)
```

Why hybrid search is useful:

- FTS helps when the query has exact regulatory terms.
- Vector search helps when the user asks in natural language.
- Hybrid search improves answer relevance.

---

### 4.8 Query Service - `app/service/query_service.py`

This file contains the main RAG answer generation logic.

Main flow:

1. Receive user query
2. Call `hybrid_search()`
3. Build context from retrieved chunks
4. Send context and question to Gemini LLM
5. Parse the LLM response
6. Return answer, citations, rule summary, confidence score, and disclaimer

Important function:

```python
handle_query(query)
```

The response format includes:

```json
{
  "query": "user question",
  "answer": "generated answer",
  "citations": [],
  "rule_summary": {},
  "confidence_score": 0.85,
  "disclaimer": "..."
}
```

---

## 5. RAG Workflow

```text
Step 1: User uploads PDF
Step 2: PDF is saved in data folder
Step 3: Text is extracted page by page
Step 4: Text is split into chunks
Step 5: Embeddings are generated
Step 6: Chunks are stored in PostgreSQL + PGVector
Step 7: User asks a question
Step 8: Hybrid search finds relevant chunks
Step 9: Context is sent to Gemini LLM
Step 10: Final answer is returned with citations
```

---

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FastAPI | Backend API |
| Streamlit | Frontend UI |
| PostgreSQL | Database |
| PGVector | Vector storage and similarity search |
| Google Generative AI Embeddings | Convert text chunks into vectors |
| Gemini LLM | Generate compliance answers |
| pypdf | Extract text from PDF |
| LangChain | Document, embedding, and vector store utilities |
| uv | Python package and environment management |

---

## 7. Environment Variables

Create a `.env` file in the project root.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDINGS_MODEL=models/gemini-embedding-2
PG_CONNECTION_STRING=postgresql+psycopg://postgres:password@localhost:5433/regulatory_rag
```

Note: Do not commit real API keys to GitHub.

---

## 8. Setup Instructions

### Step 1: Create virtual environment

```bash
uv venv
```

### Step 2: Activate virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
uv sync
```

If dependencies are missing, install them using:

```bash
uv add fastapi uvicorn streamlit python-dotenv pypdf langchain langchain-postgres langchain-google-genai psycopg langchain-text-splitters
```

---

## 9. Database Setup

Start PostgreSQL with PGVector enabled.

Create database:

```sql
CREATE DATABASE regulatory_rag;
```

Connect to the database and enable vector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 10. How to Run the Project

### Start FastAPI backend

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit frontend

Open a second terminal and run:

```bash
streamlit run app.py
```

---

## 11. API Examples

### Upload and ingest PDF

```bash
curl -X POST "http://127.0.0.1:8000/upload-and-ingest" ^
  -F "file=@data/Capstone_Project_1_Regulatory_Compliance_System_FAQ.pdf"
```

### Ask a compliance question

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is the maximum LTV ratio for gold loans?\"}"
```

### List documents

```bash
curl -X GET "http://127.0.0.1:8000/documents"
```

### Delete document

```bash
curl -X DELETE "http://127.0.0.1:8000/delete/Capstone_Project_1_Regulatory_Compliance_System_FAQ.pdf"
```

---

## 12. Sample Questions

You can test the chatbot with these questions:

```text
What is the maximum LTV ratio for gold loans?
When is a term loan classified as NPA?
What is the minimum CET1 ratio under Basel III?
What are the AML KYC requirements?
What is the capital adequacy requirement?
```

---

## 13. Presentation Explanation

You can explain the project like this:

> This project is a Regulatory Compliance Intelligence System. It uses RAG architecture to help users ask compliance-related questions from uploaded regulatory PDFs. The user uploads a PDF through the Streamlit UI. The backend FastAPI service stores the file, extracts text from the PDF, splits the text into smaller chunks, creates embeddings, and stores them in PostgreSQL with PGVector. When a user asks a question, the system performs hybrid retrieval using full-text search and vector search. The best matching chunks are sent to the Gemini LLM, which generates a concise answer with citations, rule summary, confidence score, and disclaimer.

---

## 14. Why This Project Is Useful

Manual compliance document search is time-consuming and error-prone.  
This project helps compliance teams quickly find accurate information from regulatory PDFs.

Key benefits:

- Faster regulatory search
- Natural language question answering
- Page-level citations
- Confidence score
- Clear disclaimer
- Useful for BFSI compliance teams

---

## 15. Limitations

Current limitations:

- Works mainly with text-based PDFs
- Scanned PDFs may not work without OCR
- Answer quality depends on the uploaded document quality
- LLM response depends on API availability and quota
- This system provides informational support, not legal advice

---

## 16. Future Enhancements

Possible future improvements:

- Add OCR support for scanned PDFs
- Add user login and role-based access
- Add support for Word, Excel, and HTML documents
- Add document version comparison
- Add regulatory change alerts
- Add advanced reranking using Cohere or cross-encoder models
- Add audit logs for compliance traceability

---

## 17. One-Line Summary

Regulatory Compliance Intelligence System is a RAG-based application that allows users to upload regulatory PDFs and ask compliance questions with  answers, citations, and confidence scores.
