import streamlit as st
import requests
from requests.exceptions import ConnectionError
from langchain_core.messages import HumanMessage, AIMessage
from app.retrieval.retrieval import agent


API = "http://127.0.0.1:8000"

st.set_page_config(page_title="PDF Ingestion", layout="centered")


def api_get(path):
    try:
        return requests.get(f"{API}{path}", timeout=5)
    except ConnectionError:
        return None


def api_post(path, **kwargs):
    try:
        return requests.post(f"{API}{path}", timeout=60, **kwargs)
    except ConnectionError:
        return None


def api_delete(path):
    try:
        return requests.delete(f"{API}{path}", timeout=10)
    except ConnectionError:
        return None


# --- Chatbot page ---
# --- Chatbot page ---
if st.session_state.get("go_to_chatbot"):
    st.title("💬 Chatbot")
    st.success("Document ready! Start chatting below.")

    if st.button("← Back"):
        st.session_state.go_to_chatbot = False
        st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"

        with st.chat_message(role):
            st.markdown(message.content)

    # User input
    if prompt := st.chat_input("Ask something about your document..."):
        # Store user message
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    },
                    config={
                        "run_name": "streamlit_chatbot",
                        "tags": ["chatbot", "retrieval"],
                    },
                )

                ai_message = response["messages"][-1].text

                st.markdown(ai_message)

        # Store assistant response
        st.session_state.messages.append(
            AIMessage(content=ai_message)
        )

    st.stop()

# --- Main page ---
st.title("📄 PDF Upload & Ingestion")

# Backend status check
status = api_get("/documents")
if status is None:
    st.error("⚠️ Backend not reachable. Run: `uvicorn main:app --reload`")

# Upload section
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_file:
    if st.button("Upload & Ingest"):
        with st.spinner("Ingesting..."):
            res = api_post(
                "/upload-and-ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if res is None:
            st.error("Backend not reachable.")
        elif res.status_code == 200:
            data = res.json()
            st.success(f"✅ {data['chunks']} chunks stored.")
            st.session_state.go_to_chatbot = True
            st.rerun()
        else:
            st.error(res.json().get("detail", "Upload failed."))

st.divider()

# Manage existing documents
st.subheader("Manage Documents")
files = status.json().get("files", []) if status and status.status_code == 200 else []

if not files:
    st.info("No documents ingested yet.")
else:
    for fname in files:
        col1, col2 = st.columns([4, 1])
        col1.write(fname)
        if col2.button("🗑 Delete", key=fname):
            r = api_delete(f"/delete/{fname}")
            if r is None:
                st.error("Backend not reachable.")
            elif r.status_code == 200:
                st.success(f"{fname} deleted.")
                st.rerun()
            else:
                st.error(r.json().get("detail", "Delete failed."))
