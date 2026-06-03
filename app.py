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
        return "Service is temporarily unavailable. Please try again in a few moments."


# --- Chatbot page ---
if st.session_state.get("go_to_chatbot"):
    st.title("💬 Chatbot")
    st.success("Document ready! Start chatting below.")

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("← Back"):
            st.session_state.go_to_chatbot = False
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

     # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Citations"):
                    for c in msg["citations"]:
                        page_info = f"Page {c['page']}" if c.get("page") is not None else ""
                        st.markdown(f"**{c['source']}** {page_info}")
                        if c.get("excerpt"):
                            st.caption(c["excerpt"])
            if msg.get("confidence_score") is not None:
                score = msg["confidence_score"]
                color = "green" if score >= 0.75 else "orange" if score >= 0.5 else "red"
                st.markdown(
                    f"<small>Confidence: <span style='color:{color}'><b>{score:.0%}</b></span></small>",
                    unsafe_allow_html=True,
                )
            if msg.get("disclaimer"):
                st.caption(f"⚠️ {msg['disclaimer']}")

    # New user input
    if prompt := st.chat_input("Ask a compliance question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching regulatory documents..."):
                res = api_post("/api/v1/query", json={"query": prompt})

            if res is None:
                st.error("Backend not reachable.")
            elif res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer returned.")
                citations = data.get("citations", [])
                confidence = data.get("confidence_score")
                disclaimer = data.get("disclaimer", "")

                st.markdown(answer)

                if citations:
                    with st.expander("📎 Citations"):
                        for c in citations:
                            page_info = f"Page {c['page']}" if c.get("page") is not None else ""
                            st.markdown(f"**{c['source']}** {page_info}")
                            if c.get("excerpt"):
                                st.caption(c["excerpt"])

                if confidence is not None:
                    color = "green" if confidence >= 0.75 else "orange" if confidence >= 0.5 else "red"
                    st.markdown(
                        f"<small>Confidence: <span style='color:{color}'><b>{confidence:.0%}</b></span></small>",
                        unsafe_allow_html=True,
                    )

                if disclaimer:
                    st.caption(f"⚠️ {disclaimer}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "confidence_score": confidence,
                    "disclaimer": disclaimer,
                })
            else:
                st.error(res.json().get("detail", "Query failed."))

    st.stop()

# --- Main page ---
st.title("Regulatory Compliance Intelligence System")
st.subheader("📄 Upload Document")

# Backend status check
status = api_get("/documents")
if status is None:
    st.error("⚠️We're having trouble connecting to the service right now. Please try again later.")
else:
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
                st.error("We couldn't process your document right now.  Please try again later.")
            elif res.status_code == 200:
                data = res.json()
                st.success(f"✅ {data['chunks']} chunks stored.")
                st.session_state.go_to_chatbot = True
                st.rerun()
            else:
                st.error("Upload failed. Please try again later.")
    st.divider()

    # Manage existing documents
    st.subheader("Manage Documents")
    files = status.json().get("files", []) if status and status.status_code == 200 else []

    if not files:
        st.info("No documents uploaded yet.")
    else:
        # Button to open chatbot if documents exist
        if st.button("💬 Go to Chatbot"):
            st.session_state.go_to_chatbot = True
            st.rerun()
        st.divider()
        for fname in files:
            col1, col2 = st.columns([4, 1])
            col1.write(fname)
            if col2.button("🗑 Delete", key=fname):
                r = api_delete(f"/delete/{fname}")
                if r is None:
                    st.error("We couldn't process your document right now. Please try again later.")
                elif r.status_code == 200:
                    st.success(f"{fname} deleted.")
                    st.rerun()
                else:
                    st.error("Delete failed. Please try again later.")
