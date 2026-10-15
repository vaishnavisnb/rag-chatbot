"""
RAG chatbot - Step 3: Streamlit chat page.

Run from the rag-chatbot folder:
    python -m streamlit run chat_app.py

Needs step1_search.py and step2_answer.py in the same folder, and GEMINI_API_KEY in .env
"""

import io
import os

import streamlit as st
from google import genai
from pypdf import PdfReader

import step2_answer  # loads .env, so GEMINI_API_KEY becomes available below
from step1_search import DOCS_FOLDER, build_index, load_pdfs, search, split_into_chunks
from step2_answer import ask_llm

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:  # when deployed, the key is stored in Streamlit secrets instead of .env
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def make_index(pages):
    if not pages:
        return None, None
    chunks = split_into_chunks(pages)
    return build_index(chunks), chunks


@st.cache_resource(show_spinner="Reading your PDFs and building the search index...")
def index_from_uploads(files):
    """files is a tuple of (filename, bytes)."""
    pages = []
    for name, data in files:
        reader = PdfReader(io.BytesIO(data))
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, name, number))
    return make_index(pages)


@st.cache_resource(show_spinner="Reading the PDFs in the docs folder...")
def index_from_folder():
    return make_index(load_pdfs(DOCS_FOLDER))


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    st.caption("If you upload nothing, the PDFs in the local `docs` folder are used.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main page ----------------
st.title("📄 Chat with your PDFs")
st.caption("Ask a question. Answers come only from your documents, with the pages they came from.")

api_key = get_api_key()
if not api_key:
    st.error("No API key found. Add GEMINI_API_KEY to your .env file and restart the app.")
    st.stop()
client = genai.Client(api_key=api_key)

if uploaded:
    index, chunks = index_from_uploads(tuple((f.name, f.getvalue()) for f in uploaded))
else:
    index, chunks = index_from_folder()

if index is None:
    st.info("Upload a PDF in the sidebar to get started (text PDFs work best, not scanned images).")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("📚 Sources"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']} (match {s['score']:.2f})")
            st.caption(s["preview"])


# Show the earlier messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results = search(question, index, chunks)
        with st.spinner("Thinking..."):
            try:
                answer = ask_llm(client, question, results)
            except Exception as error:
                answer = f"Sorry, I couldn't get an answer from the model right now. ({error})"
        st.markdown(answer)
        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "score": score,
                "preview": chunk["text"][:300].replace("\n", " ") + "...",
            }
            for chunk, score in results
        ]
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
