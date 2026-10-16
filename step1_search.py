"""
RAG chatbot - Step 1: read PDFs, split them into chunks, and search them.
No API key needed. Run from the project folder:
    python step1_search.py
Put your PDF files inside the `docs` folder first.
"""

import glob
import os

import faiss
import numpy as np
from pypdf import PdfReader

DOCS_FOLDER = "docs"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 5             # how many chunks to show for each question

_model = None
_kind = None


def _load_model():
    """Use sentence-transformers if it is installed (local use), otherwise the
    much lighter FastEmbed (used on free hosting, which has little memory)."""
    try:
        from sentence_transformers import SentenceTransformer

        print("Loading embedding model (first time takes a minute)...")
        return "sentence-transformers", SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        from fastembed import TextEmbedding

        print("Loading lightweight embedding model...")
        return "fastembed", TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    """Turn a list of texts into vectors (numbers that capture meaning)."""
    global _model, _kind
    if _model is None:
        _kind, _model = _load_model()
    if _kind == "sentence-transformers":
        vectors = _model.encode(texts, show_progress_bar=False)
    else:
        vectors = list(_model.embed(texts))
    vectors = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)  # normalise so inner product = cosine similarity


def load_pdfs(folder):
    """Return a list of (text, filename, page_number) for every page of every PDF."""
    pages = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        reader = PdfReader(path)
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, os.path.basename(path), number))
    return pages


def split_into_chunks(pages):
    """Cut each page into overlapping chunks, remembering where each came from."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for text, source, page_number in pages:
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "source": source, "page": page_number})
    return chunks


def build_index(chunks):
    vectors = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine similarity (vectors are normalised)
    index.add(vectors)
    return index


def search(question, index, chunks, k=TOP_K):
    scores, ids = index.search(embed([question]), min(k, len(chunks)))
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def main():
    pages = load_pdfs(DOCS_FOLDER)
    if not pages:
        print(f"No readable PDFs found in the '{DOCS_FOLDER}' folder. Add a PDF and run again.")
        return

    chunks = split_into_chunks(pages)
    print(f"Read {len(pages)} pages and made {len(chunks)} chunks.")
    index = build_index(chunks)
    print("Ready! Type a question (or 'quit' to stop).\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        for rank, (chunk, score) in enumerate(search(question, index, chunks), start=1):
            print(f"\n[{rank}] {chunk['source']} - page {chunk['page']}  (match: {score:.2f})")
            print(chunk["text"][:400].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
