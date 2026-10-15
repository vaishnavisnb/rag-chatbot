"""
RAG chatbot - Step 2: find the relevant passages, then let a free LLM (Google Gemini)
write an answer using ONLY those passages, with sources.

Setup: create a file named .env in this folder containing one line:
    GEMINI_API_KEY=your_key_here
Run:
    python step2_answer.py
"""

import os

from dotenv import load_dotenv
from google import genai

from step1_search import DOCS_FOLDER, build_index, load_pdfs, search, split_into_chunks

# If you get a "model not found" error, check the current model names at
# https://ai.google.dev/gemini-api/docs/models and change this line.
MODEL = "gemini-3.8-flash"

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


def build_prompt(question, results):
    context = ""
    for chunk, _score in results:
        context += f"[Source: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}\n\n"
    return (
        "You are a helpful assistant answering questions about the user's documents.\n"
        "Use ONLY the context below. If the answer is not in the context, say "
        "\"I couldn't find that in the documents.\" Do not make anything up.\n"
        "Keep the answer clear and short, and mention the source file and page you used.\n\n"
        f"CONTEXT:\n{context}\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def ask_llm(client, question, results):
    response = client.models.generate_content(model=MODEL, contents=build_prompt(question, results))
    return response.text


def main():
    if not API_KEY:
        print("No API key found. Create a file named .env in this folder with this line:")
        print("GEMINI_API_KEY=your_key_here")
        return

    client = genai.Client(api_key=API_KEY)

    pages = load_pdfs(DOCS_FOLDER)
    if not pages:
        print(f"No readable PDFs found in the '{DOCS_FOLDER}' folder.")
        return
    chunks = split_into_chunks(pages)
    index = build_index(chunks)
    print(f"Ready! {len(pages)} pages, {len(chunks)} chunks. Type a question (or 'quit').\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        results = search(question, index, chunks)
        try:
            answer = ask_llm(client, question, results)
        except Exception as error:  # show a friendly message instead of a long traceback
            print(f"\nCould not get an answer from the model: {error}\n")
            continue
        print("\n" + answer)
        print("\nSources used:")
        for chunk, score in results:
            print(f"  - {chunk['source']}, page {chunk['page']} (match {score:.2f})")
        print()


if __name__ == "__main__":
    main()
