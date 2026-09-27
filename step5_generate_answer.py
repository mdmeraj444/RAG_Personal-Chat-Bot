"""Retrieve relevant PDF chunks from Pinecone and answer questions with Groq."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq


INDEX_NAME = "agentic-ai-rag"
TOP_K = 3
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
LLM_MODEL = "qwen/qwen3.8-27b"


def load_api_keys():
    """Load keys from .env beside this script, without printing their values."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")

    required_keys = ("NVIDIA_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY")
    missing_keys = [
        key for key in required_keys if not os.getenv(key, "").strip()
    ]

    if missing_keys:
        raise RuntimeError(
            "Missing API keys: " + ", ".join(missing_keys)
            + f". Add them to {env_path} or set them as environment variables."
        )


def source_details(doc):
    """Return a filename and a one-based PDF page number when available."""
    source = str(doc.metadata.get("source") or "Unknown document")
    # Handle metadata paths created on either Windows or Linux.
    source = source.replace("\\", "/").rsplit("/", 1)[-1]

    try:
        page_index = float(doc.metadata["page"])
        page = (
            int(page_index) + 1
            if page_index.is_integer() and page_index >= 0
            else "Unknown"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        page = "Unknown"

    return source, page


def format_docs(docs):
    """Add a source label to each retrieved chunk for citations."""
    formatted = []
    for doc in docs:
        source, page = source_details(doc)
        formatted.append(f"[{source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def answer_question(question, vector_store, llm):
    """Retrieve fresh context for this question and generate its answer."""
    docs = vector_store.similarity_search(question, k=TOP_K)

    if not docs:
        return "I don't know based on the available documents.", []

    context = format_docs(docs)
    prompt = f"""You are a helpful assistant that answers questions
using only the provided context.

Rules:
1. Answer only using the context below.
2. If the answer is not present in the context, say:
   "I don't know based on the available documents."
3. Keep the answer clear and concise.
4. Cite the source document and page number for factual claims.
5. If a source label says Unknown, do not invent a filename or page number.

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content, docs


def main():
    # The three clients below read their API keys from os.environ.
    # Do not overwrite these variables with empty strings or hard-coded keys.
    load_api_keys()

    embedding = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        truncate="END",
    )
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding,
    )
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
    )

    print(f"Connected to Pinecone Index: '{INDEX_NAME}'\n")

    questions = [
        "What is the leave policy?",
        "What is the refund policy?",
        "How should I report a security incident?",
        "What is the salary structure?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Query: {question}\n")

        answer, docs = answer_question(question, vector_store, llm)
        print("Answer:")
        print(answer)

        print("\n--- Retrieved Sources ---")
        if not docs:
            print("No documents retrieved.")
        for i, doc in enumerate(docs, 1):
            source, page = source_details(doc)
            print(f"[{i}] {source}, Page {page}")


if __name__ == "__main__":
    main()
