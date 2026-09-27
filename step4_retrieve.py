"""Retrieve relevant chunks from an existing Pinecone index."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "agentic-ai-rag"
TOP_K = 3
PREVIEW_CHARS = 250


def source_details(doc):
    """Return the filename and a one-based PDF page number, if available."""
    source = str(doc.metadata.get("source") or "Unknown document")
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


def main():
    # Load .env from the same folder as this script.
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")

    required_keys = ("NVIDIA_API_KEY", "PINECONE_API_KEY")
    missing_keys = [
        key for key in required_keys if not os.getenv(key, "").strip()
    ]
    if missing_keys:
        raise RuntimeError(
            "Missing API keys: " + ", ".join(missing_keys)
            + f". Add them to {env_path} or set them as environment variables."
        )

    embedding = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        truncate="END",
    )

    # Connect to the index containing your previously uploaded PDF chunks.
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding,
    )
    print(f"Connected to Pinecone index: '{INDEX_NAME}'\n")

    question = "What is the leave policy?"
    print(f"Query: {question}\n")

    results = vector_store.similarity_search(question, k=TOP_K)
    print(f"Retrieved {len(results)} result(s), requested up to {TOP_K}:\n")

    if not results:
        print("No chunks were returned. Check the index and its namespace.")
        return

    for i, doc in enumerate(results, 1):
        source, page = source_details(doc)
        preview = doc.page_content[:PREVIEW_CHARS]
        if len(doc.page_content) > PREVIEW_CHARS:
            preview += "..."

        print(f"--- Result {i} | {source}, Page {page} ---")
        print(preview)
        print()


if __name__ == "__main__":
    main()
