"""Load PDF pages, create embeddings, and upload chunks to Pinecone.

Place this script, your .env file, and the Data folder in the project folder.
This script preserves existing vectors. Stable IDs prevent additional copies
when unchanged chunks are uploaded again using this version of the script.
It does not remove vectors from older scripts or from edited/deleted PDFs.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
INDEX_NAME = "agentic-ai-rag"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
EMBEDDING_DIMENSION = 2048
METRIC = "cosine"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
UPLOAD_BATCH_SIZE = 32
INDEX_READY_TIMEOUT = 120


def load_chunks(data_dir):
    """Read PDF text and retain the source filename and zero-based page."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Create this folder and add your PDFs: {data_dir}")

    pdf_files = sorted(
        (path for path in data_dir.iterdir()
         if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found directly inside {data_dir}")

    documents = []
    for filepath in pdf_files:
        pages = PyPDFLoader(str(filepath)).load()
        text_pages = []
        for page in pages:
            # A filename works for citations on Windows and other computers.
            page.metadata["source"] = filepath.name
            if page.page_content.strip():
                text_pages.append(page)
        documents.extend(text_pages)
        print(
            f"Loaded: {filepath.name} -> {len(pages)} pages "
            f"({len(text_pages)} with text)"
        )
        if not text_pages:
            print("  No text extracted; scanned PDFs may need OCR first.")

    if not documents:
        raise ValueError("No PDF text was extracted. Scanned PDFs need OCR first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created; nothing was uploaded.")

    print(f"\nTotal pages with text: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def chunk_id(chunk):
    """Create a repeatable ID for an unchanged chunk at the same location."""
    identity = json.dumps(
        [
            chunk.metadata["source"],
            chunk.metadata.get("page"),
            chunk.metadata.get("start_index"),
            chunk.page_content,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def connect_index(pc):
    """Create only if missing, validate the configuration, and wait briefly."""
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"Reusing Pinecone index: {INDEX_NAME}")

    deadline = time.monotonic() + INDEX_READY_TIMEOUT
    while True:
        description = pc.describe_index(name=INDEX_NAME)
        if (description.dimension != EMBEDDING_DIMENSION
                or description.metric != METRIC):
            raise ValueError(
                f"Index '{INDEX_NAME}' has dimension={description.dimension}, "
                f"metric={description.metric}; expected "
                f"dimension={EMBEDDING_DIMENSION}, metric={METRIC}. "
                "Use a compatible index, or choose a new index name in your "
                "indexing, retrieval, and app scripts. No vectors were changed."
            )
        if description.status["ready"]:
            print("Index is ready!")
            return pc.Index(host=description.host)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Index '{INDEX_NAME}' was not ready within "
                f"{INDEX_READY_TIMEOUT} seconds. Check Pinecone and try again."
            )
        time.sleep(2)


def main():
    env_path = PROJECT_DIR / ".env"
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")

    required_keys = ("NVIDIA_API_KEY", "PINECONE_API_KEY")
    missing_keys = [key for key in required_keys if not os.getenv(key, "").strip()]
    if missing_keys:
        raise RuntimeError(
            "Missing API keys: " + ", ".join(missing_keys)
            + f". Add them to {env_path} or set them as environment variables."
        )

    # Check the local documents before creating any cloud resources.
    chunks = load_chunks(DATA_DIR)
    embedding = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = connect_index(pc)
    vector_store = PineconeVectorStore(index=index, embedding=embedding)

    print("\nCreating embeddings and uploading to Pinecone...")
    for start in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        batch = chunks[start:start + UPLOAD_BATCH_SIZE]
        vector_store.add_documents(
            documents=batch,
            ids=[chunk_id(chunk) for chunk in batch],
        )
        print(f"Uploaded {start + len(batch)}/{len(chunks)} chunks")

    print(f"\nDone: uploaded {len(chunks)} chunks to '{INDEX_NAME}'.")
    print("Pinecone may take a short time to make new vectors searchable.")


if __name__ == "__main__":
    main()
