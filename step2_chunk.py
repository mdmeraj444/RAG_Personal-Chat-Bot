#In this file we will chunk data into small pieces so that it can answer smoothly
#Splitting long pages into smaller overlapping chunk

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from step1_load import data_dir

## 1. Load PDF documents

data_dir = Path(__file__).parent / "Data"

chunk_size = 1000
chunk_overlap=200

all_documents = []

for filepath in sorted(data_dir.glob("*.pdf")):

    loader = PyPDFLoader(str(filepath))
    pages = loader.load()

    all_documents.extend(pages)

    print(f"Loaded: {filepath.name} ----> {len(pages)} pages")

print(f"\nTotal pages loaded: {len(all_documents)}")

# -------------------------
# 2. Create chunks
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(all_documents) #Important it will break into small chunks


print(f"Total chunks created: {len(chunks)}")


# -------------------------
# 3. Preview first chunk
# -------------------------

if chunks:

    chunk = chunks[0]

    print("\n--- First Chunk ---")

    print(f'Source : {chunk.metadata["source"]}')
    print(f'Page   : {chunk.metadata["page"] + 1}')
    print(f'Length : {len(chunk.page_content)} characters')

    print("\nContent:")
    print(chunk.page_content)

