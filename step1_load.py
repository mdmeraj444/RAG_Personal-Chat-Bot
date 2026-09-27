from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

data_dir = Path(__file__).parent / "Data"

all_documents = []

for filepath in sorted(data_dir.glob("*.pdf")):

    loader = PyPDFLoader(str(filepath))
    pages = loader.load()

    all_documents.extend(pages)

    print(f"Loaded: {filepath.name} ----> {len(pages)} pages")


print(f"\nTotal documents loaded: {len(all_documents)} pages")

if all_documents:

    print("\n--- Preview of first page ---")

    doc = all_documents[0]

    print(f'Source  : {doc.metadata["source"]}')
    print(f'Page    : {doc.metadata["page"] + 1}')
    print(f'Content : {doc.page_content[:300]}')

else:

    print("No PDF files found.")