from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

def load_all_documents(data_dir: str) -> List[Any]:
    """Load documents and consolidate multi-page files into single Document objects"""
    data_path = Path(data_dir).resolve()
    print("Loading documents from: {data_path}")
    documents = []

    # PDF files - CONSOLIDATE PAGES
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()  # Returns list of pages
            
            # Consolidate all pages into one document
            full_text = "\n\n".join([page.page_content for page in pages])
            
            consolidated_doc = Document(
                page_content=full_text,
                metadata={
                    'source': str(pdf_file),
                    'name': pdf_file.name,
                    'type': 'pdf',
                    'pages': len(pages)
                }
            )
            
            print(f"  ✓ Loaded {pdf_file.name} ({len(pages)} pages)")
            documents.append(consolidated_doc)
        except Exception as e:
            print(f"  ✗ Failed to load {pdf_file.name}: {e}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = str(txt_file)
                doc.metadata['name'] = txt_file.name
                doc.metadata['type'] = 'txt'
            print(f"  ✓ Loaded {txt_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ Failed to load {txt_file.name}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = str(csv_file)
                doc.metadata['name'] = csv_file.name
                doc.metadata['type'] = 'csv'
            print(f"  ✓ Loaded {csv_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file.name}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = str(xlsx_file)
                doc.metadata['name'] = xlsx_file.name
                doc.metadata['type'] = 'xlsx'
            print(f"  ✓ Loaded {xlsx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ Failed to load {xlsx_file.name}: {e}")

    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = str(docx_file)
                doc.metadata['name'] = docx_file.name
                doc.metadata['type'] = 'docx'
            print(f"  ✓ Loaded {docx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ Failed to load {docx_file.name}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema='.',
                text_content=False
            )
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = str(json_file)
                doc.metadata['name'] = json_file.name
                doc.metadata['type'] = 'json'
            print(f"  ✓ Loaded {json_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ Failed to load {json_file.name}: {e}")
    
    print(f"\n[SUCCESS] Loaded {len(documents)} document files")
    
    # Show summary
    if documents:
        print(f"\n[INFO] Sample metadata from first document:")
        print(f"  source: {documents[0].metadata.get('source')}")
        print(f"  name: {documents[0].metadata.get('name')}")
        print(f"  type: {documents[0].metadata.get('type')}")
        # print(f"  Content preview: {documents[0].page_content[:250]}...")
    
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"\nLoaded {len(docs)} documents.")
    if docs:
        # print("\nFirst document metadata:", docs[0].metadata)
        print("\nFirst document preview:", docs[0].page_content[:200])