from pathlib import Path
from rag_knowledge_base.loaders.pdf_loader import load_pdf_file
from rag_knowledge_base.loaders.text_loader import load_text_file
from rag_knowledge_base.loaders.docx_loader import load_docx_file

def load_file(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        documents = load_pdf_file(file_path)
    elif suffix == ".docx":
        document = load_docx_file(file_path)
        documents = [document]
    elif suffix in [".txt",".text",".md"]:
        document = load_text_file(file_path)
        documents = [document]
    else:
        documents = []
    return documents

def load_directory(directory_path):
    directory_path = Path(directory_path)
    all_documents = []
    for file_path in directory_path.iterdir():
        if not file_path.is_file():
            continue
        documents = load_file(file_path)
        all_documents.extend(documents)
    return all_documents