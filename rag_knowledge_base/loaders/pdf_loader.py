from pathlib import Path
from rag_knowledge_base.utils.cleaner import clean_text
from pypdf import PdfReader

def load_pdf_file(file_path):
    file_path = Path(file_path)
    reader = PdfReader(file_path)
    documents = []
    for page_index,page in enumerate(reader.pages):
        text = page.extract_text()
        if text is None:
            continue
        text = clean_text(text)

        if not text:
            continue
        document={
            "text": text,
            "source": file_path.name,
            "file_type": file_path.suffix.lower(),
            "page": page_index+1,
            "title": file_path.stem
        }
        documents.append(document)
    return documents
