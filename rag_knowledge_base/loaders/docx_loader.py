from docx import Document
from pathlib import Path
from rag_knowledge_base.utils.cleaner import clean_text

def load_docx_file(file_path):
    file_path = Path(file_path)
    doc = Document(file_path)
    paragraphs = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)
    text = clean_text(text)

    document = {
        "text": text,
        "source": file_path.name,
        "file_type": file_path.suffix.lower(),
        "page": None,
        "title": file_path.stem
    }

    return document