from pathlib import Path
from rag_knowledge_base.utils.cleaner import clean_text

def load_text_file(file_path):
    file_path = Path(file_path)
    with open(file_path,"r",encoding="utf-8") as f:
        text = f.read()
    text = clean_text(text)
    document = {
        "text": text,
        "source": file_path.name,
        "file_type": file_path.suffix.lower(),
        "page": None,
        "title": file_path.stem
    }
    return document
