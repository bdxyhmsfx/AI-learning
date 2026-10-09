from transformers import AutoTokenizer
import hashlib
from rag_knowledge_base.loaders.document_loader import load_directory

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def split_text_by_tokens(text,chunk_size=200,chunk_overlap=30):
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size 和 chunk_overlap 参数不合法")
    encoding=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoding["offset_mapping"]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(len(offsets),start+chunk_size)
        start_char = offsets[start][0]
        end_char = offsets[end-1][1]
        chunk_text = text[start_char:end_char]
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(offsets):
            break
        start = end - chunk_overlap
    return chunks

def create_document_id(source):
    return hashlib.md5(source.encode("utf-8")).hexdigest()
def create_chunk_id(document_id,page,chunk_index):
    raw=(f"{document_id}-{page}-{chunk_index}")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def chunk_document(document,chunk_size=200,chunk_overlap=30):
    text_chunks = split_text_by_tokens(document["text"],chunk_size,chunk_overlap)
    document_id = create_document_id(document["source"])
    chunks = []
    for chunk_index,chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue
        chunk_id = create_chunk_id(document_id,document["page"],chunk_index)
        chunk={
            "chunk_id":chunk_id,
            "document_id":document_id,
            "text":chunk_text,
            "source":document["source"],
            "file_type":document["file_type"],
            "page":document["page"],
            "title":document["title"],
            "chunk_index":chunk_index,
            "token_count":len(tokenizer.encode(chunk_text,add_special_tokens=False))
        }
        chunks.append(chunk)
    return chunks

def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if text in seen:
            continue
        seen.add(text)
        unique_chunks.append(chunk)
    return unique_chunks

def print_chunk_statistics(chunks):
    if not chunks:
        print("没有 Chunk")
        return
    token_counts=[chunk["token_count"] for chunk in chunks]
    total_chunks=len(chunks)
    avg_tokens = sum(token_counts) / total_chunks
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    short_chunks = sum(1 for count in token_counts if count < 30)
    print("\n===== Chunk Statistics =====")
    print("Chunk 总数：", total_chunks)
    print("平均 Token 数：",round(avg_tokens, 2))
    print("最短 Chunk：",min_tokens)
    print("最长 Chunk：",max_tokens)
    print("小于 30 Tokens 的 Chunk：",short_chunks)

if __name__ == "__main__":
    documents = load_directory(r"E:\daily_code\rag_knowledge_base\data")
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    print("去重前 Chunk 数量：",len(all_chunks))
    all_chunks = deduplicate_chunks(all_chunks)
    print("去重后 Chunk 数量：",len(all_chunks))
    print_chunk_statistics(all_chunks)