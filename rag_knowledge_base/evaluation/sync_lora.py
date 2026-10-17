from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_file
from rag_knowledge_base.chunking.chunker import chunk_document
from rag_knowledge_base.vector_store.qdrant_store import embed_chunks,upsert_chunks,client,COLLECTION_NAME

BASE_DIR = Path(__file__).resolve().parent.parent
try:
    documents = load_file(BASE_DIR / "data" / "lora.txt")
    if not documents:
        raise ValueError("LoRA 文档加载失败")
    lora_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        lora_chunks.extend(chunks)
    print("LoRA Chunk 数量：", len(lora_chunks))

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError("Qdrant Collection 不存在")
    embeddings = embed_chunks(lora_chunks)
    upsert_chunks(lora_chunks, embeddings)

    result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    print("Qdrant 当前 Point 数量：", result.count)

finally:
    client.close()