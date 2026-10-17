from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks

BASE_DIR = Path(__file__).resolve().parent.parent
documents = load_directory(BASE_DIR / "data")

all_chunks=[]
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
print("文档数量：", len(documents))
print("Chunk 数量：", len(all_chunks))

print("\n===== 已加载的文档 =====")
for document in documents:
    print(document["source"], "page =", document["page"])

for i, chunk in enumerate(all_chunks, start=1):

    print("\n" + "=" * 70)
    print("序号：", i)
    print("Chunk ID：", chunk["chunk_id"])
    print("Source：", chunk["source"])
    print("Page：", chunk["page"])
    print("Chunk Index：", chunk["chunk_index"])

    print("\n文本内容：")
    print(chunk["text"])

from rag_knowledge_base.vector_store.qdrant_store import client,COLLECTION_NAME
try:
    stored_ids = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )

        for point in points:
            stored_ids.add(str(point.id).replace("-", "").lower())

        if offset is None:
            break

    local_ids = {
        chunk["chunk_id"].replace("-", "").lower()
        for chunk in all_chunks
    }

    print("\n===== Qdrant 一致性检查 =====")
    print("本地 Chunk 数：", len(local_ids))
    print("Qdrant Point 数：", len(stored_ids))
    print("本地与 Qdrant ID 一致：", local_ids == stored_ids)

    if local_ids != stored_ids:
        print("仅在本地存在：", local_ids - stored_ids)
        print("仅在 Qdrant 存在：", stored_ids - local_ids)

finally:
    client.close()