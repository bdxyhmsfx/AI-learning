from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import (chunk_document,deduplicate_chunks)
from rag_knowledge_base.vector_store.qdrant_store import (embed_chunks,create_collection,upsert_chunks,client,COLLECTION_NAME,search_chunks)

documents = load_directory(r"E:\daily_code\rag_knowledge_base\data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
embeddings = embed_chunks(all_chunks)

vector_size = embeddings.shape[1]
create_collection(vector_size)
upsert_chunks(all_chunks,embeddings)
count_result = client.count(collection_name=COLLECTION_NAME,exact=True)
print("Qdrant Point 数量：",count_result.count)

query = "Multi-Head Attention 有什么作用？"
results = search_chunks(query,k=3)
for i, result in enumerate(results, start=1):
    print("\n" + "=" * 60)
    print("排名：", i)
    print("score:",result["score"])
    print("source:",result["source"])
    print("page:",result["page"])
    print("chunk_index:",result["chunk_index"])
    print("\ntext:")
    print(result["text"])
client.close()