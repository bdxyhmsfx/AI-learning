from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import (chunk_document,deduplicate_chunks)
from rag_knowledge_base.retrieval.bm25_retriever import (build_bm25_index,bm25_search)


documents = load_directory(r"E:\daily_code\rag_knowledge_base\data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
bm25 = build_bm25_index(all_chunks)
query = "Multi-Head Attention 有什么作用？"
results = bm25_search(query=query,chunks=all_chunks,bm25=bm25,k=3)

# for i, result in enumerate(results,start=1):
#     print("\n" + "=" * 60)
#     print("排名：",i)
#     print("score：",result["score"])
#     print("source：",result["source"])
#     print("page：",result["page"])
#     print("chunk_index：",result["chunk_index"])
#     print("\ntext：")
#     print(result["text"])
from rag_knowledge_base.vector_store.qdrant_store import (search_chunks,client)
from rag_knowledge_base.retrieval.hybrid_retriever import (rrf_fusion)
query = "Multi-Head Attention 有什么作用？"
dense_results = search_chunks(query,k=5)
bm25_results = bm25_search(query,all_chunks,bm25,k=5)
hybrid_results = rrf_fusion(dense_results,bm25_results,top_k=3)
for i, result in enumerate(hybrid_results, start=1):
    print("\n" + "=" * 60)
    print("排名：", i)
    print("chunk_id：", result["chunk_id"])
    print("RRF Score：", result["score"])
    print("Retriever：", result["retriever"])
    print("Source：", result["source"])
    print("Page：", result["page"])
    print("\nText：")
    print(result["text"])
client.close()