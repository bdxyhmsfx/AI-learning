from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import (chunk_document,deduplicate_chunks)
from rag_knowledge_base.retrieval.bm25_retriever import (build_bm25_index)
from rag_knowledge_base.retrieval.hybrid_retriever import (hybrid_search)
from rag_knowledge_base.vector_store.qdrant_store import client


# 1. 加载 Document
BASE_DIR = Path(__file__).resolve().parent
documents = load_directory(BASE_DIR / "data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)

bm25 = build_bm25_index(all_chunks)

query = "Multi-Head Attention 有什么作用？"

try:
    results = hybrid_search(
        query=query,
        all_chunks=all_chunks,
        bm25=bm25,
        k=3,
        candidate_k=5
    )

    print("Query:", query)
    print("Hybrid Results:", len(results))

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print("排名：", i)
        print("chunk_id：", result["chunk_id"])
        print("RRF Score：", result["score"])
        print("Retriever：", result["retriever"])
        print("Source：", result["source"])
        print("Page：", result["page"])

        print("\nText：")
        print(result["text"])

finally:
    client.close()