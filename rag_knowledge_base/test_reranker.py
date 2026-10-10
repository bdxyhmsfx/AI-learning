from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.retrieval.reranker import rerank_results
from rag_knowledge_base.vector_store.qdrant_store import client

BASE_DIR = Path(__file__).resolve().parent
documents = load_directory(BASE_DIR/"data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
bm25 = build_bm25_index(all_chunks)
queries = [
    "为什么 Transformer 要使用多个注意力头？",
    "Scaled Dot-Product Attention",
    "RAG 和 Fine-Tuning 有什么区别？",
    "Causal Mask 的作用是什么？"
]
try:
    for query in queries:
        hybrid_results = hybrid_search(query=query,all_chunks=all_chunks,bm25=bm25,k=5,candidate_k=5)
        reranked_results = rerank_results(query=query,hybrid_results=hybrid_results,k=3)
        print("\n========== Hybrid Results ==========")

        for i, result in enumerate(hybrid_results, start=1):
            print("\n排名：", i)
            print("RRF Score：", result["score"])
            print("Source：", result["source"])
            print("Chunk ID：", result["chunk_id"])
            print("Text：", result["text"][:200])
        print("\n========== Reranked Results ==========")

        for i, result in enumerate(reranked_results, start=1):
            print("\n排名：", i)
            print("RRF Score：", result["score"])
            print("Rerank Score：", result["rerank_score"])
            print("Source：", result["source"])
            print("Chunk ID：", result["chunk_id"])
            print("Text：", result["text"][:200])

finally:
    client.close()