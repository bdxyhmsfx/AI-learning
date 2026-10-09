from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import (chunk_document,deduplicate_chunks)
from rag_knowledge_base.retrieval.bm25_retriever import (build_bm25_index,bm25_search)
from rag_knowledge_base.retrieval.hybrid_retriever import (hybrid_search)
from rag_knowledge_base.vector_store.qdrant_store import (search_chunks,client)

BASE_DIR = Path(__file__).resolve().parent
documents = load_directory(BASE_DIR / "data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
bm25 = build_bm25_index(all_chunks)

def print_results(name, results):
    print(f"\n===== {name} =====")
    for i, result in enumerate(results, start=1):
        print(f"\n排名：{i}")
        print("score:", round(result["score"], 4))
        print("source:", result["source"])
        print("page:", result["page"])
        print("chunk_id:", result["chunk_id"])
        print(
            "text:",
            result["text"][:150].replace("\n", " ")
        )
queries = [
    "为什么 Transformer 要使用多个注意力头？",
    "Scaled Dot-Product Attention",
    "RAG 和 Fine-Tuning 有什么区别？",
    "Causal Mask 的作用是什么？"
]

try:
    for query in queries:
        print("\n" + "#" * 70)
        print("Query:", query)
        dense_results = search_chunks(query,k=3)

        bm25_results = bm25_search(query,all_chunks,bm25,k=3)

        hybrid_results = hybrid_search(query,all_chunks,bm25,k=3,candidate_k=5)

        print_results("Dense", dense_results)
        print_results("BM25", bm25_results)
        print_results("Hybrid", hybrid_results)
finally:
    client.close()