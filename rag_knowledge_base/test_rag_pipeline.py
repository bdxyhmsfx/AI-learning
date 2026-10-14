from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
from rag_knowledge_base.vector_store.qdrant_store import client
from rag_knowledge_base.rag_pipeline import rag_pipeline

BASE_DIR = Path(__file__).resolve().parent
try:
    documents = load_directory(BASE_DIR / "data")
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    bm25 = build_bm25_index(all_chunks)

    queries = [
        "Multi-Head Attention 有什么作用？",
        "RAG 和 Fine-Tuning 有什么区别？",
        "Causal Mask 的作用是什么？",
        "什么是 Kubernetes 的 Pod？"
    ]
    for query in queries:
        result = rag_pipeline(query=query,all_chunks=all_chunks,bm25=bm25)
        print("\n========== Query ==========")
        print(query)

        print("\n========== Answer ==========")
        print(result["answer"])

        print("\n========== Citations ==========")

        for citation_id, metadata in result["citations"].items():

            source = metadata["source"]
            page = metadata["page"]
            chunk_id = metadata["chunk_id"]

            if page is not None:
                print(f"[{citation_id}] {source} · 第{page}页")
            else:
                print(f"[{citation_id}] {source}")

            print("Chunk ID：", chunk_id)

        print("\n========== Citation Check ==========")
        print("是否包含引用：", result["has_citations"])
        print("无效引用编号：", result["invalid_ids"])

        if result["invalid_ids"]:
            print("警告：回答包含无效引用编号，需要检查后再展示。")

        if not result["has_citations"]:
            print("提示：回答没有提供引用编号。")

finally:
    client.close()