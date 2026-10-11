from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.retrieval.reranker import rerank_results, reranker_model
from rag_knowledge_base.vector_store.qdrant_store import client
import torch
from time import perf_counter


def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


BASE_DIR = Path(__file__).resolve().parent
documents = load_directory(BASE_DIR / "data")
all_chunks = []
for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)
all_chunks = deduplicate_chunks(all_chunks)
bm25 = build_bm25_index(all_chunks)
query = "RAG 和 Fine-Tuning 有什么区别？"
#query = "Causal Mask 的作用是什么？"
try:
    sync_gpu()
    start = perf_counter()
    hybrid_results = hybrid_search(query=query, all_chunks=all_chunks, bm25=bm25, k=5, candidate_k=5)

    sync_gpu()
    hybrid_end = perf_counter()
    reranked_results = rerank_results(query=query, hybrid_results=hybrid_results, k=3)
    sync_gpu()
    rerank_end = perf_counter()

    tokenizer = reranker_model.tokenizer
    for i, result in enumerate(hybrid_results, start=1):
        encoding = tokenizer(query, result["text"], add_special_tokens=True, truncation=False)
        token_count = len(encoding["input_ids"])
        print(
            f"Chunk {i}：{token_count} Tokens",
            f"是否超长：{token_count > 512}"
        )

    hybrid_time = hybrid_end - start
    rerank_time = rerank_end - hybrid_end
    total_time = rerank_end - start

    print("\n===== 耗时统计 =====")
    print(f"Hybrid：{hybrid_time:.4f} 秒")
    print(f"Reranker：{rerank_time:.4f} 秒")
    print(f"总耗时：{total_time:.4f} 秒")

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
    '''测试候选找回数量'''
    # for candidate_k in [5, 9]:
    #
    #     hybrid_results = hybrid_search(
    #         query=query,
    #         all_chunks=all_chunks,
    #         bm25=bm25,
    #         k=candidate_k,
    #         candidate_k=candidate_k
    #     )
    #
    #     reranked_results = rerank_results(
    #         query=query,
    #         hybrid_results=hybrid_results,
    #         k=3
    #     )
    #
    #     print(f"\n===== Candidate K = {candidate_k} =====")
    #
    #     for i, result in enumerate(reranked_results, start=1):
    #
    #         print("\n排名：", i)
    #         print("Chunk ID：", result["chunk_id"])
    #         print("Rerank Score：", result["rerank_score"])
    #         print("Text：", result["text"])

finally:
    client.close()
