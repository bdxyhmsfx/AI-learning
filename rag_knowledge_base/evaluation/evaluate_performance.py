import json
import math
import time
import statistics
from pathlib import Path
import torch
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document,deduplicate_chunks
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.retrieval.reranker import rerank_results
from rag_knowledge_base.generation.context_builder import build_context
from rag_knowledge_base.generation.prompt_builder import build_prompt
from rag_knowledge_base.generation.llm_generator import generate_answer
from rag_knowledge_base.generation.citation import validate_citations
from rag_knowledge_base.vector_store.qdrant_store import client


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "datasets" / "eval_cases.json"
OUTPUT_PATH = EVAL_DIR / "results" / "performance_results.json"
TEST_MODE = False
TEST_IDS = {"q001", "q003", "q008", "q011", "q014"}
NO_ANSWER = "根据当前提供的知识库资料，无法确定答案。"

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_call(func, *args, **kwargs):
    sync_cuda()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    sync_cuda()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms

def measure_rag(query, all_chunks, bm25):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    query = query.strip()
    times = {
        "retrieval_ms": 0.0,
        "reranker_ms": 0.0,
        "context_ms": 0.0,
        "prompt_ms": 0.0,
        "generation_ms": 0.0,
        "citation_ms": 0.0
    }

    sync_cuda()
    total_start = time.perf_counter()

    def finish(answer, has_citations=False, invalid_ids=None):
        sync_cuda()
        times["total_ms"] = (time.perf_counter() - total_start) * 1000

        return {
            "answer": answer,
            "times_ms": times,
            "has_citations": has_citations,
            "invalid_ids": invalid_ids or []
        }

    # Retrieval
    hybrid_results, times["retrieval_ms"] = timed_call(
        hybrid_search,
        query,
        all_chunks,
        bm25,
        k=5,
        candidate_k=5
    )

    if not hybrid_results:
        return finish(NO_ANSWER)

    # Reranker
    reranked_results, times["reranker_ms"] = timed_call(
        rerank_results,
        query,
        hybrid_results,
        k=3
    )

    # Context
    (context, citations), times["context_ms"] = timed_call(
        build_context,
        reranked_results
    )

    if not context:
        return finish(NO_ANSWER)

    # Prompt
    prompt, times["prompt_ms"] = timed_call(
        build_prompt,
        query,
        context
    )

    # Generation
    answer, times["generation_ms"] = timed_call(
        generate_answer,
        prompt
    )

    if not answer:
        return finish(NO_ANSWER)

    # Citation
    citation_result, times["citation_ms"] = timed_call(
        validate_citations,
        answer,
        citations
    )

    return finish(answer,citation_result["has_citations"],citation_result["invalid_ids"])

#  计算百分位数（Nearest-Rank 方法）
def percentile(values, p):
    values = sorted(values)
    index = math.ceil(len(values) * p / 100) - 1
    return values[max(0, index)]

def main():

    documents = load_directory(BASE_DIR / "data")
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document))
    all_chunks = deduplicate_chunks(all_chunks)
    bm25 = build_bm25_index(all_chunks)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    if TEST_MODE:
        selected_cases = [case for case in eval_cases if case["id"] in TEST_IDS]
    else:
        selected_cases = eval_cases
    if not selected_cases:
        raise ValueError("没有可测试的问题")

    # 先执行一次预热，不参与正式统计
    print("正在进行模型预热……")
    measure_rag(
        selected_cases[0]["question"],
        all_chunks,
        bm25
    )
    print("预热完成，开始正式测试")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in selected_cases:
        # 重置单次请求的 PyTorch 显存峰值统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = measure_rag(case["question"],all_chunks,bm25)
        peak_gpu_mb = None
        if torch.cuda.is_available():
            peak_gpu_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2))
        record = {
            "id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "answer": result["answer"],
            "times_ms": result["times_ms"],
            "peak_gpu_mb": peak_gpu_mb
        }
        records.append(record)
        print("\n" + "=" * 60)
        print("问题：", case["id"])
        print("Retrieval：", round(result["times_ms"]["retrieval_ms"], 2), "ms")
        print("Reranker：", round(result["times_ms"]["reranker_ms"], 2), "ms")
        print("Generation：", round(result["times_ms"]["generation_ms"], 2), "ms")
        print("Total：", round(result["times_ms"]["total_ms"], 2), "ms")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    # 汇总各个阶段的平均耗时
    print("\n" + "=" * 60)
    print("Performance Summary")
    stage_names = [
        "retrieval_ms",
        "reranker_ms",
        "context_ms",
        "prompt_ms",
        "generation_ms",
        "citation_ms",
        "total_ms"
    ]
    for stage in stage_names:
        values = [record["times_ms"][stage]for record in records]
        print(f"{stage}: {statistics.mean(values):.2f} ms")
    # 端到端延迟分布
    latencies = [
        record["times_ms"]["total_ms"]
        for record in records
    ]
    print("\nMean：", round(statistics.mean(latencies), 2), "ms")
    print("P50：", round(statistics.median(latencies), 2), "ms")
    print("P95：", round(percentile(latencies, 95), 2), "ms")
    print("\n测试数量：", len(records))
    print("结果保存至：", OUTPUT_PATH)
try:
    main()
finally:
    client.close()