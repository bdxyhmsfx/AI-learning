import json
from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document,deduplicate_chunks
from rag_knowledge_base.vector_store.qdrant_store import search_chunks,client
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index,bm25_search
from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.evaluation.retrieval_metrics import calculate_metrics,aggregate_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent

K = 5
CANDIDATE_K = 10
try:

    documents = load_directory(BASE_DIR / "data")
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    bm25 = build_bm25_index(all_chunks)
    dataset_path = EVAL_DIR / "datasets" / "eval_cases.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    metric_records = {
        "Dense": [],
        "BM25": [],
        "Hybrid": []
    }
    for case in eval_cases:
        if not case["answerable"]:
            continue
        query = case["question"]
        gold_ids = case["gold_chunk_ids"]

        dense_results = search_chunks(query,k=K)
        bm25_results = bm25_search(query,all_chunks,bm25,k=K)
        hybrid_results = hybrid_search(query,all_chunks,bm25,k=K,candidate_k=CANDIDATE_K)
        results_dict = {
            "Dense": dense_results,
            "BM25": bm25_results,
            "Hybrid": hybrid_results
        }
        print("\n" + "=" * 70)
        print("问题：", case["id"], query)
        for name, results in results_dict.items():
            metrics = calculate_metrics(results,gold_ids,k=K)
            metric_records[name].append(metrics)
            print(
                f"{name:<8}"
                f"Hit={metrics['hit']}  "
                f"Recall={metrics['recall']:.3f}  "
                f"Precision={metrics['precision']:.3f}  "
                f"RR={metrics['rr']:.3f}"
            )
    print("\n" + "=" * 70)
    print(f"Retrieval Evaluation Summary @ K={K}")
    for name, records in metric_records.items():
        summary = aggregate_metrics(records)
        print(f"\n{name}")
        print(f"Hit Rate@{K}：", round(summary["hit_rate"], 4))
        print(f"Recall@{K}：", round(summary["recall"], 4))
        print(f"Precision@{K}：", round(summary["precision"], 4))
        print(f"MRR@{K}：", round(summary["mrr"], 4))
    print("\n实际评估问题数量：", len(metric_records["Dense"]))
finally:
    client.close()