import json
from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document,deduplicate_chunks
from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.retrieval.reranker import rerank_results
from rag_knowledge_base.vector_store.qdrant_store import client
from rag_knowledge_base.evaluation.retrieval_metrics import calculate_metrics,aggregate_metrics
BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_K = 10
FINAL_K = 3
try:
    documents = load_directory(BASE_DIR / "data")
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    bm25 = build_bm25_index(all_chunks)
    dataset_path = (BASE_DIR / "evaluation" / "datasets" / "eval_cases.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    metric_records = {
        "Hybrid": [],
        "Reranker": []
    }
    candidate_records = []
    for case in eval_cases:
        if not case["answerable"]:
            continue
        query = case["question"]
        gold_ids = case["gold_chunk_ids"]
        hybrid_candidates = hybrid_search(
            query=query,
            all_chunks=all_chunks,
            bm25=bm25,
            k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )
        hybrid_top3 = hybrid_candidates[:FINAL_K]
        reranked_top3 = rerank_results(
            query=query,
            hybrid_results=hybrid_candidates,
            k=FINAL_K
        )

        candidate_metrics = calculate_metrics(
            hybrid_candidates,
            gold_ids,
            k=CANDIDATE_K
        )
        candidate_records.append(candidate_metrics)
        results_dict = {
            "Hybrid": hybrid_top3,
            "Reranker": reranked_top3
        }

        print("\n" + "=" * 70)
        print("问题：", case["id"], query)
        print(
            "Candidate Recall@10：",
            round(candidate_metrics["recall"], 3)
        )
        for name, results in results_dict.items():
            metrics = calculate_metrics(
                results,
                gold_ids,
                k=FINAL_K
            )
            metric_records[name].append(metrics)
            print(
                f"{name:<9}"
                f"Hit={metrics['hit']}  "
                f"Recall={metrics['recall']:.3f}  "
                f"Precision={metrics['precision']:.3f}  "
                f"RR={metrics['rr']:.3f}"
            )
    print("\n" + "=" * 70)
    print("Reranker Evaluation Summary @ K=3")
    summaries = {}
    for name, records in metric_records.items():
        summary = aggregate_metrics(records)
        summaries[name] = summary
        print(f"\n{name}")
        print("Hit Rate@3：", round(summary["hit_rate"], 4))
        print("Recall@3：", round(summary["recall"], 4))
        print("Precision@3：", round(summary["precision"], 4))
        print("MRR@3：", round(summary["mrr"], 4))
    candidate_summary = aggregate_metrics(candidate_records)
    print("\nCandidate Recall@10：",
          round(candidate_summary["recall"], 4))
    print("\n===== Reranker 改进幅度 =====")
    for key in ["hit_rate", "recall", "precision", "mrr"]:
        delta = summaries["Reranker"][key] - summaries["Hybrid"][key]
        print(f"{key}：{delta:+.4f}")
    print("\n评估问题数量：", len(metric_records["Hybrid"]))
finally:
    client.close()