from rag_knowledge_base.evaluation.retrieval_metrics import calculate_metrics,aggregate_metrics

gold_ids = ["A", "C"]
results = [
    {"chunk_id": "B"},
    {"chunk_id": "A"},
    {"chunk_id": "D"},
    {"chunk_id": "C"},
    {"chunk_id": "E"}
]

metrics_3 = calculate_metrics(results, gold_ids, k=3)

print("Top-3：", metrics_3)

metrics_5 = calculate_metrics(results, gold_ids, k=5)

print("Top-5：", metrics_5)

assert metrics_3["hit"] == 1
assert metrics_3["recall"] == 0.5
assert abs(metrics_3["precision"] - 1 / 3) < 1e-9
assert metrics_3["rr"] == 0.5

assert metrics_5["hit"] == 1
assert metrics_5["recall"] == 1.0
assert metrics_5["precision"] == 0.4
assert metrics_5["rr"] == 0.5

summary = aggregate_metrics([metrics_3, metrics_5])

print("平均指标：", summary)