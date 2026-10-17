def calculate_metrics(results, gold_chunk_ids, k=5):
    if k < 1:
        raise ValueError("k 必须大于 0")
    if not gold_chunk_ids:
        raise ValueError("gold_chunk_ids 不能为空")
    top_k_results = results[:k]
    retrieved_ids = [result["chunk_id"] for result in top_k_results]
    gold_set = set(gold_chunk_ids)
    retrieved_set = set(retrieved_ids)
    matched_ids = gold_set & retrieved_set
    hit = int(len(matched_ids) > 0)
    recall = len(matched_ids) / len(gold_set)
    precision = len(matched_ids) / k
    rr = 0
    for rank,chunk_id in enumerate(retrieved_ids,start=1):
        if chunk_id in gold_set:
            rr = 1/rank
            break
    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "rr": rr,
    }

def aggregate_metrics(metric_results):
    if not metric_results:
        raise ValueError("metric_results 不能为空")

    n = len(metric_results)

    return {
        "hit_rate": sum(m["hit"] for m in metric_results) / n,
        "recall": sum(m["recall"] for m in metric_results) / n,
        "precision": sum(m["precision"] for m in metric_results) / n,
        "mrr": sum(m["rr"] for m in metric_results) / n
    }