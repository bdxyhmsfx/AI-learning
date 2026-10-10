import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker_model = CrossEncoder(MODEL_NAME,device=device,max_length=512)

def rerank_results(query,hybrid_results,k=3):
    if k<1:
        raise ValueError("k必须大于等于1")
    if not hybrid_results:
        return []
    pairs = [(query,result["text"]) for result in hybrid_results]
    scores = reranker_model.predict(pairs)
    reranked_results = []
    for result,score in zip(hybrid_results,scores):
        new_result = result.copy()
        new_result["rerank_score"] = score
        reranked_results.append(new_result)
    reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results[:k]