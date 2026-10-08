from rag_knowledge_base.retrieval.bm25_retriever import bm25_search
from rag_knowledge_base.vector_store.qdrant_store import search_chunks

def rrf_fusion(dense_results,bm25_results,top_k=5,rrf_k=60):
    fused={}
    for result_list in (dense_results,bm25_results):
        for rank,result in enumerate(result_list,start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id]=result.copy()
                fused[chunk_id]["score"]=0.0
                fused[chunk_id]["retriever"]="hybrid"
            fused[chunk_id]["score"]+=1/(rrf_k+rank)
    sorted_results = sorted(fused.values(),key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]

def hybrid_search(query,all_chunks,bm25,k=3,candidate_k = 5):
    if k < 1 or candidate_k < k:
        raise ValueError("需要满足 1 <= k <= candidate_k")
    dense_results = search_chunks(query,k=candidate_k)
    bm25_results = bm25_search(query,all_chunks,bm25,k=candidate_k)
    hybrid_results = rrf_fusion(dense_results,bm25_results,top_k=k)
    return hybrid_results