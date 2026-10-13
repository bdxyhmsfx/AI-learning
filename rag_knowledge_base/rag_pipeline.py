from rag_knowledge_base.retrieval.hybrid_retriever import hybrid_search
from rag_knowledge_base.retrieval.reranker import rerank_results

from rag_knowledge_base.generation.context_builder import build_context
from rag_knowledge_base.generation.prompt_builder import build_prompt
from rag_knowledge_base.generation.llm_generator import generate_answer
from rag_knowledge_base.generation.citation import validate_citations

def rag_pipeline(query,all_chunks,bm25):
    if not isinstance(query,str) or not query.strip():
        raise ValueError("query 不能为空")
    query=query.strip()
    empty_result = {
        "answer": "根据当前提供的知识库资料，无法确定答案。",
        "citations": {},
        "invalid_ids": [],
        "has_citations": False
    }
    hybrid_results  = hybrid_search(query=query,all_chunks=all_chunks,bm25=bm25,k=5,candidate_k=5);
    if not hybrid_results:
        return empty_result
    reranked_results = rerank_results(query=query,hybrid_results=hybrid_results,k=3)
    context,citations=build_context(reranked_results)
    if not context:
        return empty_result
    prompt=build_prompt(query=query,context=context)
    answer=generate_answer(prompt)
    if not answer:
        return empty_result
    citation_result = validate_citations(answer,citations)
    result = {
        "answer": answer,
        "citations": citation_result["used_citations"],
        "invalid_ids": citation_result["invalid_ids"],
        "has_citations": citation_result["has_citations"]
    }

    return result