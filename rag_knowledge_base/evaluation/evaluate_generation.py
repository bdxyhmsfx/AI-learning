import json
from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks
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
TEST_MODE = False
if TEST_MODE:
    selected_ids = {"q001", "q008", "q009", "q014"}
    output_path = EVAL_DIR / "results" / "generation_smoke.json"
else:
    selected_ids = None
    output_path = EVAL_DIR / "results" / "generation_results.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
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
    records = []
    for case in eval_cases:
        if selected_ids is not None and case["id"] not in selected_ids:
            continue
        query = case["question"]
        hybrid_results = hybrid_search(
            query=query,
            all_chunks=all_chunks,
            bm25=bm25,
            k=5,
            candidate_k=5
        )
        reranked_results = rerank_results(
            query=query,
            hybrid_results=hybrid_results,
            k=3
        )
        context, citations = build_context(reranked_results)
        if not context:
            answer = "根据当前提供的知识库资料，无法确定答案。"
        else:
            prompt = build_prompt(query, context)
            answer = generate_answer(prompt)
            if not answer:
                answer = "根据当前提供的知识库资料，无法确定答案。"
        citation_result = validate_citations(answer, citations)
        record = {
            "id": case["id"],
            "question": query,
            "answerable": case["answerable"],
            "gold_chunk_ids": case["gold_chunk_ids"],
            "expected_facts": case["expected_facts"],
            "category": case["category"],

            "retrieved_chunk_ids": [
                item["chunk_id"] for item in reranked_results
            ],

            "context": context,
            "answer": answer,

            "available_citations": citations,
            "used_citations": citation_result["used_citations"],
            "invalid_ids": citation_result["invalid_ids"],
            "has_citations": citation_result["has_citations"],

            "scores": {
                "relevance": None,
                "correctness": None,
                "faithfulness": None,
                "abstention": None
            },
            "notes": ""
        }
        records.append(record)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 65)
        print("问题：", case["id"], query)
        print("回答：", answer)
        print("引用编号有效性：", citation_result["invalid_ids"])
    print("\n已完成问题数量：", len(records))
    print("结果保存位置：", output_path)
finally:
    client.close()
