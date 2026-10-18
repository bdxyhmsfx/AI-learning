import json
from pathlib import Path
from unittest.mock import patch
from rag_knowledge_base.rag_pipeline import rag_pipeline

EVAL_DIR = Path(__file__).resolve().parent
RESULT_PATH = EVAL_DIR / "results" / "generation_results_scored.json"

NO_ANSWER = "根据当前提供的知识库资料，无法确定答案。"

def test_invalid_query():
    invalid_queries = ["", "   ", None, 123]
    for query in invalid_queries:
        try:
            rag_pipeline(query, [], None)
        except ValueError:
            print("PASS：非法 Query 被正确拒绝：", repr(query))
        else:
            raise AssertionError(f"非法 Query 未被拒绝：{repr(query)}")

def test_empty_retrieval():
    with patch(
        "rag_knowledge_base.rag_pipeline.hybrid_search",
        return_value=[]
    ), patch(
        "rag_knowledge_base.rag_pipeline.generate_answer"
    ) as mock_generate:
        result = rag_pipeline(
            "Multi-Head Attention 是什么？",
            [],
            None
        )
        assert result["answer"] == NO_ANSWER
        mock_generate.assert_not_called()
        print("PASS：检索为空时正确兜底，且未调用 LLM")

def test_empty_context():

    with patch(
        "rag_knowledge_base.rag_pipeline.hybrid_search",
        return_value=[{"chunk_id": "A", "text": "测试"}]
    ), patch(
        "rag_knowledge_base.rag_pipeline.rerank_results",
        return_value=[{"chunk_id": "A", "text": "测试"}]
    ), patch(
        "rag_knowledge_base.rag_pipeline.build_context",
        return_value=("", {})
    ), patch(
        "rag_knowledge_base.rag_pipeline.generate_answer"
    ) as mock_generate:
        result = rag_pipeline(
            "测试问题",
            [],
            None
        )
        assert result["answer"] == NO_ANSWER
        mock_generate.assert_not_called()
        print("PASS：Context 为空时正确兜底，且未调用 LLM")

def test_invalid_citation():
    available = {
        1: {
            "source": "lora.txt",
            "page": None,
            "chunk_id": "A"
        }
    }
    with patch(
        "rag_knowledge_base.rag_pipeline.hybrid_search",
        return_value=[{"chunk_id": "A", "text": "LoRA 相关资料"}]
    ), patch(
        "rag_knowledge_base.rag_pipeline.rerank_results",
        return_value=[{"chunk_id": "A", "text": "LoRA 相关资料"}]
    ), patch(
        "rag_knowledge_base.rag_pipeline.build_context",
        return_value=("[1] 来源:lora.txt\n内容：LoRA 相关资料", available)
    ), patch(
        "rag_knowledge_base.rag_pipeline.build_prompt",
        return_value=[{"role": "user", "content": "测试"}]
    ), patch(
        "rag_knowledge_base.rag_pipeline.generate_answer",
        return_value="LoRA 是一种微调方法。[9]"
    ):
        result = rag_pipeline("LoRA 是什么？", [], None)
        assert result["invalid_ids"] == [9]
        assert result["has_citations"] is True

        print("PASS：成功识别无效引用 [9]")

def test_retrieval_failure():
    with patch(
        "rag_knowledge_base.rag_pipeline.hybrid_search",
        side_effect=RuntimeError("模拟检索服务异常")
    ):
        try:
            rag_pipeline("测试问题", [], None)
        except RuntimeError as e:
            print("发现待改进问题：检索异常向调用方传播：", e)
        else:
            print("检索异常已被内部处理")
def test_out_of_scope():

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    cases = [record for record in records if not record["answerable"]]
    success = sum(record["scores"]["abstention"] == 1 for record in cases)
    print("\n知识库外问题数量：", len(cases))
    print("正确拒答数量：", success)
    for record in cases:
        print(record["id"],"Abstention：",record["scores"]["abstention"])

if __name__ == "__main__":
    print("===== Robustness Testing =====")
    test_invalid_query()
    test_empty_retrieval()
    test_empty_context()
    test_invalid_citation()
    test_retrieval_failure()
    test_out_of_scope()

    print("\nRobustness Testing 完成")