from unittest.mock import patch
from rag_knowledge_base.rag_pipeline import rag_pipeline
print("\n===== 测试 1：空 Query =====")
try:
    rag_pipeline("", [], None)

except ValueError as e:
    print("正确捕获异常：", e)

print("\n===== 测试 2：空检索结果 =====")
with patch("rag_knowledge_base.rag_pipeline.hybrid_search",return_value=[]):
    with patch("rag_knowledge_base.rag_pipeline.generate_answer") as mock_generate:

        result = rag_pipeline(query="测试问题",all_chunks=[],bm25=None)
        print(result)
        mock_generate.assert_not_called()
        print("Qwen 未被调用：通过")

# 准备模拟的检索结果和来源
fake_results = [
    {
        "chunk_id": "abc123",
        "text": "Multi-Head Attention 使用多个注意力头。",
        "source": "attention.pdf",
        "page": 3
    }
]

fake_citations = {
    1: {
        "source": "attention.pdf",
        "page": 3,
        "chunk_id": "abc123"
    }
}


# 用于模拟完整 Pipeline 的前几个阶段
def test_generated_answer(fake_answer):

    with (
        patch(
            "rag_knowledge_base.rag_pipeline.hybrid_search",
            return_value=fake_results
        ),
        patch(
            "rag_knowledge_base.rag_pipeline.rerank_results",
            return_value=fake_results
        ),
        patch(
            "rag_knowledge_base.rag_pipeline.build_context",
            return_value=("[1] 内容：Multi-Head Attention 使用多个注意力头。",
                          fake_citations)
        ),
        patch(
            "rag_knowledge_base.rag_pipeline.generate_answer",
            return_value=fake_answer
        )
    ):
        return rag_pipeline(query="Multi-Head Attention 有什么作用？",all_chunks=[],bm25=None)
# 测试 3：Qwen 生成空字符串
print("\n===== 测试 3：空生成结果 =====")
result = test_generated_answer("")
print(result)
# 测试 4：Qwen 编造无效引用
print("\n===== 测试 4：无效引用 =====")
result = test_generated_answer("多个注意力头可以学习不同的信息。[9]")
print(result)

from rag_knowledge_base.generation.context_builder import build_context
print("\n===== 测试 5：缺失页码 =====")
context, citations = build_context([
    {
        "chunk_id": "def456",
        "text": "RAG 可以利用外部知识库补充信息。",
        "source": "rag.docx",
        "page": None
    }
])

print(context)
print(citations)