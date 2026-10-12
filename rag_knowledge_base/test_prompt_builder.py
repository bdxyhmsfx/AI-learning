from rag_knowledge_base.generation.context_builder import build_context
from rag_knowledge_base.generation.prompt_builder import build_prompt

reranked_results = [
    {
        "chunk_id": "abc123",
        "text": "Multi-Head Attention 可以从不同表示子空间学习信息。",
        "source": "attention.pdf",
        "page": 3
    },
    {
        "chunk_id": "def456",
        "text": "Transformer 使用 Attention 机制处理序列信息。",
        "source": "transformer.md",
        "page": None
    }
]
query = "为什么 Transformer 要使用多个注意力头？"
context, citations = build_context(reranked_results)
prompt = build_prompt(query, context)

print("========== System ==========")
print(prompt[0]["content"])

print("\n========== User ==========")
print(prompt[1]["content"])

print("\n========== 类型检查 ==========")
print("Prompt 类型：", type(prompt))
print("第一条消息类型：", type(prompt[0]))
print("消息数量：", len(prompt))

empty_prompt = build_prompt(
    query="什么是 Kubernetes Pod？",
    context=""
)

print("\n========== 空 Context 测试 ==========")
print(empty_prompt[1]["content"])