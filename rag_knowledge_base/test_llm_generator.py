from rag_knowledge_base.generation.context_builder import build_context
from rag_knowledge_base.generation.prompt_builder import build_prompt
from rag_knowledge_base.generation.llm_generator import generate_answer

reranked_results = [
    {
        "chunk_id": "abc123",
        "text": (
            "Multi-Head Attention 会使用多个注意力头，"
            "每个注意力头可以从不同的表示子空间学习信息。"
            "不同注意力头可以关注不同类型的关系，"
            "例如局部信息和长距离依赖。"
        ),
        "source": "attention.pdf",
        "page": 3
    }
]
query = "Multi-Head Attention 为什么要使用多个注意力头？"

context, citations = build_context(reranked_results)
prompt = build_prompt(query, context)
answer = generate_answer(prompt)

print("========== Query ==========")
print(query)

print("\n========== Context ==========")
print(context)

print("\n========== Answer ==========")
print(answer)

print("\nAnswer 类型：", type(answer))