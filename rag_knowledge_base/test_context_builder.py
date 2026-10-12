from rag_knowledge_base.generation.context_builder import (build_context)

reranked_results = [
    {
        "chunk_id": "abc123",
        "text": "Multi-Head Attention 通过多个注意力头学习不同类型的信息。",
        "source": "attention.pdf",
        "page": 3,
        "score": 0.032,
        "rerank_score": 0.997
    },

    {
        "chunk_id": "def456",
        "text": "Transformer 使用 Self-Attention 机制处理序列。",
        "source": "transformer.md",
        "page": None,
        "score": 0.031,
        "rerank_score": 0.911
    },

    {
        "chunk_id": "abc123",
        "text": "Multi-Head Attention 通过多个注意力头学习不同类型的信息。",
        "source": "attention.pdf",
        "page": 3,
        "score": 0.030,
        "rerank_score": 0.900
    }
]

context, citations = build_context(reranked_results, max_chars=4000)
print("========== Context ==========")
print(context)

print("\n========== Citations ==========")
print(citations)

print("\nContext 类型：", type(context))
print("Citations 类型：", type(citations))
