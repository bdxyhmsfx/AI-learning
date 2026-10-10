'''
只是手动构造了hybrid_results,并没有和前面的联合起来，只是检验了一下rerank_results的可行性
'''
from rag_knowledge_base.retrieval.reranker import (rerank_results)
query = "Multi-Head Attention 有什么作用？"
hybrid_results = [
    {
        "chunk_id": "chunk_lora",
        "text": "LoRA 使用低秩矩阵减少大模型微调所需的参数量。",
        "source": "lora.txt",
        "page": None,
        "score": 0.032,
        "retriever": "hybrid"
    },

    {
        "chunk_id": "chunk_transformer",
        "text": "Transformer 使用 Attention 和前馈网络处理输入序列。",
        "source": "transformer.md",
        "page": None,
        "score": 0.031,
        "retriever": "hybrid"
    },

    {
        "chunk_id": "chunk_attention",
        "text": "Multi-Head Attention 通过多个注意力头学习不同表示子空间的信息。",
        "source": "attention.pdf",
        "page": 3,
        "score": 0.030,
        "retriever": "hybrid"
    }
]


reranked_results = rerank_results(
    query,
    hybrid_results,
    k=3
)
print("====== Reranker 排名 ======")

for i, result in enumerate(reranked_results, start=1):

    print("\n排名：", i)
    print("Chunk ID：", result["chunk_id"])
    print("原始 RRF Score：", result["score"])
    print("Rerank Score：", result["rerank_score"])
    print("Source：", result["source"])
    print("Text：", result["text"])