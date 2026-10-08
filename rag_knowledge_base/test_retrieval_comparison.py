from pathlib import Path

from rag_knowledge_base.loaders.document_loader import load_directory

from rag_knowledge_base.chunking.chunker import (
    chunk_document,
    deduplicate_chunks
)

from rag_knowledge_base.retrieval.bm25_retriever import (
    build_bm25_index,
    bm25_search
)

from rag_knowledge_base.retrieval.hybrid_retriever import (
    hybrid_search
)

from rag_knowledge_base.vector_store.qdrant_store import (
    search_chunks,
    client
)


# 1. 加载文档
BASE_DIR = Path(__file__).resolve().parent

documents = load_directory(
    BASE_DIR / "data"
)

# 2. 生成 Chunk
all_chunks = []

for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)

all_chunks = deduplicate_chunks(all_chunks)

# 3. 建立 BM25 Index
bm25 = build_bm25_index(all_chunks)


# 4. 打印检索结果
def print_results(name, results):

    print(f"\n===== {name} =====")

    for i, result in enumerate(results, start=1):

        print(f"\n排名：{i}")
        print("score:", round(result["score"], 4))
        print("source:", result["source"])
        print("page:", result["page"])
        print("chunk_id:", result["chunk_id"])

        # 打印前 150 个字符，方便比较
        print(
            "text:",
            result["text"][:150].replace("\n", " ")
        )


# 5. 准备 Query
queries = [
    "为什么 Transformer 要使用多个注意力头？",
    "Scaled Dot-Product Attention",
    "RAG 和 Fine-Tuning 有什么区别？",
    "Causal Mask 的作用是什么？"
]


try:
    # 6. 对每个 Query 进行三种检索
    for query in queries:

        print("\n" + "#" * 70)
        print("Query:", query)

        # Dense Retrieval
        dense_results = search_chunks(
            query,
            k=3
        )

        # BM25 Retrieval
        bm25_results = bm25_search(
            query,
            all_chunks,
            bm25,
            k=3
        )

        # Hybrid Retrieval
        hybrid_results = hybrid_search(
            query,
            all_chunks,
            bm25,
            k=3,
            candidate_k=5
        )

        # 7. 输出对比结果
        print_results("Dense", dense_results)
        print_results("BM25", bm25_results)
        print_results("Hybrid", hybrid_results)

finally:
    client.close()