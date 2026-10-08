import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

def cosine_similarity(query, documents):
    dot_products = query @ documents.T

    query_norm = torch.norm(query)
    document_norms = torch.norm(
        documents,
        dim=1
    )
    similarities = dot_products / (
        query_norm * document_norms
    )
    return similarities

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

query_text = "怎么学习深度学习？"

documents_text = [
    "深度学习可以从 PyTorch 开始学习",
    "数据库使用 SQL 查询数据",
    "CNN 常用于图像处理"
]

query = model.encode(
    [query_text],
    convert_to_tensor=True
)

documents = model.encode(
    documents_text,
    convert_to_tensor=True
)

cosine_similarities = cosine_similarity(
    query,
    documents
)

print("\nCosine Similarity:")
print(cosine_similarities)

scores = cosine_similarities.squeeze(0)

sorted_scores, sorted_indices = torch.sort(
    scores,
    descending=True
)

print("\nCosine 排序结果:")

for score, index in zip(
    sorted_scores,
    sorted_indices
):
    print(
        f"{score.item():.4f}",
        documents_text[index.item()]
    )


# =========================
# 6. Top-K
# =========================

k = 2

top_scores, top_indices = torch.topk(
    scores,
    k=k
)


print(f"\nTop-{k} 文档:")

for score, index in zip(top_scores,top_indices):
    print(f"{score.item():.4f}",documents_text[index.item()])
model.similarity_fn_name = (
    SimilarityFunction.MANHATTAN
)

manhattan_similarity = model.similarity(
    query,
    documents
)

print("\nManhattan Similarity:")
print(manhattan_similarity)

manhattan_scores = (
    manhattan_similarity.squeeze(0)
)

sorted_manhattan_scores, \
sorted_manhattan_indices = torch.sort(
    manhattan_scores,
    descending=True
)

print("\nManhattan 排序结果:")

for score, index in zip(sorted_manhattan_scores,sorted_manhattan_indices):
    print(f"{score.item():.4f}",documents_text[index.item()])