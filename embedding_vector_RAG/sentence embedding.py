from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "我喜欢学习人工智能",
    "我正在学习机器学习",
    "今天晚上吃什么"
]

# 1. 文本 → Embedding
embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings)

# 2. Embedding → 相似度
similarities = model.similarity(
    embeddings,
    embeddings
)

print(similarities)