import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "深度学习可以从 PyTorch 开始学习",
    "数据库使用 SQL 查询数据",
    "CNN 常用于图像处理"
]
document_embeddings = model.encode(documents,convert_to_numpy=True)
document_embeddings = document_embeddings.astype("float32")
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
query = ["怎么学习深度学习？"]
query_embedding = model.encode(query,convert_to_numpy=True)
query_embedding = query_embedding.astype("float32")
k=2
distances, indices = index.search(query_embedding,k)
for i in range(k):
    doc_index = indices[0][i]
    print(documents[doc_index])
'''import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "深度学习可以从 PyTorch 开始学习",
    "数据库使用 SQL 查询数据",
    "CNN 常用于图像处理"
]

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

document_embeddings = document_embeddings.astype(
    "float32"
)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(
    dimension
)

index.add(
    document_embeddings
)

query = "怎么学习深度学习？"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

print("\ndistances:")
print(distances)

print("\nindices:")
print(indices)

print("\nTop-K documents:")

for i in range(k):

    doc_index = indices[0][i]

    print(
        documents[doc_index]
    )'''