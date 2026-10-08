import faiss
from sentence_transformers import SentenceTransformer
from simple_chunk import text_split

documents=[
    "深度学习可以从 PyTorch 开始学习",
    "数据库使用 SQL 查询数据",
    "CNN 常用于图像处理",
    "Transformer 的核心机制之一是 Attention",
    "LoRA 的 r 表示低秩矩阵的秩",
    "RAG 会先检索外部知识，再把知识交给大模型"
]

chunks=[]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
for document in documents:
    documents_chunks = text_split(document,30,10)
    chunks.extend(documents_chunks)
documents_embeddings = model.encode(chunks,convert_to_numpy=True).astype("float32")
dimension = documents_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(documents_embeddings)

def retriever(query,k):
    query_embedding = model.encode([query],convert_to_numpy=True).astype("float32")
    distances,indices = index.search(query_embedding,k)
    results = []
    for i in range(k):
        text_chunk = chunks[indices[0][i]]
        results.append({
            "text":text_chunk,
            "distance":distances[0][i],
            "index":indices[0][i]
        })
    return results

if __name__=="__main__":
    query = "LoRA 中的 r 是什么意思？"
    results = retriever(query,6)
    for result in results:
        print(f"\ndistance: {result['distance']:.4f}")
        print(f"index: {result['index']}")
        print(f"text: {result['text']}")