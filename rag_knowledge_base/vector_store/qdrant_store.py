from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = BASE_DIR / "qdrant_storage"
COLLECTION_NAME = "rag_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
client = QdrantClient(path=str(QDRANT_PATH))
embedding_model = SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings

def create_collection(vector_size):
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}'已存在")
        return
    client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size,distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' 创建成功")

def upsert_chunks(chunks,embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "chunks 和 embeddings 数量必须一致"
        )
    points = []
    for chunk,embedding in zip(chunks,embeddings):
        point = PointStruct(
            id = chunk["chunk_id"],
            vector = embedding.tolist(),
            payload = {
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "page": chunk["page"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"]
            }
        )
        points.append(point)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"成功写入 {len(points)} 个 Points")

def search_chunks(query,k=3,source=None,file_type=None,page=None):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_vector = query_embedding[0].tolist()
    condition=[]
    if source is not None:
        condition.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        )
    if file_type is not None:
        condition.append(
            FieldCondition(
                key="file_type",
                match=MatchValue(value=file_type),
            )
        )
    if page is not None:
        condition.append(
            FieldCondition(
                key="page",
                match=MatchValue(value=page),
            )
        )
    query_filter = None
    if condition:
        query_filter=Filter(
            must=condition
        )
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True
    ).points
    results = []
    for point in search_result:
        result = {
            "chunk_id": point.id,
            "document_id": point.payload["document_id"],
            "score": point.score,
            "text": point.payload["text"],
            "source": point.payload["source"],
            "file_type": point.payload["file_type"],
            "page": point.payload["page"],
            "title": point.payload["title"],
            "chunk_index": point.payload["chunk_index"],
            "token_count": point.payload["token_count"],
            "retriever": "dense"
        }
        results.append(result)
    return results

if __name__ == "__main__":
    print("Qdrant path:", QDRANT_PATH)
    print("Qdrant Client 创建成功")
    client.close()