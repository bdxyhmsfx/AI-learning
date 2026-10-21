from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from rag_knowledge_base.api.schemas import RAGResponse, RAGRequest
import logging

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    qdrant_client = None
    try:
        print("===== RAG 服务初始化开始 =====")
        from rag_knowledge_base.loaders.document_loader import load_directory
        from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks
        from rag_knowledge_base.retrieval.bm25_retriever import build_bm25_index
        documents = load_directory(BASE_DIR / "data")
        if not documents:
            raise RuntimeError("知识库中没有可加载的文档")
        all_chunks = []
        for document in documents:
            chunks = chunk_document(document)
            all_chunks.extend(chunks)
        all_chunks = deduplicate_chunks(all_chunks)
        if not all_chunks:
            raise RuntimeError("知识库没有有效 Chunk")
        print("文档数量：", len(documents))
        print("Chunk 数量：", len(all_chunks))
        from rag_knowledge_base.vector_store.qdrant_store import client, COLLECTION_NAME
        qdrant_client = client
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            raise RuntimeError(f"Qdrant Collection 不存在：{COLLECTION_NAME}")
        indexed_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print("Qdrant Points 数量：", indexed_count)
        if indexed_count != len(all_chunks):
            raise RuntimeError(
                "Qdrant Points 数量与当前 Chunk 数量不一致，"
                "请先检查并同步索引"
            )
        bm25 = build_bm25_index(all_chunks)
        print("BM25 初始化完成")
        print("正在加载 RAG 模型……")
        from rag_knowledge_base.rag_pipeline import rag_pipeline
        app.state.all_chunks = all_chunks
        app.state.bm25 = bm25
        app.state.rag_pipeline = rag_pipeline
        app.state.ready = True
        print("===== RAG 服务初始化完成 =====")
        yield
    finally:
        app.state.ready = False
        if qdrant_client is not None:
            qdrant_client.close()
            print("Qdrant Client 已关闭")
        print("===== RAG 服务已关闭 =====")


app = FastAPI(title="RAG Knowledge Base API", version="0.1.0", lifespan=lifespan)


@app.get("/hello")
def hello():
    return {"Hello": "World"}


@app.post("/demo-query", response_model=RAGResponse)
def demo_query(request: RAGRequest):
    return {
        "answer": (
            f"模拟响应：已收到问题“{request.query}”。"
            "当前尚未调用 RAG Pipeline。"
        ),
        "citations": {},
        "invalid_ids": [],
        "has_citations": False
    }


@app.get("/init-status")
def init_status(request: Request):
    return {
        "ready": request.app.state.ready,
        "chunk_count": len(request.app.state.all_chunks)
    }


@app.get("/health")
def health(request: Request):
    state = request.app.state
    if not getattr(state, "ready", False):
        raise HTTPException(status_code=503, detail="RAG 服务尚未就绪")
    return {
        "status": "ready",
        "chunk_count": len(state.all_chunks)
    }


@app.post("/query", response_model=RAGResponse)
def query_rag(data: RAGRequest, request: Request):
    state = request.app.state
    if not getattr(state, "ready", False):
        raise HTTPException(status_code=503, detail="RAG 服务尚未就绪")
    try:
        result = state.rag_pipeline(
            query=data.query,
            all_chunks=state.all_chunks,
            bm25=state.bm25
        )
    except Exception as e:
        logger.exception("RAG 问答过程中发生异常")
        raise HTTPException(status_code=500, detail="RAG 查询处理失败，请稍后重试")from e
    return result
