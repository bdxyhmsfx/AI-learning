from fastapi import FastAPI
from rag_knowledge_base.api.schemas import RAGResponse,RAGRequest,CitationInfo

app=FastAPI(title="RAG Knowledge Base API",version="0.1.0")
@app.get("/hello")
def hello():
    return {"Hello": "World"}

@app.post("/demo-query",response_model=RAGResponse)
def demo_query(request:RAGRequest):
    return {
        "answer": (
            f"模拟响应：已收到问题“{request.query}”。"
            "当前尚未调用 RAG Pipeline。"
        ),
        "citations": {},
        "invalid_ids": [],
        "has_citations": False
    }