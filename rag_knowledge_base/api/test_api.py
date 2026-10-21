from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from rag_knowledge_base.api.main import app

client = TestClient(app, raise_server_exceptions=False)

def test_health_ready():
    with patch.object(app.state, "ready", True, create=True), \
            patch.object(app.state, "all_chunks", [1, 2, 3], create=True):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["chunk_count"] == 3

def test_health_not_ready():
    with patch.object(app.state,"ready",False,create=True):
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "RAG 服务尚未就绪"

def test_query_success():
    fake_result = {
        "answer": "多头注意力可以学习不同的关系。[1]",
        "citations": {
            1: {
                "source": "attention.pdf",
                "page": 3,
                "chunk_id": "test_chunk"
            }
        },
        "invalid_ids": [],
        "has_citations": True
    }
    fake_pipeline = Mock(return_value=fake_result)
    fake_chunks = [{"chunk_id":"test_chunk"}]
    fake_bm25 = object()
    with patch.object(app.state,"ready",True,create=True),\
        patch.object(app.state,"all_chunks",fake_chunks,create=True),\
        patch.object(app.state,"rag_pipeline",fake_pipeline,create=True),\
        patch.object(app.state,"bm25",fake_bm25,create=True):
        response = client.post("/query",json={"query": "  Multi-Head Attention 是什么？  "})
        assert response.status_code == 200
        data = response.json()

        assert data["answer"]==fake_result["answer"]
        assert data["has_citations"] is True
        assert data["citations"]["1"]["source"] == "attention.pdf"
        fake_pipeline.assert_called_once_with(
            query="Multi-Head Attention 是什么？",
            all_chunks=fake_chunks,
            bm25=fake_bm25
        )

def test_query_invalid():
    fake_pipeline = Mock()
    with patch.object(app.state,"ready",True,create=True),\
        patch.object(app.state,"rag_pipeline",fake_pipeline,create=True):
        invalid_payloads = [
            {"query": ""},
            {"query": "   "},
            {},
            {"query": 123}
        ]
        for payload in invalid_payloads:
            response = client.post("/query",json=payload)
            assert response.status_code == 422
        fake_pipeline.assert_not_called()

def test_query_not_ready():
    with patch.object(app.state, "ready", False, create=True):
        response = client.post(
            "/query",
            json={"query": "什么是 RAG？"}
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "RAG 服务尚未就绪"


def test_query_failure():
    fake_pipeline = Mock(side_effect=RuntimeError("模拟 Qdrant 查询失败"))
    with patch.object(app.state, "ready", True, create=True), \
         patch.object(app.state, "all_chunks", [], create=True), \
         patch.object(app.state, "bm25", None, create=True), \
         patch.object(app.state, "rag_pipeline", fake_pipeline, create=True):

        response = client.post("/query",json={"query": "什么是 RAG？"})
        assert response.status_code == 500
        assert response.json()["detail"] == ("RAG 查询处理失败，请稍后重试")
        assert "模拟 Qdrant 查询失败" not in response.text