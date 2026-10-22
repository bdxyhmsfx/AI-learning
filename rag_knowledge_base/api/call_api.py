import requests

BASE_URL = "http://127.0.0.1:8000"
health_response = requests.get(f"{BASE_URL}/health", timeout=10)
assert health_response.status_code == 200

health_data = health_response.json()
assert health_data["status"] == "ready"
assert health_data["chunk_count"] > 0
print("健康检查通过：", health_data)

# print("Health Status：", health_response.status_code)
# print("Health Response：", health_response.json())

payload = {"query": "Multi-Head Attention 为什么使用多个注意力头？"}
query_response = requests.post(f"{BASE_URL}/query", json=payload, timeout=120)
assert query_response.status_code == 200
result = query_response.json()

required_fields = {
    "answer",
    "citations",
    "invalid_ids",
    "has_citations"
}

assert required_fields.issubset(result.keys())
assert isinstance(result["answer"], str)
assert result["answer"].strip()
print("\nRAG 问答接口通过")
print("Answer：", result["answer"])
print("Citations：", result["citations"])

invalid_response = requests.post(
    f"{BASE_URL}/query",
    json={"query": "   "},
    timeout=10
)

assert invalid_response.status_code == 422
print("\n参数校验通过")
print("\n===== E2E 基础验收通过 =====")

# print("\nQuery Status：", query_response.status_code)
# print("Query Response：", query_response.json())