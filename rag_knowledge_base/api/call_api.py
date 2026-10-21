import requests

BASE_URL = "http://127.0.0.1:8000"
health_response = requests.get(f"{BASE_URL}/health",timeout=10)
print("Health Status：", health_response.status_code)
print("Health Response：", health_response.json())

payload = {"query":"Causal Mask 的作用是什么？"}
query_response = requests.post(f"{BASE_URL}/query",json=payload,timeout=60)

print("\nQuery Status：", query_response.status_code)
print("Query Response：", query_response.json())