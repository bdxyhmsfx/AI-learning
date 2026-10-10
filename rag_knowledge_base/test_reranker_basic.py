import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker_model = CrossEncoder(MODEL_NAME,device=device,max_length=512)
print("运行设备：", device)
query = "Multi-Head Attention 有什么作用？"
texts = [
    "Multi-Head Attention 通过多个注意力头学习不同表示子空间的信息。",

    "Transformer 使用 Attention 和前馈网络处理输入序列。",

    "LoRA 使用低秩矩阵减少大模型微调所需的参数量。"
]
pairs = [(query, text)for text in texts]
scores = reranker_model.predict(pairs)

print("\nscores:", scores)
print("scores.shape:", scores.shape)

results = sorted(zip(texts, scores),key=lambda x: float(x[1]),reverse=True)

for i, (text, score) in enumerate(results, start=1):
    print("\n排名：", i)
    print("Score：", float(score))
    print("Text：", text)