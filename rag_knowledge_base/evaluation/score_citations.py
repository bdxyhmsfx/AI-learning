import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULT_PATH = EVAL_DIR / "results" / "citation_evaluation.json"
with open(RESULT_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)
support_scores = []
total_citations = 0

for record in records:
    for citation_id, item in record["citations"].items():
        total_citations += 1
        score = item["support"]
        if score is None:
            continue
        if score not in [0, 1, 2]:
            raise ValueError(
                f"{record['id']} 引用 {citation_id} 的评分无效"
            )
        support_scores.append(score)
print("\n===== Citation Support Evaluation =====")
print("引用数量：", total_citations)
print("已评分引用：", len(support_scores))

if support_scores:
    full_support = support_scores.count(2)
    partial_support = support_scores.count(1)
    unsupported = support_scores.count(0)

    full_support_rate = (full_support / len(support_scores))
    average_support = (sum(support_scores) / len(support_scores))

    print("充分支持：", full_support)
    print("部分支持：", partial_support)
    print("不支持：", unsupported)

    print(f"Full Support Rate：{full_support_rate:.4f}")
    print(f"Average Support：{average_support:.4f} / 2")
    if len(support_scores) < total_citations:
        print("注意：还有引用尚未评分，当前仅为阶段性结果。")
else:
    print("尚未进行 Citation Support 人工评分")