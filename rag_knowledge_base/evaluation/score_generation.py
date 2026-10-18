import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULT_PATH = EVAL_DIR / "results" / "generation_results_scored.json"

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)
answerable_records = [record for record in records if record["answerable"]]
unanswerable_records = [record for record in records if not record["answerable"]]

def calculate_average(records, key):
    values = []
    for record in records:
        score = record["scores"][key]
        if score is not None:
            values.append(score)
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)

print("\n===== Generation Evaluation =====")
for key in ["relevance", "correctness", "faithfulness"]:
    average, count = calculate_average(answerable_records, key)
    if average is None:
        print(f"{key}：尚未评分")
    else:
        print(f"{key}：{average:.3f} / 2（已评分 {count} 题）")
print("\n===== Abstention Evaluation =====")
average, count = calculate_average(unanswerable_records,"abstention")
if average is None:
    print("Abstention：尚未评分")
else:
    print(f"Abstention Success Rate：{average:.3f}")
    print(f"已评分：{count} / {len(unanswerable_records)}")
print("\n可回答问题：", len(answerable_records))
print("知识库外问题：", len(unanswerable_records))