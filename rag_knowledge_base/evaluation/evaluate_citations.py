import json
import re
from pathlib import Path
from rag_knowledge_base.generation.citation import extract_citation_ids
EVAL_DIR = Path(__file__).resolve().parent

INPUT_PATH = EVAL_DIR / "results" / "generation_results_scored.json"
OUTPUT_PATH = EVAL_DIR / "results" / "citation_evaluation.json"

def extract_context_blocks(context):
    blocks = {}
    parts = re.split(
        r"(?=^\[\d+\] 来源:)",
        context,
        flags=re.MULTILINE
    )
    for part in parts:
        match = re.match(r"^\[(\d+)\] 来源:", part)
        if match:
            citation_id = match.group(1)
            blocks[citation_id] = part.strip()
    return blocks

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    generation_records = json.load(f)
citation_records = []

for record in generation_records:
    answer = record["answer"]
    used_ids = extract_citation_ids(answer)
    available = record["available_citations"]
    context_blocks = extract_context_blocks(record["context"])
    citation_items = {}
    for citation_id in used_ids:
        key = str(citation_id)
        is_valid = key in available
        citation_items[key] = {
            "valid": is_valid,
            "source": available.get(key, {}).get("source"),
            "evidence_text": context_blocks.get(key, ""),
            "support": None,
            "notes": ""
        }
    citation_records.append({
        "id": record["id"],
        "question": record["question"],
        "answerable": record["answerable"],
        "answer": answer,
        "citations": citation_items
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(citation_records, f, ensure_ascii=False, indent=2)

answerable_records = [
    record for record in citation_records
    if record["answerable"]
]

total_citations = sum(
    len(record["citations"])
    for record in citation_records
)

valid_citations = sum(
    item["valid"]
    for record in citation_records
    for item in record["citations"].values()
)

answers_with_citations = sum(
    bool(record["citations"])
    for record in answerable_records
)

validity_rate = (
    valid_citations / total_citations
    if total_citations else None
)

coverage_rate = (
    answers_with_citations / len(answerable_records)
    if answerable_records else None
)


print("\n===== Citation Evaluation =====")
print("问题数量：", len(citation_records))
print("实际使用引用数量：", total_citations)
print("有效引用数量：", valid_citations)

if validity_rate is not None:
    print(f"Citation Validity：{validity_rate:.4f}")
else:
    print("Citation Validity：无引用，无法计算")

if coverage_rate is not None:
    print(f"Citation Coverage：{coverage_rate:.4f}")
else:
    print("Citation Coverage：无可回答问题，无法计算")

print("已保存：", OUTPUT_PATH)