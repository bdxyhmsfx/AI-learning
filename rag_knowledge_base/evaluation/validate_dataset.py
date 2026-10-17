import json
from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory
from rag_knowledge_base.chunking.chunker import chunk_document, deduplicate_chunks

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "eval_cases.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    eval_cases = json.load(f)
documents = load_directory(BASE_DIR / "data")
all_chunks = []
for document in documents:
    all_chunks.extend(chunk_document(document))
all_chunks = deduplicate_chunks(all_chunks)
valid_chunk_ids = {chunk["chunk_id"] for chunk in all_chunks}
seen_ids = set()
errors = []
for case in eval_cases:
    case_id = case["id"]
    gold_ids = case["gold_chunk_ids"]
    answerable = case["answerable"]
    if case_id in seen_ids:
        errors.append(f"{case_id}：题目 ID 重复")
    seen_ids.add(case_id)
    for chunk_id in gold_ids:
        if chunk_id not in valid_chunk_ids:
            errors.append(f"{case_id}：Chunk ID 不存在：{chunk_id}")
    if answerable and not gold_ids:
        errors.append(f"{case_id}：可回答问题没有正确 Chunk")
    if not answerable and gold_ids:
        errors.append(f"{case_id}：不可回答问题却标注了 Chunk")
print("评估问题数量：", len(eval_cases))
print("知识库 Chunk 数量：", len(all_chunks))

if errors:
    print("\n发现以下问题：")
    for error in errors:
        print(error)
else:
    print("\n基础结构检查通过")