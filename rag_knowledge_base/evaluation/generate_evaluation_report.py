"""4.2.7.8: 汇总 RAG Evaluation，生成 Markdown 与 JSON 综合报告。

放入 rag_knowledge_base/evaluation/ 目录，运行本文件。
4.2.7.3/4 的结果原先只打印到终端，故在下方手动记录原实验汇总；
重新实验后需要更新对应常量，不能把旧数据伪装成新结果。
"""
import json
import math
import platform
import statistics
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# 下列两组数字来自已经运行过的 4.2.7.3 和 4.2.7.4 控制台汇总。
# 它们对应不同实验设置；不会在本脚本中重新计算或重新检索。
RETRIEVAL = {
    "sample_size": 13,
    "k": 5,
    "results": {
        "Dense":  {"hit_rate": 0.8462, "recall": 0.8462, "precision": 0.1846, "mrr": 0.4936},
        "BM25":   {"hit_rate": 1.0000, "recall": 0.9615, "precision": 0.2154, "mrr": 0.8846},
        "Hybrid": {"hit_rate": 1.0000, "recall": 0.9231, "precision": 0.2000, "mrr": 0.8333},
    },
}

RERANKER = {
    "sample_size": 13,
    "candidate_k": 10,
    "final_k": 3,
    "candidate_recall": 1.0000,
    "results": {
        "Hybrid Top-3": {"hit_rate": 1.0000, "recall": 0.9231, "precision": 0.3333, "mrr": 0.8333},
        "Reranker Top-3": {"hit_rate": 1.0000, "recall": 0.9231, "precision": 0.3333, "mrr": 1.0000},
    },
}


def read_json(filename):
    path = RESULTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"缺少评估文件：{path}。请先完成对应阶段，或确认文件名。")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{filename} 应为非空 JSON 列表")
    return data


def average_scored(records, key):
    values = [r["scores"][key] for r in records]
    if any(v is None for v in values):
        raise ValueError(f"{key} 尚有未评分问题，请先完成人工标注")
    return statistics.mean(values)


def percentile_nearest_rank(values, percentile):
    items = sorted(values)
    index = math.ceil(len(items) * percentile / 100) - 1
    return items[max(0, index)]


def summarize_generation(records):
    answerable = [r for r in records if r["answerable"]]
    out_of_scope = [r for r in records if not r["answerable"]]
    if not answerable or not out_of_scope:
        raise ValueError("Generation 结果必须包含可回答题与知识库外题")
    return {
        "total_questions": len(records),
        "answerable_questions": len(answerable),
        "out_of_scope_questions": len(out_of_scope),
        "relevance": average_scored(answerable, "relevance"),
        "correctness": average_scored(answerable, "correctness"),
        "faithfulness": average_scored(answerable, "faithfulness"),
        "abstention_rate": average_scored(out_of_scope, "abstention"),
        "failed_abstentions": [r["id"] for r in out_of_scope if r["scores"]["abstention"] == 0],
    }


def summarize_citations(records):
    items = [(r["id"], key, value) for r in records for key, value in r["citations"].items()]
    if not items:
        raise ValueError("没有引用可评估，无法计算 Citation Validity 与 Support")
    missing = [(qid, key) for qid, key, item in items if item["support"] not in (0, 1, 2)]
    if missing:
        raise ValueError(f"Citation Support 未完成标注：{missing}")
    answerable = [r for r in records if r["answerable"]]
    if not answerable:
        raise ValueError("没有可回答问题，无法计算引用覆盖率")
    supports = [item["support"] for _, _, item in items]
    return {
        "total_citations": len(items),
        "valid_citations": sum(item["valid"] is True for _, _, item in items),
        "validity_rate": sum(item["valid"] is True for _, _, item in items) / len(items),
        "answer_level_coverage": sum(bool(r["citations"]) for r in answerable) / len(answerable),
        "full_support_rate": supports.count(2) / len(supports),
        "average_support": statistics.mean(supports),
        "support_counts": {str(score): supports.count(score) for score in (0, 1, 2)},
        "missing_citations": [r["id"] for r in answerable if not r["citations"]],
        "unsupported_citations": [f"{qid} [{key}]" for qid, key, item in items if item["support"] == 0],
        "partial_citations": [f"{qid} [{key}]" for qid, key, item in items if item["support"] == 1],
    }


def summarize_performance(records):
    stages = ["retrieval_ms", "reranker_ms", "context_ms", "prompt_ms",
              "generation_ms", "citation_ms", "total_ms"]
    means = {stage: statistics.mean(r["times_ms"][stage] for r in records) for stage in stages}
    latencies = [r["times_ms"]["total_ms"] for r in records]
    peaks = [r["peak_gpu_mb"] for r in records if r.get("peak_gpu_mb") is not None]
    return {
        "sample_size": len(records),
        "question_ids": [r["id"] for r in records],
        "stage_mean_ms": means,
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile_nearest_rank(latencies, 95),
        "generation_time_ratio": means["generation_ms"] / means["total_ms"],
        "mean_peak_gpu_mib": statistics.mean(peaks) if peaks else None,
    }


def metric_row(name, result):
    return f'| {name} | {result["hit_rate"]:.4f} | {result["recall"]:.4f} | {result["precision"]:.4f} | {result["mrr"]:.4f} |'


def make_report(summary):
    ret = summary["retrieval"]
    rer = summary["reranker"]
    gen = summary["generation"]
    cite = summary["citation"]
    perf = summary["performance"]
    ret_rows = "\n".join(metric_row(name, result) for name, result in ret["results"].items())
    rer_rows = "\n".join(metric_row(name, result) for name, result in rer["results"].items())
    gpu_line = (f'{perf["mean_peak_gpu_mib"]:.1f} MiB（PyTorch allocated 峰值平均值）'
                if perf["mean_peak_gpu_mib"] is not None else "未记录 CUDA 显存峰值")
    return f"""# RAG 系统综合评估报告（4.2.7.8）

生成时间：{summary['generated_at']}  
知识库：attention.pdf、rag.docx、transformer.md、lora.txt（当前测试版本）  
模型：all-MiniLM-L6-v2 / BAAI/bge-reranker-base / Qwen3-0.6B  
代码版本：请在正式归档时补充 `git rev-parse HEAD` 的输出。  
运行环境：Python {platform.python_version()}；{platform.platform()}

## 1. 数据与实验设置

- Golden Dataset：{gen['total_questions']} 道（可回答 {gen['answerable_questions']}，知识库外 {gen['out_of_scope_questions']}）。
- Retrieval 实验：{ret['sample_size']} 道可回答题，评估 Top-{ret['k']}；指标来源于 **4.2.7.3 原始控制台记录（手动录入）**。
- Reranker 实验：{rer['sample_size']} 道可回答题，Hybrid 候选 Top-{rer['candidate_k']}，最终 Top-{rer['final_k']}；指标来源于 **4.2.7.4 原始控制台记录（手动录入）**。
- 正式 RAG Pipeline：Hybrid Top-5（candidate_k=5），Reranker Top-3，Qwen3-0.6B。
- 性能实验：{perf['sample_size']} 道查询，预热后单次测试；不包括模型启动和索引构建时间。
- 注意：Reranker 研究实验和正式 Pipeline 的候选参数不同，不把两者解释为同一配置下的结果。

## 2. Retrieval Evaluation（K={ret['k']}）

| 方法 | Hit Rate | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
{ret_rows}

当前小知识库中 BM25 优于其他方案，不能推广为通用结论。

## 3. Reranker Evaluation（候选 K={rer['candidate_k']}，最终 K={rer['final_k']}）

| 方法 | Hit Rate | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
{rer_rows}

Candidate Recall@{rer['candidate_k']} = {rer['candidate_recall']:.4f}。  
MRR 提升 = {rer['results']['Reranker Top-3']['mrr'] - rer['results']['Hybrid Top-3']['mrr']:+.4f}；召回未改善。

## 4. Generation Evaluation

| 指标 | 结果 | 评价对象 |
|---|---:|---|
| Relevance | {gen['relevance']:.3f} / 2 | {gen['answerable_questions']} 道可回答题 |
| Correctness | {gen['correctness']:.3f} / 2 | {gen['answerable_questions']} 道可回答题 |
| Faithfulness | {gen['faithfulness']:.3f} / 2 | {gen['answerable_questions']} 道可回答题 |
| Abstention | {gen['abstention_rate']:.2%} | {gen['out_of_scope_questions']} 道知识库外题 |

拒答失败题号：{', '.join(gen['failed_abstentions']) or '无'}。

## 5. Citation Evaluation

| 指标 | 结果 |
|---|---:|
| Validity | {cite['validity_rate']:.2%}（{cite['valid_citations']}/{cite['total_citations']}） |
| Answer-level Coverage | {cite['answer_level_coverage']:.2%} |
| Full Support Rate | {cite['full_support_rate']:.2%} |
| Average Support | {cite['average_support']:.3f} / 2 |

充分支持 {cite['support_counts']['2']} 个，部分支持 {cite['support_counts']['1']} 个，不支持 {cite['support_counts']['0']} 个。  
可回答但缺少引用：{', '.join(cite['missing_citations']) or '无'}。  
无证据支持的引用：{', '.join(cite['unsupported_citations']) or '无'}。  
证据仅部分支持的引用：{', '.join(cite['partial_citations']) or '无'}。

注：Coverage 为“答案是否有任何引用”的粗粒度指标，不是逐条事实覆盖率。

## 6. Performance Evaluation

| 指标 | 数值 |
|---|---:|
| Mean | {perf['mean_ms']:.2f} ms |
| P50（Median） | {perf['p50_ms']:.2f} ms |
| P95（Nearest-Rank） | {perf['p95_ms']:.2f} ms |
| Retrieval 平均耗时 | {perf['stage_mean_ms']['retrieval_ms']:.2f} ms |
| Reranker 平均耗时 | {perf['stage_mean_ms']['reranker_ms']:.2f} ms |
| Generation 平均耗时 | {perf['stage_mean_ms']['generation_ms']:.2f} ms |
| Generation 耗时占比 | {perf['generation_time_ratio']:.2%} |
| GPU 显存峰值 | {gpu_line} |

小样本 P95 不宜用作正式服务等级指标；模型加载与预处理未计入请求延迟。

## 7. Robustness 结论（基于 4.2.7.7 终端测试记录）

- 非法 Query：空字符串、空白、None、数字被拒绝。
- 空检索与空 Context：返回兜底文本，未继续调用 Qwen。
- 无效引用：[9] 能被发现，但当前 Pipeline 不会自动修复引用。
- 检索服务故障：RuntimeError 向调用方传播，尚无 API 级异常恢复策略。
- 知识库外问题：有 {len(gen['failed_abstentions'])} 道拒答失败，不应将“检索结果非空”视为“证据充分”。

本节的 Robustness 信息来源于此前的控制台运行结果，**不是本报告脚本重新执行的测试**。

## 8. 关键错误案例与优化优先级

1. **高优先级（证据边界）**：q014 Kubernetes Pod 问题中，模型未拒答，还引用 Attention 文本。应研究检索证据充分性判断与严格拒答策略，避免只凭最大相似度直接设置任意阈值。
2. **高优先级（引用可信度）**：q009 引用 [2] 只能部分支持结论；q007、q011 缺少引用。后续研究事实级 Citation Support 与输出校验。
3. **中优先级（答案质量）**：q008 缺少 Prompt 构建环节；q011 错误暗示 lora_alpha 影响可训练参数数量。后续改进 Prompt 和证据组织，再通过同一测试集复测。
4. **中优先级（Chunk 边界）**：部分知识在相邻 Chunk 之间被截断；可比较 Chunk 大小、Overlap 和 Context 组合策略。
5. **后续性能优化**：Generation 是单请求主要延迟来源，可尝试记录输入/输出 Token 数、GPU 推理参数与生成长度，再进行对照实验。

## 9. 局限性与下一轮实验

- 当前知识库只有少量文档和 13 个 Chunk；测试集仅 16 道，且部分问题与资料措辞接近。
- Generation、Citation 使用人工三级评分，存在评分人主观性；测试集不等于独立业务验证集。
- Retrieval、Reranker、端到端 Pipeline 的参数不完全一致；跨实验只能谨慎比较。
- 性能测试当前为 {perf['sample_size']} 个预热后单次请求，没有并发、重复测量或冷启动统计。
- 下一轮可以在固定数据集和参数配置的前提下，一次只改变一个变量；保留此次结果作为 Baseline，并扩充测试问题。

**结论：** 当前系统已具备完整的检索、重排序、生成和评估链路；进一步改进应优先关注证据不足时的拒答和引用支持情况，其次是答案正确性与生成延迟。
"""


def main():
    generation_records = read_json("generation_results_scored.json")
    citation_records = read_json("citation_evaluation.json")
    performance_records = read_json("performance_results.json")

    generation_ids = {r["id"] for r in generation_records}
    citation_ids = {r["id"] for r in citation_records}
    if len(generation_ids) != len(generation_records) or len(citation_ids) != len(citation_records):
        raise ValueError("评估记录存在重复题号")
    if generation_ids != citation_ids:
        raise ValueError("Generation 与 Citation 题号不一致，请检查是否来自同一批评估")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval": RETRIEVAL,
        "reranker": RERANKER,
        "generation": summarize_generation(generation_records),
        "citation": summarize_citations(citation_records),
        "performance": summarize_performance(performance_records),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "evaluation_report.md").write_text(
        make_report(summary), encoding="utf-8"
    )
    print("综合报告已生成：")
    print(RESULTS_DIR / "evaluation_summary.json")
    print(RESULTS_DIR / "evaluation_report.md")
    print(f"Generation Correctness：{summary['generation']['correctness']:.3f} / 2")
    print(f"Citation Full Support：{summary['citation']['full_support_rate']:.2%}")
    print(f"Mean Latency：{summary['performance']['mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
