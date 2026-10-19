# RAG 系统综合评估报告（4.2.7.8）

生成时间：2026-10-19T15:25:45+08:00  
知识库：attention.pdf、rag.docx、transformer.md、lora.txt（当前测试版本）  
模型：all-MiniLM-L6-v2 / BAAI/bge-reranker-base / Qwen3-0.6B  
代码版本：8280d0b96c2167b299633938bff846243e99fd91
运行环境：Python 3.10.11；Windows-10-10.0.26200-SP0

## 1. 数据与实验设置

- Golden Dataset：16 道（可回答 13，知识库外 3）。
- Retrieval 实验：13 道可回答题，评估 Top-5；指标来源于 **4.2.7.3 原始控制台记录（手动录入）**。
- Reranker 实验：13 道可回答题，Hybrid 候选 Top-10，最终 Top-3；指标来源于 **4.2.7.4 原始控制台记录（手动录入）**。
- 正式 RAG Pipeline：Hybrid Top-5（candidate_k=5），Reranker Top-3，Qwen3-0.6B。
- 性能实验：16 道查询，预热后单次测试；不包括模型启动和索引构建时间。
- 注意：Reranker 研究实验和正式 Pipeline 的候选参数不同，不把两者解释为同一配置下的结果。

## 2. Retrieval Evaluation（K=5）

| 方法 | Hit Rate | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
| Dense | 0.8462 | 0.8462 | 0.1846 | 0.4936 |
| BM25 | 1.0000 | 0.9615 | 0.2154 | 0.8846 |
| Hybrid | 1.0000 | 0.9231 | 0.2000 | 0.8333 |

当前小知识库中 BM25 优于其他方案，不能推广为通用结论。

## 3. Reranker Evaluation（候选 K=10，最终 K=3）

| 方法 | Hit Rate | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
| Hybrid Top-3 | 1.0000 | 0.9231 | 0.3333 | 0.8333 |
| Reranker Top-3 | 1.0000 | 0.9231 | 0.3333 | 1.0000 |

Candidate Recall@10 = 1.0000。  
MRR 提升 = +0.1667；召回未改善。

## 4. Generation Evaluation

| 指标 | 结果 | 评价对象 |
|---|---:|---|
| Relevance | 2.000 / 2 | 13 道可回答题 |
| Correctness | 1.692 / 2 | 13 道可回答题 |
| Faithfulness | 1.846 / 2 | 13 道可回答题 |
| Abstention | 66.67% | 3 道知识库外题 |

拒答失败题号：q014。

## 5. Citation Evaluation

| 指标 | 结果 |
|---|---:|
| Validity | 100.00%（12/12） |
| Answer-level Coverage | 84.62% |
| Full Support Rate | 83.33% |
| Average Support | 1.750 / 2 |

充分支持 10 个，部分支持 1 个，不支持 1 个。  
可回答但缺少引用：q007, q011。  
无证据支持的引用：q014 [1]。  
证据仅部分支持的引用：q009 [2]。

注：Coverage 为“答案是否有任何引用”的粗粒度指标，不是逐条事实覆盖率。

## 6. Performance Evaluation

| 指标 | 数值 |
|---|---:|
| Mean | 2594.79 ms |
| P50（Median） | 2469.68 ms |
| P95（Nearest-Rank） | 5760.76 ms |
| Retrieval 平均耗时 | 6.88 ms |
| Reranker 平均耗时 | 31.13 ms |
| Generation 平均耗时 | 2556.68 ms |
| Generation 耗时占比 | 98.53% |
| GPU 显存峰值 | 2375.9 MiB（PyTorch allocated 峰值平均值） |

小样本 P95 不宜用作正式服务等级指标；模型加载与预处理未计入请求延迟。

## 7. Robustness 结论（基于 4.2.7.7 终端测试记录）

- 非法 Query：空字符串、空白、None、数字被拒绝。
- 空检索与空 Context：返回兜底文本，未继续调用 Qwen。
- 无效引用：[9] 能被发现，但当前 Pipeline 不会自动修复引用。
- 检索服务故障：RuntimeError 向调用方传播，尚无 API 级异常恢复策略。
- 知识库外问题：有 1 道拒答失败，不应将“检索结果非空”视为“证据充分”。

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
- 性能测试当前为 16 个预热后单次请求，没有并发、重复测量或冷启动统计。
- 下一轮可以在固定数据集和参数配置的前提下，一次只改变一个变量；保留此次结果作为 Baseline，并扩充测试问题。

**结论：** 当前系统已具备完整的检索、重排序、生成和评估链路；进一步改进应优先关注证据不足时的拒答和引用支持情况，其次是答案正确性与生成延迟。
