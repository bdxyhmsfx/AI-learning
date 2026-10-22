# RAG Knowledge Base

基于本地知识库的 RAG（Retrieval-Augmented Generation，检索增强生成）学习项目。系统支持加载本地文档，通过混合检索、重排序和本地大语言模型生成回答，并返回文档引用信息；同时提供 FastAPI 接口与自动化测试。

> 项目定位：学习与实验原型，尚未针对生产环境的可靠性、并发和安全性进行完整验证。

## 1. 核心功能

- **文档加载**：支持 PDF、DOCX、TXT、Markdown 等本地资料。
- **文本处理**：基于 Token 进行 Chunk 切分、Overlap 和去重。
- **混合检索**：Qdrant Dense Retrieval + Jieba/BM25，采用 RRF 融合结果。
- **重排序**：使用 BAAI/bge-reranker-base（Cross-Encoder）。
- **答案生成**：本地 Qwen/Qwen3-0.6B；结合检索上下文构建 Prompt。
- **引用检查**：返回引用来源、页码、Chunk ID，以及无效引用编号。
- **HTTP 服务**：FastAPI 提供 `/query`、`/health` 等接口，并使用 Pydantic 校验请求。
- **评估与测试**：包含 Retrieval、Reranker、Generation、Citation、Performance 评估，以及 pytest API 测试。

## 2. 技术栈

| 组件 | 方案 |
| --- | --- |
| 编程语言 | Python 3.10 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| 向量数据库 | 本地 Qdrant（`qdrant-client`） |
| 稀疏检索 | Jieba + BM25Okapi（`rank-bm25`） |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| Reranker | `BAAI/bge-reranker-base` |
| 大语言模型 | `Qwen/Qwen3-0.6B` |
| 接口服务 | FastAPI + Uvicorn + Pydantic |
| API 测试 | requests、Postman、Swagger UI、TestClient、pytest |

## 3. 系统处理流程

```text
本地文档 → 文档解析 → Token Chunking / 去重
                              ↓
                      Qdrant 向量索引

用户 Query
    ├── Dense Retrieval（Qdrant）
    └── Sparse Retrieval（BM25）
              ↓
          RRF Fusion
              ↓
       Cross-Encoder Reranker
              ↓
       Context + Prompt
              ↓
         Qwen3-0.6B
              ↓
         Citation Validation
              ↓
          JSON Response
```

目前正式 `rag_pipeline.py` 使用 Hybrid Top-5 候选，再通过 Reranker 返回 Top-3 上下文结果。

## 4. 主要目录

```text
rag_knowledge_base/
├── api/
│   ├── main.py                 # FastAPI 路由、Lifespan、异常处理
│   ├── schemas.py              # Pydantic 请求及响应模型
│   ├── call_api.py             # requests 真实 HTTP 调用
│   └── test_api.py             # TestClient + Mock + pytest
├── loaders/                    # PDF / DOCX / 文本加载
├── chunking/                   # Token Chunking、去重
├── vector_store/               # 本地 Qdrant、Embedding
├── retrieval/                  # BM25、Hybrid、RRF、Reranker
├── generation/                 # Context、Prompt、LLM、Citation
├── evaluation/                 # 评估脚本与评估结果
│   └── results/
│       └── evaluation_report.md
├── data/                       # 本地知识库资料
├── rag_pipeline.py             # RAG 主链路
├── test_qdrant.py              # 现有的实验性索引重建脚本（有覆盖风险）
├── requirements.txt            # Python 直接依赖
└── README.md
```

## 5. 环境准备

推荐 Windows + Python 3.10。GPU 推理需要适配本机驱动及 CUDA 环境的 PyTorch；开发环境中使用过 CUDA 12.6 构建。CPU 也可尝试，但生成性能可能明显不同。

在**仓库根目录**（包含 `rag_knowledge_base/` 的目录）操作：

```powershell
python -m venv .venv_rag
.\.venv_rag\Scripts\Activate.ps1
python -m pip install -r rag_knowledge_base/requirements.txt
```

若计划使用 NVIDIA GPU，建议先按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 安装适配的 PyTorch，再安装其余依赖；`requirements.txt` 中普通的 `torch` 版本声明不保证自动选择与开发环境相同的 CUDA 构建。

首次加载 Embedding、Reranker、Qwen 模型时，可能需要联网从 Hugging Face 下载权重并写入本地缓存。请预留磁盘空间。

> 开发期间曾复用 `.venv_d2l`；其中 `d2l==1.0.3` 与已安装的 `numpy`、`requests` 版本存在要求冲突。建议新环境只为本项目安装所需依赖，而不是直接复制整个 D2L 环境。

## 6. 数据与向量索引

样例知识库资料位于 `rag_knowledge_base/data/`，包括：`attention.pdf`、`rag.docx`、`transformer.md`、`lora.txt`。此前测试版本加载得到 6 个 Document 和 13 个 Chunk，实际数量会随资料及切分参数变化。

向量数据库存储在 `rag_knowledge_base/qdrant_storage/`，该目录已在 `.gitignore` 中排除，**克隆仓库后不会自动拥有已有索引**。

**首次构建（或明确需要重建）索引：** 当前仓库提供 `rag_knowledge_base/test_qdrant.py`，但它属于实验脚本，运行前必须注意：

1. 脚本目前将资料路径写死为 `E:\daily_code\rag_knowledge_base\data`。若项目不在这个位置，应先改为基于脚本位置的相对路径，例如：

   ```python
from pathlib import Path
from rag_knowledge_base.loaders.document_loader import load_directory

BASE_DIR = Path(__file__).resolve().parent
documents = load_directory(BASE_DIR / "data")
   ```

2. **脚本会删除同名的已有 Qdrant Collection，然后重建。现有索引如果包含额外数据，执行后可能丢失。只应在首次初始化或明确确认要覆盖时运行。**
3. 重建时先停止 Uvicorn 及其他访问同一本地 Qdrant 数据库的 Python 进程，避免本地文件锁冲突。

确认后，从仓库根目录运行：

```powershell
python -m rag_knowledge_base.test_qdrant
```

正常完成后，再启动 API。当前 Lifespan 会检查 Collection 是否存在，并检查 Qdrant Point 数量与 Chunk 数量是否一致；它**不会自动生成缺失的索引**。后续可将实验脚本改造为更安全、可重复执行的独立初始化命令。

## 7. 启动 API

在仓库根目录执行：

```powershell
python -m uvicorn rag_knowledge_base.api.main:app --host 127.0.0.1 --port 8000
```

服务启动时加载本地文档、构建 BM25、检查 Qdrant 索引并加载所需模型。默认本机地址：

- Swagger API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>
- RAG 问答：`POST http://127.0.0.1:8000/query`

本地 Qdrant 模式下建议使用单个 Uvicorn Worker，避免多个进程同时打开同一数据库目录。使用 `Ctrl+C` 正常结束服务，Lifespan 会关闭 Qdrant Client。

## 8. API 使用方式

### `GET /health`

服务就绪时返回 HTTP 200，例如：

```json
{
  "status": "ready",
  "chunk_count": 13
}
```

未就绪时返回 HTTP 503。当前健康检查主要反映初始化状态，并不是每次都执行完整的向量检索或模型推理。

### `POST /query`

请求体示例：

```json
{
  "query": "Causal Mask 的作用是什么？"
}
```

返回字段结构示例（内容仅为格式示意，并非真实推理结果）：

```json
{
  "answer": "模型实际生成的回答。[1]",
  "citations": {
    "1": {
      "source": "attention.pdf",
      "page": 3,
      "chunk_id": "示例ChunkID"
    }
  },
  "invalid_ids": [],
  "has_citations": true
}
```

`query` 为必填字符串，限制长度为 1～2000 字符，且不能全部为空白。主要 HTTP 状态码：

| 状态码 | 说明 |
| --- | --- |
| 200 | 请求成功处理（包括正常的知识库拒答） |
| 422 | 请求数据不符合 Pydantic 校验规则 |
| 503 | RAG 服务未就绪 |
| 500 | RAG 内部处理异常 |

`/hello`、`/demo-query`、`/init-status` 为学习阶段保留的辅助接口，其中 `/demo-query` **不会**执行真实 RAG 推理。

## 9. 测试

**真实服务测试**：先启动 Uvicorn，再在另一个终端运行：

```powershell
python -m rag_knowledge_base.api.call_api
```

也可以使用 Swagger UI 或 Postman 发送请求。

**隔离的 API 自动化测试**（不必先启动 Uvicorn；使用 Mock 模拟 RAG Pipeline）：

```powershell
python -m pytest rag_knowledge_base/api/test_api.py -v
```

当前有六项 API 测试，覆盖健康检查、正常问答、参数校验、未就绪和内部异常等行为。Mock 测试不运行真实 Qwen，因此不能替代真实端到端测试。

## 10. 评估结果与局限

完整实验记录：[RAG 系统综合评估报告](evaluation/results/evaluation_report.md)。该报告基于 2026-10-19 的小规模本地实验：16 道问题（13 道可回答、3 道知识库外）。

部分指标：

- Hybrid Retrieval Top-5 Hit Rate：**1.0000**。
- Reranker 实验（候选 Top-10、最终 Top-3）MRR：**1.0000**；该实验与正式 Pipeline 的候选参数不同。
- Generation Correctness：**1.692 / 2**。
- 知识库外问题拒答成功率：**66.67%（2/3）**。
- 平均单请求耗时：**2594.79 ms**（不包含模型启动和索引构建）。

上述数字仅代表当前小数据集和当时实验设置，不能直接推断大型知识库或并发服务表现。

**已知限制：**

1. **知识库外问题拒答不稳定**：例如 Kubernetes Pod 问题曾出现无证据作答及无关引用。HTTP 200 不代表答案一定正确。
2. **引用支持度不足**：现有引用校验主要检查引用编号有效性，尚不能保证每条事实都得到可靠支持。
3. **索引初始化不够工程化**：当前索引重建使用实验脚本，存在硬编码路径和覆盖旧 Collection 的风险。
4. **单机本地原型**：尚未全面验证并发性能、生产部署、限流、认证及冷启动成本。

## 11. 后续改进方向

- 增加证据充分性判断与更可靠的拒答策略。
- 研究答案事实与引用证据的对应关系，改进 Citation Support。
- 编写安全、幂等的索引初始化/增量更新脚本。
- 引入独立的项目虚拟环境及更完整的依赖锁定与部署方案。
- 扩充评估数据集，开展重复测量和并发测试。
