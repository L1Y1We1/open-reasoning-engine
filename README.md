# Open Reasoning Engine

一套完全开源、可本地自建的 Reasoning Engine：

```text
FastAPI → LangGraph 推理工作流 → LlamaIndex RAG → Qdrant / Milvus
                           ↘ Ollama（Qwen / Llama）
```

默认组合是 **LangGraph + LlamaIndex + Qdrant + Ollama/Qwen**。文档、向量与模型推理都留在
自己的机器或服务器上，不依赖闭源模型 API。

## 它不只是普通 RAG

每次查询依次经过：

1. 分析问题并生成独立检索问题；
2. 从向量库检索证据；
3. 严格基于证据生成带 `[1]` 引用的答案；
4. 校验答案是否被证据支持；
5. 若校验失败，改写检索问题并重试一次；
6. 返回答案、来源、相关度与可观察的步骤摘要。

## 最快启动

要求：Docker Desktop / Docker Engine，建议至少 8 GB 内存。默认 7B 模型在仅 CPU 环境也能运行，
但速度较慢。

```bash
cp .env.example .env
docker compose up -d --build
```

第一次启动会下载 Qwen 模型和中文嵌入模型，需要几分钟。服务就绪后打开：

- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>
- Qdrant 控制台：<http://localhost:6333/dashboard>

Windows PowerShell 可用：

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

## 导入文档

支持 `.txt`、`.md`、`.pdf`、`.docx`、`.csv`，单文件默认上限 25 MB。

```bash
curl -X POST http://localhost:8000/v1/documents \
  -F "files=@data/example.md"
```

也可以在本机 Python 环境用命令行批量导入目录：

```bash
pip install -e .
reasoning-engine ingest ./data
```

## 发起推理查询

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"这个系统的推理流程是什么？"}'
```

响应示例：

```json
{
  "answer": "系统先分析问题并检索证据……[1]",
  "rewritten_question": "Reasoning Engine 的完整推理工作流",
  "citations": [
    {
      "index": 1,
      "source": "example.md",
      "text": "本项目的推理工作流……",
      "score": 0.82,
      "metadata": {}
    }
  ],
  "reasoning_steps": [
    "分析问题并生成检索计划",
    "从向量知识库检索证据",
    "基于证据生成带引用答案",
    "校验答案与证据的一致性",
    "输出最终答案"
  ],
  "retries": 0
}
```

命令行查询：

```bash
reasoning-engine ask "这个系统默认使用什么向量数据库？"
```

## 换模型

编辑 `.env`：

```dotenv
OLLAMA_MODEL=qwen3:8b
```

然后拉取并重启：

```bash
docker compose run --rm model-pull
docker compose restart api
```

也可换成 Ollama 中的 Llama、DeepSeek-R1 蒸馏版等模型。模型名必须与本地 Ollama 标签一致。

## 切换 Milvus

Milvus 更适合数据量大、需要横向扩展的环境。覆盖文件会自动为 API 镜像安装 Milvus 可选依赖：

```bash
pip install -e ".[milvus]"
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d
```

日常单机使用推荐保持默认 Qdrant。

## 本地开发

Python 3.11–3.13：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
pytest
ruff check .
uvicorn reasoning_engine.api:app --reload
```

本地运行 API 时，把 `.env` 中的服务地址改成：

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
```

## 主要配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | 生成与审校模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地嵌入模型 |
| `VECTOR_STORE` | `qdrant` | `qdrant` 或 `milvus` |
| `SIMILARITY_TOP_K` | `5` | 每轮证据数量 |
| `MAX_RETRIES` | `1` | 审校失败后的最大重试数 |
| `CHUNK_SIZE` | `700` | 文档分块大小 |
| `CHUNK_OVERLAP` | `100` | 相邻分块重叠 |

## 生产部署提醒

- 反向代理 API，并增加身份认证、TLS、速率限制与请求体限制。
- 不要把 Qdrant、Milvus 或 Ollama 端口公开到互联网；Compose 默认仅绑定本机。
- 为向量库卷做快照和异地备份，固定经过验证的镜像版本。
- 监控模型延迟、无证据回答比例、检索命中率和纠偏次数。
- 修改嵌入模型或维度时使用新 collection 并重新导入，避免向量维度冲突。

## 项目结构

```text
src/reasoning_engine/
├── api.py          # FastAPI 接口
├── engine.py       # LangGraph 推理与纠偏工作流
├── retrieval.py    # LlamaIndex + Qdrant/Milvus
├── llm.py          # Ollama 模型适配器
├── config.py       # 环境配置
└── cli.py          # 导入与查询命令行
```

## License

MIT
