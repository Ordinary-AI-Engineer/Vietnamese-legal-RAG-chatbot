# ⚖️ LegalBot AI — Vietnamese Legal Consultation System

> A production-grade RAG chatbot for Vietnamese law, featuring Hybrid Retrieval (BM25 + Vector Search), Reciprocal Rank Fusion, Cross-encoder Reranking, Citation Enforcement, and a CI-gated Evaluation Pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-1.9-red)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:3b-orange)
![MRR@10](https://img.shields.io/badge/MRR@10-0.81-brightgreen)
![Recall@5](https://img.shields.io/badge/Recall@5-87%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![LegalBot AI Demo](docs/screenshots/demo.png)

---

## Architecture

```
Browser (index.html)
      │
      ▼
FastAPI  (:8000)
      │
      ├─ GET  /          →  Web UI (dark-mode chat interface)
      ├─ POST /chat       →  RAG Pipeline
      ├─ GET  /history    →  Chat history
      └─ GET  /health     →  System status + chatbot readiness
            │
            ▼
      LawChatbot
      ├── LawRetriever  ─────────────────────────────────────────────┐
      │   ├── Targeted Search  (article + law detection, top priority)│
      │   ├── Vector Search    (multilingual-e5-large → Qdrant)      │
      │   ├── BM25 Search      (rank_bm25 over full corpus)          │
      │   └── RRF Fusion       (Reciprocal Rank Fusion, 3 streams)   │
      │                                                               │
      ├── CrossEncoderReranker  (optional, disabled by default)       │
      │                                                               │
      └── Ollama LLM (qwen2.5:3b) ← citation-enforced prompt         │
                                                                      │
                                        Qdrant Vector DB (:6333) ─────┘
```

---

## Evaluation Results

| Metric | Baseline (RRF only) | With Cross-encoder Reranker |
|--------|--------------------|-----------------------------|
| MRR@10 | **0.8083** ✅ | 0.6667 |
| Recall@5 | **86.7%** (13/15) ✅ | 73.3% (11/15) |
| Precision@5 | **0.4533** ✅ | 0.3733 |

> **Finding:** `BAAI/bge-reranker-base` is primarily trained on English/Chinese data and degrades retrieval quality on Vietnamese legal text. Pure RRF outperforms reranking for this domain. The reranker is disabled by default and can be re-enabled once a suitable Vietnamese cross-encoder becomes available.

---

## Project Structure

```
ChatBotLawFinal/
├── app/
│   ├── main.py                  # FastAPI server (eager init, thread-safe lock)
│   └── static/
│       └── index.html           # Web UI — pure HTML/CSS/JS, no framework
├── src/
│   ├── retrival/
│   │   ├── retriever.py         # LawRetriever — 3-stream Hybrid RRF
│   │   └── reranker.py          # CrossEncoderReranker (optional)
│   ├── llm/
│   │   └── llm_client.py        # LawChatbot — RAG chain orchestration
│   ├── prompts/
│   │   └── prompt_templates.py  # System prompt with citation enforcement
│   ├── chunking/
│   │   └── chunker.py           # PDF → article-level chunks with metadata
│   ├── ingestion/
│   │   └── loader.py            # PDF document loader
│   └── vectordb/
│       └── vector_store.py      # Qdrant client wrapper
├── evaluation/
│   ├── eval_dataset.json        # 15 ground-truth Q&A pairs
│   ├── retriever_eval.py        # MRR@10, Recall@5, Precision@5 metrics
│   └── run_eval.py              # CI gate — exits with code 1 on failure
├── data/
│   └── raw/                     # Vietnamese legal PDF documents
├── reindex.py                   # Full corpus re-ingestion to Qdrant
├── docker-compose.yml           # Qdrant + API services
├── Dockerfile
├── config.yaml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally: `ollama serve`
- Docker (for Qdrant)

### 1. Install dependencies

```bash
git clone <repo-url>
cd ChatBotLawFinal

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull qwen2.5:3b
```

### 3. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 4. Index the legal corpus

```bash
# Place PDF files in data/raw/ then run:
python reindex.py
```

### 5. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000**

> **Note:** On first startup the server takes ~20 seconds to initialize the embedding model and BM25 index. The `/health` endpoint reports `chatbot_ready: true` when ready.

---

## Production Deployment (Docker)

```bash
# Start both Qdrant and the API:
docker-compose up -d

# Verify:
curl http://localhost:8000/health
```

> Ollama must be running on the host machine. The API container connects via `host.docker.internal`.

---

## API Reference

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Article 198 of the Penal Code regulate?"}'
```

**Response:**
```json
{
  "answer": "Article 198 of the Penal Code regulates the offense of deceiving customers...",
  "session_id": "1e54a21d-...",
  "latency_ms": 22584,
  "timestamp": "2026-06-19T02:24:30"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "uptime_seconds": 120,
  "history_count": 5,
  "model": "qwen2.5:3b",
  "chatbot_ready": true
}
```

Full interactive documentation: **http://localhost:8000/docs** (Swagger UI)

---

## Running Evaluation

```bash
# Fast CI mode (~30s, no LLM required):
python evaluation/run_eval.py --mode retriever

# Compare with and without reranker:
python evaluation/run_eval.py --mode retriever --reranker

# Full end-to-end pipeline (requires Ollama):
python evaluation/run_eval.py --mode e2e
```

**CI Thresholds** — the script exits with code `1` if any threshold is not met:

| Metric | Minimum |
|--------|---------|
| MRR@10 | ≥ 0.60 |
| Recall@5 | ≥ 0.70 |
| Precision@5 | ≥ 0.30 |

---

## Legal Corpus

| Prefix | Document |
|--------|----------|
| `5.1` | Civil Code |
| `5.2` | Maritime Code |
| `5.3` | Code of Criminal Procedure |
| `5.4` | Code of Civil Procedure |
| `5.5` | Penal Code |
| `5.6` | Law on Intellectual Property |
| `5.7` | Law on Handling Administrative Violations |
| `5.8` | Customs Law |
| `5.9` | Commercial Law |
| `5.10` | Law on Competition |

---

## Configuration (`config.yaml`)

```yaml
llm:
  model_name: qwen2.5:3b     # swap to llama3.2:3b, phi3:mini, etc.
  base_url: http://localhost:11434

vector_db:
  url: http://localhost:6333
  collection_name: law_database
  embedding_model: intfloat/multilingual-e5-large
```

---

## Troubleshooting

**`503 Service Unavailable` on startup**
→ The chatbot is still initializing (~20s). Poll `/health` until `chatbot_ready: true`.

**Low Recall in evaluation**
→ Root cause: macOS filesystem stores paths in NFD Unicode form while JSON ground-truth uses NFC. The metric script normalizes both with `unicodedata.normalize("NFC", s)`. Without this fix, string matching silently fails even for visually identical strings.

**Reranker degrades retrieval quality**
→ Expected behavior: `BAAI/bge-reranker-base` is not optimized for Vietnamese. The reranker is disabled by default (`reranker = None` in `llm_client.py`). To enable it, uncomment the `CrossEncoderReranker` block.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Ollama (`qwen2.5:3b`) |
| Embeddings | `intfloat/multilingual-e5-large` via fastembed |
| Vector DB | Qdrant |
| BM25 | `rank_bm25` |
| Reranker | `BAAI/bge-reranker-base` via sentence-transformers |
| API | FastAPI + Uvicorn |
| Frontend | Pure HTML/CSS/JS (no framework) |
| Orchestration | LangChain (LCEL) |

---

## License

MIT License — For educational and research purposes only. All legal information provided by this system is for reference only and does not constitute legal advice.
