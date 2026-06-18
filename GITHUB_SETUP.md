# GitHub Portfolio Setup

## Bước 1 — Tạo repo trên GitHub

1. Vào https://github.com/new
2. Repository name: `legalbot-ai` hoặc `vietnamese-legal-rag`
3. Description: `Production-grade RAG chatbot for Vietnamese law | Hybrid Search (BM25 + Vector) + FastAPI + Next.js`
4. Set **Public** (để portfolio thấy được)
5. **Không** tick "Add README" (mình đã có rồi)
6. Click "Create repository"

## Bước 2 — Push code

```bash
cd /Users/doanngocthanh/ChatBotLawFinal

# Add tất cả files (gitignore sẽ tự loại PDF, venv, qdrant_data)
git add .

# Commit đầu tiên
git commit -m "feat: production RAG chatbot for Vietnamese legal consultation

- Hybrid retrieval: Targeted + Vector (multilingual-E5) + BM25 via RRF
- MRR@10=0.81, Recall@5=87% on 15-query eval set
- FastAPI backend with eager init + thread-safe LLM lock
- Next.js 16 frontend with citation extraction
- CI-gated evaluation pipeline
- Docker Compose: Qdrant + API"

# Kết nối với GitHub repo vừa tạo (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/legalbot-ai.git
git branch -M main
git push -u origin main
```

## Bước 3 — Thêm GitHub topics

Sau khi push xong, vào repo → ⚙️ → Topics, thêm:
```
rag  llm  langchain  fastapi  nextjs  qdrant  vietnamese-nlp  information-retrieval  ollama  bm25
```

## Bước 4 — Đặt "About" description

```
⚖️ Production RAG chatbot for Vietnamese law | Hybrid BM25+Vector search | MRR@10=0.81 | FastAPI + Next.js
```
