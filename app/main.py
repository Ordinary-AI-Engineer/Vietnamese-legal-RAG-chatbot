import logging
import os
import sys
import time
import uuid
import traceback
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Setup logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("legal-api")

# ── Path setup ─────────────────────────────────────────────────
app_dir      = os.path.dirname(os.path.abspath(__file__))   # .../ChatBotLawFinal/app/
project_root = os.path.dirname(app_dir)                     # .../ChatBotLawFinal/
sys.path.insert(0, project_root)

# ── Global state ───────────────────────────────────────────────
chatbot      = None
_lock        = threading.Lock()
chat_history: List[dict] = []
startup_time = datetime.now()


# ── Startup / Shutdown ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo chatbot khi server start (eager load — tránh lazy-init timeout)."""
    global chatbot
    logger.info("🚀 Server starting — initializing Legal Chatbot...")
    try:
        from src.llm.llm_client import LawChatbot
        config_path = os.path.join(project_root, "config.yaml")
        chatbot = LawChatbot(config_path=config_path)
        logger.info("✅ Legal Chatbot ready. Server is accepting requests.")
    except Exception:
        logger.error("❌ Failed to initialize chatbot:\n" + traceback.format_exc())
    yield
    logger.info("🛑 Server shutting down.")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="⚖️ Legal AI Chatbot API",
    description="RAG-powered Vietnamese Legal Consultation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
    history_count: int
    model: str
    chatbot_ready: bool


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Kiểm tra trạng thái hệ thống."""
    import yaml
    cfg_path = os.path.join(project_root, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    uptime = int((datetime.now() - startup_time).total_seconds())
    return HealthResponse(
        status="ok" if chatbot is not None else "initializing",
        uptime_seconds=uptime,
        history_count=len(chat_history),
        model=cfg["llm"]["model_name"],
        chatbot_ready=chatbot is not None,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """Gửi câu hỏi pháp lý và nhận câu trả lời có trích dẫn nguồn."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot đang khởi tạo, vui lòng thử lại sau 30 giây.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if len(req.question) > 1000:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 1000 ký tự).")

    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"📨 [{session_id[:8]}] Q: {req.question[:80]}...")

    t0 = time.time()
    try:
        with _lock:  # 1 câu hỏi tại một thời điểm (Ollama single-threaded)
            answer = chatbot.rag_chain.invoke(req.question)
    except Exception:
        logger.error(f"❌ Chat error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý câu hỏi. Vui lòng thử lại.")

    latency_ms = int((time.time() - t0) * 1000)
    ts = datetime.now().isoformat()

    chat_history.append({
        "session_id": session_id,
        "question":   req.question,
        "answer":     answer,
        "latency_ms": latency_ms,
        "timestamp":  ts,
    })

    logger.info(f"✅ [{session_id[:8]}] Done in {latency_ms}ms")
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        latency_ms=latency_ms,
        timestamp=ts,
    )


@app.get("/history", tags=["Chat"])
def get_history(limit: int = 20):
    """Lấy lịch sử chat gần nhất."""
    return {"history": chat_history[-limit:], "total": len(chat_history)}


@app.delete("/history", tags=["Chat"])
def clear_history():
    """Xóa toàn bộ lịch sử chat."""
    chat_history.clear()
    return {"message": "Đã xóa lịch sử."}


# ── Static files (Web UI) ──────────────────────────────────────
static_dir = os.path.join(app_dir, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_ui():
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
