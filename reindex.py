"""
reindex.py — Chạy lại toàn bộ pipeline khi đổi Embedding Model.

Dùng khi:
  - Bạn vừa đổi model trong config.yaml (ví dụ: MiniLM → bge-m3)
  - Bạn muốn cập nhật toàn bộ dữ liệu PDF mới vào Qdrant

Cách chạy:
  cd /Users/doanngocthanh/ChatBotLawFinal
  python reindex.py
"""

import sys
import os

# Đảm bảo root project luôn trong sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.loader import PDFDocumentLoader
from src.chunking.chunker import DocumentChunker
from src.vectordb.vector_store import VectorDBManager
import time
import urllib.request

def wait_for_qdrant(url: str = "http://localhost:6333", retries: int = 30, delay: int = 2):
    """Chờ Qdrant sẵn sàng trước khi chạy pipeline."""
    print(f"🔍 Kiểm tra Qdrant tại {url}...")
    for i in range(retries):
        try:
            urllib.request.urlopen(f"{url}/healthz", timeout=3)
            print(f"✅ Qdrant đã sẵn sàng!")
            return True
        except Exception:
            print(f"   ⏳ Chờ Qdrant khởi động... ({i+1}/{retries})")
            time.sleep(delay)
    raise RuntimeError("❌ Qdrant không phản hồi sau nhiều lần thử. Hãy chắc chắn Docker đang chạy.")

def main():
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU RE-INDEX TOÀN BỘ DỮ LIỆU")
    print("   Model: intfloat/multilingual-e5-large (1024 chiều)")
    print("="*60)

    # Kiểm tra Qdrant trước khi bắt đầu
    wait_for_qdrant()

    # BƯỚC 1: Đọc PDF
    print("\n📂 BƯỚC 1: Đọc tài liệu PDF...")
    loader = PDFDocumentLoader(data_directory="data/raw")
    raw_docs = loader.load_all_pdfs()

    if not raw_docs:
        print("❌ Không có tài liệu nào để xử lý. Kiểm tra thư mục data/raw.")
        return

    # BƯỚC 2: Băm nhỏ & bơm ngữ cảnh
    print("\n✂️  BƯỚC 2: Băm nhỏ & bơm ngữ cảnh pháp lý...")
    chunker = DocumentChunker(config_path="config.yaml")
    all_chunks = chunker.split_documents(raw_docs)

    # BƯỚC 3: Re-index vào Qdrant (xóa collection cũ, tạo mới)
    print("\n🗄️  BƯỚC 3: Đang nhồi dữ liệu vào Qdrant...")
    print("⚠️  force_recreate=True → Collection cũ sẽ bị XÓA và tạo lại.")
    db_manager = VectorDBManager(config_path="config.yaml")
    db_manager.build_database(all_chunks, batch_size=100, force_recreate=True)

    print("\n" + "="*60)
    print("🎉 RE-INDEX HOÀN TẤT!")
    print(f"   Tổng chunks đã nhồi: {len(all_chunks)}")
    print("   Bây giờ có thể chạy retriever.py để kiểm tra kết quả.")
    print("="*60)

if __name__ == "__main__":
    main()
