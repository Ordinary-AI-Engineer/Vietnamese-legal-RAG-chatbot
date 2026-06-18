import yaml
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
import qdrant_client

class VectorDBManager:
    """
    Quản lý Kho dữ liệu Vector (Qdrant).
    Nhiệm vụ: Nhận các đoạn text, gọi Embedder dịch ra số (trên RAM), 
    rồi lưu vĩnh viễn cả chữ lẫn số vào Database Qdrant.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # 1. Đọc cấu hình
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.collection_name = self.config['vector_store']['collection_name']
        self.host = self.config['vector_store']['host']
        self.port = self.config['vector_store']['port']
        self.url = f"http://{self.host}:{self.port}"
        
        # 2. Khởi tạo Client kết nối tới máy chủ Qdrant đang chạy ngầm
        # ✅ FIX timeout: tăng lên 300s cho collection lớn (mặc định chỉ 5s)
        self.client = qdrant_client.QdrantClient(url=self.url, timeout=300)
        print(f"🔌 Đã kết nối tới Qdrant tại {self.url}")

        # 3. Lấy cỗ máy dịch thuật (Embedder) từ module bên cạnh
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        from src.embeddings.embedder import TextEmbedder
        self.embedder = TextEmbedder(config_path).get_embedding_model()

    def build_database(self, chunks: List[Document], batch_size: int = 100, force_recreate: bool = True):
        """
        ✨ CƠ CHẾ BATCHING INGESTION (CHUẨN ENTERPRISE)
        Nhận vào toàn bộ chunks, chia thành các mẻ nhỏ để không làm tràn RAM.
        
        Args:
            force_recreate: Nếu True (mặc định), XÓA collection cũ trước khi tạo mới.
                            ⚠️ BẮT BUỘC = True khi đổi embedding model (số chiều thay đổi).
        """
        from qdrant_client.models import Distance, VectorParams

        total_chunks = len(chunks)
        print(f"📦 Đang chuẩn bị nhồi {total_chunks} đoạn văn bản vào Qdrant...")
        print(f"⚙️ Chế độ an toàn RAM: Xử lý theo mẻ (Batching) - {batch_size} chunks/mẻ.")
        
        # ✅ FIX ROOT CAUSE: Dùng self.client (timeout=300s) cho MỌI thao tác
        # KHÔNG dùng Qdrant.from_documents() — nó tạo client nội bộ với timeout mặc định 5s

        # Bước 1: Xóa collection cũ nếu cần
        if force_recreate:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in collections:
                print(f"🗑️  Đang xóa collection cũ '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
                print(f"✅ Đã xóa collection cũ.")

        # Bước 2: Lấy số chiều vector bằng cách embed thử 1 câu mẫu
        print("📐 Đang xác định số chiều vector của model...")
        sample_vector = self.embedder.embed_query("test")
        vector_size = len(sample_vector)
        print(f"   → Model tạo ra vector {vector_size} chiều.")

        # Bước 3: Tạo collection mới thủ công qua self.client (đã có timeout=300s)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"🗄️  Đã tạo collection '{self.collection_name}' ({vector_size} chiều, COSINE).")

        # Bước 4: Dùng Qdrant() constructor với client đã có — KHÔNG tạo client mới
        vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embedder,
        )

        # Bước 5: Nhồi dữ liệu theo từng mẻ
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            current_progress = min(i + batch_size, total_chunks)
            print(f"  ⏳ Tiến độ: {current_progress}/{total_chunks} chunks...")
            
        print(f"✅ HOÀN TẤT! Toàn bộ {total_chunks} chunks đã nằm an toàn trong ổ cứng.")

# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM TỔNG HỢP (PIPELINE END-TO-END)
# ==========================================
if __name__ == "__main__":
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    
    from src.ingestion.loader import PDFDocumentLoader
    from src.chunking.chunker import DocumentChunker

    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU CHẠY ĐƯỜNG ỐNG DỮ LIỆU (DATA PIPELINE)")
    print("="*50)

    # BƯỚC 1: Đọc PDF
    loader = PDFDocumentLoader(data_directory="data/raw")
    raw_docs = loader.load_all_pdfs()

    # BƯỚC 2: Băm nhỏ và tiêm ngữ cảnhs
    chunker = DocumentChunker(config_path="config.yaml")
    all_chunks = chunker.split_documents(raw_docs)

    # BƯỚC 3: Nhồi toàn bộ vào Database (Batching)
    if all_chunks:
        db_manager = VectorDBManager(config_path="config.yaml")
        
        # Bắt đầu nhồi toàn bộ dữ liệu (8000+ chunks) với mẻ 100 đoạn/lần
        db_manager.build_database(all_chunks, batch_size=100)