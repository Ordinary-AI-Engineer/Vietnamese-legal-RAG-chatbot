import yaml
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class TextEmbedder:
    """
    Nhân viên "Dịch thuật" (Embedding).
    Nhiệm vụ: Chuyển đổi các đoạn văn bản (chunks) thành các chuỗi số (vector) 
    để máy tính có thể so sánh ý nghĩa toán học.
    """
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình tại {config_path}")

        # Đọc cấu hình
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        provider = self.config['embeddings']['provider']
        model_name = self.config['embeddings']['model_name']
        
        print(f"⏳ Đang khởi động cỗ máy Embedding: {provider} ({model_name})...")
        print(f"⚡ Lưu ý: Lần chạy đầu tiên sẽ mất chút thời gian để tải model về máy tính.")
        
        # Khởi tạo mô hình FastEmbed
        # Thư viện này chạy cực kỳ tối ưu trên kiến trúc Apple Silicon
        self.embedding_model = FastEmbedEmbeddings(model_name=model_name)
        
        print("✅ Khởi động thành công! Đã sẵn sàng chuyển chữ thành số.")

    def get_embedding_model(self):
        """
        Trả về mô hình embedding để các bộ phận khác (như Vector DB) sử dụng.
        """
        return self.embedding_model

# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM (TESTING)
# ==========================================
if __name__ == "__main__":
    # Khởi tạo cỗ máy
    my_embedder = TextEmbedder(config_path="config.yaml")
    embedder_tool = my_embedder.get_embedding_model()
    
    # Dịch thử một câu luật đơn giản
    test_sentence = "Người nào xúc phạm nghiêm trọng nhân phẩm, danh dự của người khác, thì bị phạt cảnh cáo."
    print(f"\n📝 Câu gốc: '{test_sentence}'")
    
    # Biến nó thành vector
    vector_result = embedder_tool.embed_query(test_sentence)
    
    print("\n🔢 KẾT QUẢ VECTOR (Tượng trưng 5 con số đầu tiên):")
    print(vector_result[:5])
    print("-" * 40)
    print(f"👉 Chiều dài của chuỗi vector: {len(vector_result)} chiều.")
