import yaml
import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Nhân viên "Băm nhỏ nguyên liệu" (Chunking) - Phiên bản LEVEL 3.
    Nhiệm vụ: Cắt tài liệu và BƠM NGỮ CẢNH (Chương, Điều) vào từng đoạn nhỏ.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Đọc cấu hình và thiết lập dao cắt Regex.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình tại {config_path}")

        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        chunk_size = self.config['chunking']['chunk_size']
        chunk_overlap = self.config['chunking']['chunk_overlap']
        
        # Dao cắt Regex (Cấp độ 2)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                r"\n(?=Phần \d+|Phần thứ)",
                r"\n(?=Chương [IVXLCDM]+)", 
                r"\n(?=Điều \d+)",          
                r"\n(?=Khoản \d+)",         
                r"\n\n",                    
                r"\n",                      
                r"\.",                      
                r" ",                       
                ""
            ],
            is_separator_regex=True
        )
        print(f"⚙️  Đã thiết lập dao cắt THÔNG MINH: {chunk_size} ký tự/đoạn.")

    def _enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        LEVEL 3: Thuật toán quét và bơm ngữ cảnh pháp lý.
        """
        print("💉 Đang tiến hành bơm ngữ cảnh (Chương, Điều) vào các đoạn cắt...")
        current_chapter = "Chưa xác định Chương"
        current_article = "Chưa xác định Điều"

        for chunk in chunks:
            text = chunk.page_content

            # 1. Quét tìm tiêu đề Chương trong đoạn text (VD: "Chương I: Quy định chung")
            chapter_match = re.search(r"(Chương\s+[IVXLCDM]+[\.\:\-\s]+[^\n]+)", text, re.IGNORECASE)
            if chapter_match:
                # Nếu thấy, cập nhật trí nhớ
                current_chapter = chapter_match.group(1).strip()

            # 2. Quét tìm tiêu đề Điều (VD: "Điều 1. Phạm vi điều chỉnh")
            article_match = re.search(r"(Điều\s+\d+[\.\:\-\s]+[^\n]+)", text, re.IGNORECASE)
            if article_match:
                # Nếu thấy, cập nhật trí nhớ
                current_article = article_match.group(1).strip()

            # 3. Gắn nhãn (Metadata) ẩn để sau này lọc dữ liệu nâng cao
            chunk.metadata['chapter'] = current_chapter
            chunk.metadata['article'] = current_article

            # 4. BƠM TRỰC TIẾP VÀO NỘI DUNG CHỮ
            # Rất quan trọng: VectorDB so sánh bằng chữ. Có dán chữ vào đây thì AI mới tìm ra.
            # Chỉ dán bối cảnh nếu đoạn này là đoạn "cắt dở" (không bắt đầu bằng chữ Chương hoặc Điều)
            if not text.startswith("Chương") and not text.startswith("Điều"):
                context_prefix = f"[{current_chapter} | {current_article}]\n"
                chunk.page_content = context_prefix + text

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Quy trình xử lý hoàn chỉnh: Cắt -> Bơm ngữ cảnh
        """
        print(f"🔪 Đang băm nhỏ {len(documents)} trang tài liệu. Vui lòng đợi...")
        
        # Bước 1: Cắt nhỏ
        raw_chunks = self.splitter.split_documents(documents)
        
        # Bước 2: Bơm ngữ cảnh (Level 3)
        enriched_chunks = self._enrich_metadata(raw_chunks)
        
        print(f"✅ Hoàn tất! Từ {len(documents)} trang, đã tạo ra {len(enriched_chunks)} đoạn thông minh.")
        return enriched_chunks

# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM (TESTING)
# ==========================================
if __name__ == "__main__":
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    
    from src.ingestion.loader import PDFDocumentLoader
    
    print("\n--- BƯỚC 1: ĐỌC TÀI LIỆU ---")
    my_loader = PDFDocumentLoader(data_directory="data/raw")
    raw_pages = my_loader.load_all_pdfs()
    
    if raw_pages:
        print("\n--- BƯỚC 2 & 3: CẮT & BƠM NGỮ CẢNH ---")
        my_chunker = DocumentChunker(config_path="config.yaml")
        final_chunks = my_chunker.split_documents(raw_pages)
        
        # Tìm thử 2 đoạn không nằm ở trang bìa để xem kết quả bơm (VD lấy đoạn thứ 50 và 51)
        test_index = min(50, len(final_chunks) - 1)
        
        print("\n--- ✂️ XEM TRƯỚC ĐOẠN ĐÃ CẮT ---")
        print(final_chunks[test_index].page_content)
        print("-" * 40)
        print("👉 METADATA ẨN CỦA ĐOẠN NÀY:", final_chunks[test_index].metadata)
        
        print("\n--- ✂️ XEM TRƯỚC ĐOẠN TIẾP THEO ---")
        print(final_chunks[test_index+1].page_content)
        print("-" * 40)
        print("👉 METADATA ẨN CỦA ĐOẠN NÀY:", final_chunks[test_index+1].metadata)