import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFDocumentLoader:
    """
    Nhân viên "Thu mua nguyên liệu".
    Nhiệm vụ: Quét thư mục được chỉ định, tìm tất cả file PDF và trích xuất chữ.
    """
    def __init__(self, data_directory: str = "data/raw"):
        """
        Khởi tạo người thu mua.
        :param data_directory: Đường dẫn tới thư mục chứa các file PDF luật.
        """
        self.data_directory = data_directory

    def load_all_pdfs(self) -> List[Document]:
        """
        Tìm và đọc toàn bộ file PDF trong thư mục.
        :return: Một danh sách chứa các trang tài liệu (Document) đã được số hóa.
        """
        # Kiểm tra xem thư mục có tồn tại không, nếu không thì báo lỗi rõ ràng
        if not os.path.exists(self.data_directory):
            print(f"❌ Lỗi: Không tìm thấy thư mục '{self.data_directory}'")
            print("👉 Hướng dẫn: Hãy tạo thư mục 'data/raw' ở gốc dự án và copy file PDF vào đó.")
            return []

        # Tìm tất cả các file kết thúc bằng .pdf trong thư mục
        pdf_files = glob.glob(os.path.join(self.data_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ Cảnh báo: Thư mục '{self.data_directory}' đang trống, không có file PDF nào.")
            return []

        all_documents = []
        print(f"📂 Tìm thấy {len(pdf_files)} file PDF. Đang bắt đầu đọc...")

        # Lặp qua từng file PDF để đọc
        for file_path in pdf_files:
            try:
                # In ra tên file đang xử lý để dễ theo dõi (bỏ bớt đường dẫn dài)
                file_name = os.path.basename(file_path)
                print(f"  ⏳ Đang đọc file: {file_name}...")
                
                # Gọi công cụ PyPDFLoader của LangChain để đọc file
                loader = PyPDFLoader(file_path)
                
                # Hàm load() sẽ trả về một list các trang (mỗi trang là một Document)
                docs = loader.load()
                
                all_documents.extend(docs)
                print(f"  ✅ Đã đọc xong {len(docs)} trang từ {file_name}.")
                
            except Exception as e:
                # Nếu file bị lỗi (hỏng, có pass bảo vệ), báo lỗi và tiếp tục file khác
                print(f"  ❌ Lỗi khi đọc file {file_path}: {e}")

        print(f"🎉 Hoàn tất! Tổng cộng đã thu hoạch được {len(all_documents)} trang tài liệu.")
        return all_documents

# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM (TESTING)
# ==========================================
# Đoạn code dưới đây CHỈ chạy khi bạn bấm nút Run trực tiếp file loader.py này.
# Nếu file này được import bởi file main.py, đoạn code dưới sẽ hoàn toàn im lặng.
if __name__ == "__main__":
    # Tạo nhân viên đọc file
    my_loader = PDFDocumentLoader(data_directory="data/raw")
    
    # Ra lệnh đi đọc
    extracted_pages = my_loader.load_all_pdfs()
    
    # Nếu đọc thành công, in thử 500 ký tự đầu tiên của trang đầu tiên xem kết quả
    if extracted_pages:
        print("\n--- 📖 XEM TRƯỚC NỘI DUNG (Trang 1) ---")
        print(extracted_pages[0].page_content[:500] + "...\n[ĐÃ CẮT BỚT]")
        print("---------------------------------------")
        print("👉 Metadata của trang này (Ví dụ: Tên file gốc):")
        print(extracted_pages[0].metadata)