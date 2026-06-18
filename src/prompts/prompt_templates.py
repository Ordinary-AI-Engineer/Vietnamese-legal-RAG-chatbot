from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from typing import List
import os

# =====================================================================
# HÀM TIỆN ÍCH: Chuyển List[Document] → Context string có metadata
# =====================================================================
def format_docs(docs: List[Document]) -> str:
    """
    Chuyển danh sách Document thành đoạn context có cấu trúc rõ ràng.
    Mỗi đoạn đều kèm NGUỒN (file, điều, chương) để LLM có thể trích dẫn chính xác.
    
    Ví dụ output:
        [Đoạn 1] 📌 Nguồn: 5.5. Bộ luật hình sự.pdf | Chương XVIII | Điều 198. Tội lừa dối khách hàng
        Điều 198. Tội lừa dối khách hàng...
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source  = doc.metadata.get('source', 'Không rõ nguồn')
        chapter = doc.metadata.get('chapter', '').strip()
        article = doc.metadata.get('article', '').strip()
        content = doc.page_content.strip()

        source_name = os.path.basename(source)
        header = f"[Đoạn {i}] 📌 Nguồn: {source_name}"
        if chapter:
            header += f" | {chapter}"
        if article:
            header += f" | {article}"

        formatted.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted)


# =====================================================================
# TEMPLATE CHO CHATBOT PHÁP LÝ (LEGAL RAG TEMPLATE)
# =====================================================================
LEGAL_SYSTEM_PROMPT = """Bạn là Luật sư trợ lý ảo chuyên tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời DỰA HOÀN TOÀN VÀO CÁC ĐOẠN VĂN BẢN PHÁP LUẬT được cung cấp bên dưới.

===== QUY TẮC TUYỆT ĐỐI =====

QUY TẮC 1 - KHÔNG BỊA ĐẶT:
Nếu Context KHÔNG có thông tin để trả lời, chỉ được nói đúng một câu này và DỪNG LẠI NGAY:
"Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu pháp luật hiện tại."
TUYỆT ĐỐI KHÔNG thêm bất kỳ thông tin nào khác sau câu đó.

QUY TẮC 2 - CHỈ DÙNG SỐ LIỆU TRONG CONTEXT:
Không được tự thêm bất kỳ con số nào (mức phạt, số năm tù, số tiền...) nếu không có trong Context.

QUY TẮC 3 - TRÍCH DẪN NGUỒN:
Cuối mỗi câu trả lời PHẢI có: "📌 Căn cứ: [tên Điều] - [tên Luật]"

QUY TẮC 4 - DISCLAIMER:
Kết thúc bằng: "⚠️ Thông tin chỉ mang tính tham khảo. Vui lòng tư vấn luật sư có chuyên môn."

===== VĂN BẢN PHÁP LUẬT (CONTEXT) =====
{context}
===================================
"""

LEGAL_HUMAN_PROMPT = """Câu hỏi: {question}"""

# Lắp ráp thành Prompt hoàn chỉnh
LEGAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(LEGAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(LEGAL_HUMAN_PROMPT)
])