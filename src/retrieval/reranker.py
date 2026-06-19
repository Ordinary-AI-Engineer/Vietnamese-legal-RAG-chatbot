"""
Cross-encoder Reranker cho RAG pháp lý.

Luồng:
    Top 20 candidates (từ Hybrid Retrieval)
            ↓
    CrossEncoderReranker.rerank(query, docs)
    → score mỗi cặp (query, doc) bằng neural model
            ↓
    Top 5-10 kết quả chất lượng cao nhất
"""

from typing import List
from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Reranker dùng Cross-encoder model để score lại các kết quả retrieval.
    Model: BAAI/bge-reranker-base — hỗ trợ tiếng Việt tốt, chạy được trên CPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_k: int = 5):
        """
        Args:
            model_name: Tên model HuggingFace. Mặc định BAAI/bge-reranker-base.
            top_k: Số kết quả trả về sau reranking.
        """
        from sentence_transformers import CrossEncoder

        print(f"⚡ Đang tải Cross-encoder Reranker ({model_name})...")
        # max_length=512 để không bị truncate văn bản pháp lý dài
        self.model = CrossEncoder(model_name, max_length=512)
        self.top_k = top_k
        print(f"✅ Reranker sẵn sàng! (top_k={top_k})")

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank danh sách documents theo điểm cross-encoder.

        Args:
            query: Câu hỏi của người dùng.
            docs: Danh sách candidate documents (thường Top 20 từ hybrid retrieval).

        Returns:
            Danh sách documents đã sắp xếp lại, chỉ giữ top_k tốt nhất.
        """
        if not docs:
            return []

        # Tạo cặp (query, doc_content) cho cross-encoder
        pairs = [(query, doc.page_content) for doc in docs]

        # Score tất cả cặp — cross-encoder cho điểm từng cặp riêng lẻ
        # (không như bi-encoder chỉ so sánh vector)
        scores = self.model.predict(pairs)

        # Gắn điểm vào metadata để có thể debug sau
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)

        # Sắp xếp theo điểm giảm dần → lấy top_k
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[: self.top_k]]

        return top_docs
