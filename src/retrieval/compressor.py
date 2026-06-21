"""
compressor.py — Contextual Compression cho Vietnamese Legal RAG.

Sau khi retrieve docs → dùng LLM trích xuất ĐÚNG đoạn liên quan đến câu hỏi,
loại bỏ phần nhiễu → LLM trả lời chính xác hơn, ít hallucination hơn.

Luồng:
    query + docs → [LLM Extractor] → docs_compressed → RAG chain
"""

import yaml
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.llms import Ollama


# Prompt trích xuất nội dung pháp lý liên quan
EXTRACTION_PROMPT = """Bạn là trợ lý pháp lý chuyên nghiệp. Nhiệm vụ của bạn là trích xuất CÁC ĐOẠN LIÊN QUAN từ văn bản luật.

Câu hỏi: {query}

Văn bản pháp lý:
{text}

Hướng dẫn:
- Chỉ giữ lại các câu/khoản/điều TRỰC TIẾP liên quan đến câu hỏi.
- Giữ nguyên văn bản gốc, KHÔNG diễn giải hay thêm thông tin.
- Nếu toàn bộ văn bản đều liên quan → trả về nguyên văn.
- KHÔNG thêm tiêu đề, giải thích, hay dấu "---".

Đoạn liên quan:"""


class LegalContextCompressor:
    """
    Contextual Compression Retriever (custom cho pháp lý Việt Nam).

    Sau khi RRF + Auto Merging trả về docs → compressor chạy LLM nhỏ
    để extract chỉ phần thật sự trả lời câu hỏi.

    Lợi ích:
        - Giảm context tokens → LLM trả lời nhanh hơn, ít hallucination
        - Loại bỏ nhiễu từ các đoạn cùng điều nhưng không liên quan
        - Đặc biệt hữu ích sau Auto Merging (parent doc có thể rất dài)
    """

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy config tại {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_cfg = config.get("llm", {})
        comp_cfg = config.get("compression", {})

        # Dùng cùng Ollama model nhưng temperature=0 → trích xuất deterministic
        self.llm = Ollama(
            model=llm_cfg.get("model_name", "qwen2.5:3b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=0.0,
        )

        # Chỉ compress top N docs (tránh quá nhiều LLM calls)
        self.max_docs     = comp_cfg.get("max_docs", 5)
        # Nếu kết quả extract ngắn hơn min_length → fallback giữ doc gốc
        self.min_length   = comp_cfg.get("min_length", 40)

        print(f"🗜️  Contextual Compressor: max_docs={self.max_docs}, min_length={self.min_length}")

    def compress(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Trích xuất phần liên quan từ top max_docs docs.
        Các docs còn lại (rank thấp) giữ nguyên.

        Args:
            query: câu hỏi người dùng
            docs:  danh sách documents sau retrieval + merging

        Returns:
            Danh sách docs đã được nén nội dung
        """
        compressed_docs = []

        for i, doc in enumerate(docs):
            if i >= self.max_docs:
                # Docs ngoài top max_docs → giữ nguyên (không tốn LLM call)
                compressed_docs.append(doc)
                continue

            prompt = EXTRACTION_PROMPT.format(
                query=query,
                text=doc.page_content
            )

            try:
                extracted = self.llm.invoke(prompt).strip()

                if len(extracted) >= self.min_length:
                    # Tạo doc mới với content đã nén, giữ nguyên metadata
                    new_doc = Document(
                        page_content=extracted,
                        metadata={**doc.metadata, "compressed": True,
                                  "original_length": len(doc.page_content),
                                  "compressed_length": len(extracted)}
                    )
                    ratio = len(extracted) / max(len(doc.page_content), 1)
                    print(f"   🗜️  Compressed doc[{i+1}]: "
                          f"{len(doc.page_content)} → {len(extracted)} chars "
                          f"({ratio:.0%})")
                    compressed_docs.append(new_doc)
                else:
                    # Extracted quá ngắn → giữ nguyên doc gốc
                    print(f"   ⚠️  Doc[{i+1}] extract quá ngắn ({len(extracted)} chars) → giữ nguyên")
                    compressed_docs.append(doc)

            except Exception as e:
                print(f"   ⚠️  Compress doc[{i+1}] thất bại: {e} → giữ nguyên")
                compressed_docs.append(doc)

        total_before = sum(len(d.page_content) for d in docs[:self.max_docs])
        total_after  = sum(len(d.page_content) for d in compressed_docs[:self.max_docs])
        if total_before > 0:
            print(f"   📉 Tổng context: {total_before} → {total_after} chars "
                  f"({total_after/total_before:.0%} còn lại)")

        return compressed_docs
