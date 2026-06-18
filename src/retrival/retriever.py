import yaml
import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
import qdrant_client

from langchain_community.retrievers import BM25Retriever

def custom_bm25_preprocess(text: str) -> List[str]:
    """
    Hàm tiền xử lý chuyên dụng Legal NLP.
    Ép các từ khóa quan trọng thành những token độc nhất để khắc phục Length Penalty của BM25.
    """
    text = text.lower()
    text = text.replace('.pdf', '')
    
    text = text.replace('bộ luật tố tụng hình sự', 'bltths')
    text = text.replace('luật tố tụng hình sự', 'bltths')
    text = text.replace('bộ luật tố tụng dân sự', 'blttds')
    text = text.replace('luật tố tụng dân sự', 'blttds')
    text = text.replace('bộ luật hình sự', 'blhs')
    text = text.replace('luật hình sự', 'blhs')
    text = text.replace('bộ luật dân sự', 'blds')
    text = text.replace('luật dân sự', 'blds')
    text = text.replace('luật sở hữu trí tuệ', 'lshtt')
    text = text.replace('luật thương mại', 'ltm')
    text = text.replace('luật hải quan', 'lhq')
    text = text.replace('luật cạnh tranh', 'lct')
    text = text.replace('luật xử lý vi phạm hành chính', 'lxlvphc')
    
    text = re.sub(r'điều\s+(\d+)', r'dieu_\1', text)
    text = re.sub(r'khoản\s+(\d+)', r'khoan_\1', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    # Stopwords pháp lý: loại bỏ từ phổ biến không phân biệt được điều luật cụ thể
    # "nội", "dung" được thêm lại vì gây BM25 match sai với "nội dung quyết định"
    stopwords = {"nội", "dung", "về", "gì", "thế", "nào", "là", "các", "có", "cho", "để", "của", "và", "như", "một", "sự", "bị", "những", "này", "thuộc", "theo", "tại", "với", "trong", "ra", "lên", "được"}
    
    return [w for w in words if w not in stopwords]

# ==========================================
# BẢN ĐỒ TUỶN LUẬT: phát hiện tên luật trong query
# Đầu file trong data/raw/ có dạng: "5.5. Bộ luật hình sự.pdf"
# ==========================================
LAW_SOURCE_MAP = [
    # (regex pattern, unique substring trong tên file)
    (re.compile(r'bộ\s*luật\s*tố\s*tụng\s*hình\s*sự|bltths', re.I),  '5.3'),
    (re.compile(r'bộ\s*luật\s*tố\s*tụng\s*dân\s*sự|blttds',  re.I),  '5.4'),
    (re.compile(r'bộ\s*luật\s*hình\s*sự|blhs',               re.I),  '5.5'),
    (re.compile(r'luật\s*sở\s*hữu\s*trí\s*tuệ|lshtt',          re.I),  '5.1'),
    (re.compile(r'luật\s*cạnh\s*tranh',                        re.I),  '5.10'),
    (re.compile(r'luật\s*thương\s*mại|ltm',                   re.I),  '5.11'),
    (re.compile(r'luật\s*xử\s*lý\s*vi\s*phạm\s*hành\s*chính|lxlvphc', re.I), '5.7'),
    (re.compile(r'luật\s*hải\s*quan',                          re.I),  '5.8'),
    (re.compile(r'luật\s*khoa\s*học|luật\s*khcn',              re.I),  '5.9'),
]

def _detect_intent(query: str):
    """
    Phát hiện (tên luật, số điều) trong query bằng regex.
    Trả về: (source_prefix: str | None, article_number: str | None)
    """
    source_prefix = None
    for pattern, prefix in LAW_SOURCE_MAP:
        if pattern.search(query):
            source_prefix = prefix
            break

    article_match = re.search(r'điều\s*(\d+)', query, re.I)
    article_number = article_match.group(1) if article_match else None

    return source_prefix, article_number


class LawRetriever:
    """
    Nhân viên "Lưu trữ & Truy xuất" (Retriever) - PHIÊN BẢN HYBRID HOÀN HẢO.
    Tự động gộp điểm RRF thủ công để khắc phục lỗi không nhận diện được văn bản của Langchain.
    """
    def __init__(self, config_path: str = "config.yaml", reranker=None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình tại {config_path}")

        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.collection_name = self.config['vector_store']['collection_name']
        self.host = self.config['vector_store']['host']
        self.port = self.config['vector_store']['port']
        self.url = f"http://{self.host}:{self.port}"
        
        retrieval_cfg = self.config.get('retrieval', {})
        self.top_k = retrieval_cfg.get('top_k', 10)
        self.candidate_pool = retrieval_cfg.get('candidate_pool', 50)
        
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        from src.embeddings.embedder import TextEmbedder
        self.embedder = TextEmbedder(config_path).get_embedding_model()

        self.client = qdrant_client.QdrantClient(url=self.url, timeout=300)
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embedder
        )
        
        self.qdrant_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.candidate_pool})
        print(f"🔌 Đã kết nối Qdrant (Vector Search).")

        self.bm25_retriever = self._build_bm25_retriever()
        # Optional cross-encoder reranker (None = dùng RRF score thuần túy)
        self.reranker = reranker
        if reranker:
            print(f"🎯 Cross-encoder Reranker đã được kích hoạt.")
        print(f"🔎 Đã khởi động HỆ THỐNG HYBRID SEARCH (Tự động gộp điểm RRF).")

    def _build_bm25_retriever(self) -> BM25Retriever:
        print("⚙️ Đang tải văn bản từ Qdrant để lập chỉ mục từ khóa BM25...")
        
        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        docs = []
        for record in scroll_result:
            content = record.payload.get("page_content", "")
            meta = record.payload.get("metadata", {})
            if content:
                source_file = os.path.basename(meta.get('source', ''))
                clean_source = re.sub(r'^[\d\.]+\s*', '', source_file).replace('.pdf', '')
                chapter = meta.get('chapter', '')
                article = meta.get('article', '')
                
                enhanced_content = f"{clean_source} {chapter} {article} {content}"
                # ✅ FIX Vấn đề 4: Lưu nội dung GỐC vào metadata để dedup sau này
                # tránh phụ thuộc vào strip prefix fragile
                meta_with_original = dict(meta)
                meta_with_original['_original_content'] = content
                docs.append(Document(page_content=enhanced_content, metadata=meta_with_original))
                
        print(f"   Đã nạp {len(docs)} đoạn văn bản vào bộ nhớ BM25.")
        
        bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=custom_bm25_preprocess)
        bm25_retriever.k = self.candidate_pool
        return bm25_retriever

    def get_relevant_laws(self, query: str) -> List[Document]:
        print(f"\n🧠 Đang phân tích câu hỏi: '{query}'...")
        
        # ✅ Bước 0: Phát hiện ý định (luật cụ thể + số điều)
        source_prefix, article_number = _detect_intent(query)
        if source_prefix:
            print(f"   🎯 Đã phát hiện: file prefix='{source_prefix}', điều='{article_number}'")
        
        # 1. Chạy 2 luồng tìm kiếm chính
        qdrant_docs = self.qdrant_retriever.invoke(query)
        bm25_docs   = self.bm25_retriever.invoke(query)
        
        # 2. Luồng thứ 3 (Targeted): chỉ chạy khi phát hiện luật cụ thể
        targeted_docs = []
        if source_prefix:
            from qdrant_client.models import Filter, FieldCondition, MatchText
            targeted_filter = Filter(
                must=[FieldCondition(
                    key="metadata.source",
                    match=MatchText(text=source_prefix)
                )]
            )
            targeted_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 20, "filter": targeted_filter}
            )
            try:
                targeted_docs = targeted_retriever.invoke(query)
                print(f"   📌 Targeted Search: tìm được {len(targeted_docs)} chunks từ luật chỉ định.")
            except Exception as e:
                print(f"   ⚠️  Targeted Search thất bại (bỏ qua): {e}")

        # 3. RRF: hằng số k — càng nhỏ → càng ưu tiên
        K_TARGETED = 10   # ưu tiên cao nhất (có filter đúng luật)
        K_BM25     = 30   # ưu tiên từ khóa
        K_VECTOR   = 60   # ngữ nghĩa

        rrf_scores = {}
        doc_map    = {}

        def get_doc_id(doc, is_bm25=False):
            """Dùng hash nội dung gốc — không bao giờ collision."""
            src  = doc.metadata.get('source', '')
            chap = doc.metadata.get('chapter', '')
            art  = doc.metadata.get('article', '')

            if is_bm25:
                original = doc.metadata.get('_original_content')
                if original is not None:
                    doc.page_content = original
                    content = original
                else:
                    content = doc.page_content
            else:
                content = doc.page_content

            return f"{src}|{chap}|{art}|{hash(content)}"

        # 4. Chấm điểm Targeted Search (cao nhất)
        for rank, doc in enumerate(targeted_docs):
            doc_id = get_doc_id(doc)
            # Bonus: nếu có số điều đúng → bonus thêm
            art_meta = doc.metadata.get('article', '')
            article_bonus = 0
            if article_number and f'{article_number}' in art_meta:
                article_bonus = 1 / K_TARGETED  # gấp đôi điểm
            rrf_scores[doc_id] = 1 / (rank + K_TARGETED) + article_bonus
            doc_map[doc_id] = doc

        # 5. Chấm điểm Vector Search
        for rank, doc in enumerate(qdrant_docs):
            doc_id = get_doc_id(doc, is_bm25=False)
            score  = 1 / (rank + K_VECTOR)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 6. Chấm điểm BM25
        for rank, doc in enumerate(bm25_docs):
            doc_id = get_doc_id(doc, is_bm25=True)
            score  = 1 / (rank + K_BM25)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 7. Xếp hạng — lấy Top candidates
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 8. Cross-encoder Reranking (nếu có reranker)
        if self.reranker:
            # Lấy Top 20 candidates từ RRF → đưa vào reranker
            rrf_pool = [doc_map[doc_id] for doc_id in sorted_doc_ids[:20]]
            reranked = self.reranker.rerank(query, rrf_pool)
            if reranked:
                print(f"   ✨ Reranked: {len(rrf_pool)} candidates → Top {len(reranked)}")
            return reranked

        # Không có reranker → trả về RRF top_k như cũ
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:self.top_k]]


# ==========================================
# KHU VỰC CHẠY THỬ NGHIỆM (TESTING)
# ==========================================
if __name__ == "__main__":
    my_retriever = LawRetriever(config_path="config.yaml")
    
    # Thử lại câu hỏi hóc búa để xem "trùm cuối" thể hiện
    test_query = "Nội dung Điều 198 Bộ luật hình sự quy định về tội gì?"
    
    retrieved_docs = my_retriever.get_relevant_laws(test_query)
    
    print(f"\n✅ TÌM THẤY {len(retrieved_docs)} ĐOẠN LUẬT TỐT NHẤT:")
    print("=" * 60)
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n🎯 KẾT QUẢ SỐ {i}:")
        source_file = os.path.basename(doc.metadata.get('source', 'Không rõ nguồn'))
        chapter = doc.metadata.get('chapter', '')
        article = doc.metadata.get('article', '')
        
        print(f"👉 Nguồn: {source_file}")
        print(f"👉 Vị trí: {chapter} | {article}")
        print("-" * 30)
        print(doc.page_content)
        print("=" * 60)