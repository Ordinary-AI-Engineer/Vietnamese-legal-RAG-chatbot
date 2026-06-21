import yaml
import os
import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
import qdrant_client

from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import Ollama

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

        # Multi-Query Expansion config
        mq_cfg = retrieval_cfg.get('multi_query', {})
        self.multi_query_enabled = mq_cfg.get('enabled', False)
        self.num_variants = mq_cfg.get('num_variants', 3)

        # Auto Merging config
        am_cfg = self.config.get('auto_merging', {})
        self.auto_merging_enabled = am_cfg.get('enabled', False)
        self.merge_threshold = am_cfg.get('merge_threshold', 2)
        
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

        self.bm25_retriever, self.parent_map = self._build_bm25_and_parent_map()

        # LLM nhỏ cho Multi-Query Expansion (dùng chung Ollama)
        llm_cfg  = self.config.get('llm', {})
        self.llm = Ollama(
            model=llm_cfg.get('model_name', 'qwen2.5:3b'),
            base_url=llm_cfg.get('base_url', 'http://localhost:11434'),
            temperature=0.3,
        )

        # Optional cross-encoder reranker
        self.reranker = reranker
        if reranker:
            print(f"🎯 Cross-encoder Reranker đã được kích hoạt.")
        if self.multi_query_enabled:
            print(f"🔀 Multi-Query Expansion: BẬT ({self.num_variants} variants)")
        if self.auto_merging_enabled:
            print(f"🔗 Auto Merging Retriever: BẬT (threshold={self.merge_threshold} chunks/điều)")
        print(f"🔎 Đã khởi động HỆ THỐNG HYBRID SEARCH (Tự động gộp điểm RRF).")

    def _build_bm25_and_parent_map(self):
        """
        Scroll Qdrant một lần duy nhất → xây dựng cả:
        - BM25Retriever (tìm kiếm từ khóa)
        - parent_map: {parent_id → Document} dùng cho Auto Merging
        """
        print("⚙️ Đang tải corpus từ Qdrant (BM25 index + Parent Map)...")

        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        bm25_docs  = []   # docs cho BM25 (enhanced content)
        parent_groups: dict = {}  # {parent_id: {chunks, metadata}}

        for record in scroll_result:
            content = record.payload.get("page_content", "")
            meta    = record.payload.get("metadata", {})
            if not content:
                continue

            source  = meta.get('source', '')
            chapter = meta.get('chapter', '')
            article = meta.get('article', '')

            # ── BM25 doc ──────────────────────────────────────
            source_file   = os.path.basename(source)
            clean_source  = re.sub(r'^[\d\.]+\s*', '', source_file).replace('.pdf', '')
            enhanced      = f"{clean_source} {chapter} {article} {content}"
            meta_bm25     = dict(meta)
            meta_bm25['_original_content'] = content
            bm25_docs.append(Document(page_content=enhanced, metadata=meta_bm25))

            # ── Parent Map ────────────────────────────────────
            # Group key = (source, article) — mỗi điều luật là 1 parent
            parent_id = f"{source}|{article}"
            if parent_id not in parent_groups:
                parent_groups[parent_id] = {
                    'chunks': [],
                    'metadata': {
                        'source': source,
                        'chapter': chapter,
                        'article': article,
                        'chunk_level': 'parent',
                    }
                }
            parent_groups[parent_id]['chunks'].append(content)

        print(f"   Đã nạp {len(bm25_docs)} chunks vào BM25.")
        print(f"   Đã xây dựng {len(parent_groups)} parent nodes (điều luật).")

        # Build BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            bm25_docs, preprocess_func=custom_bm25_preprocess
        )
        bm25_retriever.k = self.candidate_pool

        # Build parent_map: merge chunks của cùng điều thành 1 Document lớn
        parent_map = {}
        for parent_id, data in parent_groups.items():
            merged_content = '\n\n'.join(data['chunks'])
            parent_map[parent_id] = Document(
                page_content=merged_content,
                metadata=data['metadata']
            )

        return bm25_retriever, parent_map

    def _auto_merge(self, docs: List[Document]) -> List[Document]:
        """
        Auto Merging Retriever:
        Nếu ≥ merge_threshold chunks từ cùng 1 điều luật được retrieve
        → thay tất cả bằng 1 Document cha (toàn bộ điều đó).
        Ngược lại → giữ nguyên chunks lẻ.
        """
        # Đếm số chunks/điều trong kết quả
        parent_hit: dict = {}   # parent_id → count
        child_groups: dict = {} # parent_id → [docs]
        orphan_docs = []        # chunks không có article metadata

        for doc in docs:
            source  = doc.metadata.get('source', '')
            article = doc.metadata.get('article', '')
            if not article:
                orphan_docs.append(doc)
                continue
            pid = f"{source}|{article}"
            parent_hit[pid]  = parent_hit.get(pid, 0) + 1
            child_groups.setdefault(pid, []).append(doc)

        result = []
        merged_pids = set()

        for pid, count in parent_hit.items():
            if count >= self.merge_threshold and pid in self.parent_map:
                # ✅ Merge: trả về toàn bộ điều luật
                parent_doc = self.parent_map[pid]
                result.append(parent_doc)
                merged_pids.add(pid)
                art_label = pid.split('|')[-1] or pid
                print(f"   🔗 Auto Merge: {count} chunks → parent [{art_label[:50]}]")
            else:
                # Giữ nguyên chunks lẻ
                result.extend(child_groups[pid])

        result.extend(orphan_docs)
        merged_count = len(merged_pids)
        if merged_count:
            print(f"   📦 Merged {merged_count} articles → context đầy đủ hơn")
        return result

    def _expand_query(self, query: str) -> List[str]:
        """
        Multi-Query Expansion: dùng LLM tạo ra {num_variants} cách hỏi khác nhau
        cho cùng một câu hỏi pháp lý. Fallback về [query] nếu LLM thất bại.
        """
        prompt = (
            f"Bạn là trợ lý pháp lý Việt Nam. "
            f"Hãy viết {self.num_variants} cách diễn đạt KHÁC NHAU cho câu hỏi dưới đây.\n"
            f"Yêu cầu:\n"
            f"- Mỗi cách trên một dòng, bắt đầu bằng số thứ tự (1. 2. 3.)\n"
            f"- Giữ nguyên ý nghĩa pháp lý cốt lõi\n"
            f"- Dùng ngôn từ khác nhau (trích dẫn điều luật, hành vi, hậu quả pháp lý...)\n"
            f"- KHÔNG giải thích, KHÔNG thêm câu dẫn\n\n"
            f"Câu hỏi gốc: {query}\n\n"
            f"{self.num_variants} cách diễn đạt khác:"
        )
        try:
            raw = self.llm.invoke(prompt)
            variants = []
            for line in raw.strip().split('\n'):
                cleaned = re.sub(r'^\d+[\.)\-]\s*', '', line.strip())
                if cleaned and len(cleaned) > 8 and cleaned.lower() != query.lower():
                    variants.append(cleaned)
            result = [query] + variants[:self.num_variants]
            print(f"   🔀 Multi-Query: {len(result)} queries ({len(result)-1} variants)")
            for i, q in enumerate(result):
                label = 'ORIGINAL' if i == 0 else f'variant {i}'
                print(f"      [{label}] {q}")
            return result
        except Exception as e:
            print(f"   ⚠️ Multi-query expansion failed, using original only: {e}")
            return [query]

    def _retrieve_single(self, query: str, source_prefix: str | None, article_number: str | None) -> Tuple[List[Document], List[Document], List[Document]]:
        """
        Chạy 3 luồng retrieval (Vector + BM25 + Targeted) cho một query đơn lẻ.
        Trả về (qdrant_docs, bm25_docs, targeted_docs).
        """
        qdrant_docs = self.qdrant_retriever.invoke(query)
        bm25_docs   = self.bm25_retriever.invoke(query)

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
                print(f"   📌 Targeted Search ({query[:30]}...): {len(targeted_docs)} chunks")
            except Exception as e:
                print(f"   ⚠️  Targeted Search thất bại: {e}")

        return qdrant_docs, bm25_docs, targeted_docs

    def get_relevant_laws(self, query: str) -> List[Document]:
        print(f"\n🧠 Đang phân tích câu hỏi: '{query}'...")

        # Bước 0: Phát hiện ý định (tên luật + số điều)
        source_prefix, article_number = _detect_intent(query)
        if source_prefix:
            print(f"   🎯 Đã phát hiện: file prefix='{source_prefix}', điều='{article_number}'")

        # Bước 1: Multi-Query Expansion (nếu bật)
        queries = self._expand_query(query) if self.multi_query_enabled else [query]

        # RRF constants — càng nhỏ → càng ưu tiên
        K_TARGETED  = 10   # filter đúng luật  → cao nhất
        K_BM25_ORIG = 30   # BM25 câu gốc
        K_VEC_ORIG  = 60   # Vector câu gốc
        K_BM25_VAR  = 50   # BM25 variant     → thấp hơn câu gốc
        K_VEC_VAR   = 80   # Vector variant

        rrf_scores: dict = {}
        doc_map:    dict = {}

        def get_doc_id(doc, is_bm25=False):
            """Hash nội dung gốc để dedup chính xác qua nhiều queries."""
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

        # Bước 2: Chạy retrieval cho từng query, gộp điểm RRF
        for q_idx, q in enumerate(queries):
            is_original = (q_idx == 0)
            k_vec = K_VEC_ORIG if is_original else K_VEC_VAR
            k_bm  = K_BM25_ORIG if is_original else K_BM25_VAR

            qdrant_docs, bm25_docs, targeted_docs = self._retrieve_single(
                q, source_prefix if is_original else None, article_number
            )

            # Targeted Search (chỉ query gốc vì đã phát hiện luật cụ thể)
            for rank, doc in enumerate(targeted_docs):
                doc_id = get_doc_id(doc)
                art_meta = doc.metadata.get('article', '')
                article_bonus = (1 / K_TARGETED) if (article_number and f'{article_number}' in art_meta) else 0
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + K_TARGETED) + article_bonus
                doc_map.setdefault(doc_id, doc)

            # Vector Search
            for rank, doc in enumerate(qdrant_docs):
                doc_id = get_doc_id(doc)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + k_vec)
                doc_map.setdefault(doc_id, doc)

            # BM25
            for rank, doc in enumerate(bm25_docs):
                doc_id = get_doc_id(doc, is_bm25=True)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + k_bm)
                doc_map.setdefault(doc_id, doc)

        # Bước 3: Xếp hạng theo tổng điểm RRF
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        print(f"   📊 RRF pool: {len(sorted_doc_ids)} unique docs từ {len(queries)} queries")

        # Bước 4: Cross-encoder Reranking (nếu có)
        if self.reranker:
            rrf_pool = [doc_map[doc_id] for doc_id in sorted_doc_ids[:20]]
            reranked = self.reranker.rerank(query, rrf_pool)
            if reranked:
                print(f"   ✨ Reranked: {len(rrf_pool)} → Top {len(reranked)}")
            final = reranked or rrf_pool
        else:
            final = [doc_map[doc_id] for doc_id in sorted_doc_ids[:self.top_k]]

        # Bước 5: Auto Merging — gộp chunks cùng điều → trả về nguyên điều
        if self.auto_merging_enabled:
            final = self._auto_merge(final)

        return final


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