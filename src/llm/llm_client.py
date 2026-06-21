import yaml
import os
import sys
import logging

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format='%(message)s')
logger = logging.getLogger(__name__)

# Đảm bảo project root trong sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import retriever, reranker, compressor
from src.retrieval.retriever import LawRetriever

from src.retrieval.compressor import LegalContextCompressor
from src.prompts.prompt_templates import LEGAL_QA_PROMPT, format_docs


class LawChatbot:
    """
    Tổng chỉ huy: Kết nối Retriever + Prompt + LLM thành pipeline RAG hoàn chỉnh.
    """
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Không tìm thấy cấu hình tại {config_path}")

        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        # 1. Khởi tạo LLM (Ollama)
        llm_model = self.config['llm']['model_name']
        llm_url   = self.config['llm']['base_url']
        logger.info(f"🤖 Đang đánh thức Luật sư AI (Ollama: {llm_model})...")
        # temperature=0.1 → AI nghiêm túc, bớt sáng tạo, phù hợp pháp lý
        self.llm = Ollama(model=llm_model, base_url=llm_url, temperature=0.1)

        # 2. Cross-encoder Reranker
        reranker_cfg = self.config.get('reranker', {})
        if reranker_cfg.get('enabled', False):
            from src.retrieval.reranker import CrossEncoderReranker
            model_name = reranker_cfg.get('model_name', "BAAI/bge-reranker-v2-m3")
            top_k = reranker_cfg.get('top_k', 5)
            reranker = CrossEncoderReranker(
                model_name=model_name,
                top_k=top_k
            )
        else:
            reranker = None


        # 3. Khởi tạo Retriever
        self.retriever = LawRetriever(config_path, reranker=reranker)

        # 4. Contextual Compressor (OPTIONAL)
        comp_cfg = self.config.get('compression', {})
        if comp_cfg.get('enabled', False):
            self.compressor = LegalContextCompressor(config_path)
        else:
            self.compressor = None
            logger.info("🗂️  Contextual Compression: TắT")

        # 5. ITER-RETGEN config
        ir_cfg = self.config.get('iter_retgen', {})
        self.iter_retgen_enabled = ir_cfg.get('enabled', False)
        self.max_iterations      = ir_cfg.get('max_iterations', 2)

        # Mini-chain dùng chung cho cả single-pass và iterative
        self.mini_chain = LEGAL_QA_PROMPT | self.llm | StrOutputParser()

        # 6. Lắp ráp chuỗi RAG (single-pass fallback)
        # Luồng:
        #   query → get_relevant_laws() → [compress] → format_docs() → {context}
        #   query → RunnablePassthrough() → {question}
        #   {context, question} → LEGAL_QA_PROMPT → LLM → StrOutputParser → str
        def retrieve_compress_format(q: str) -> str:
            docs = self.retriever.get_relevant_laws(q)
            if self.compressor:
                docs = self.compressor.compress(q, docs)
            return format_docs(docs)

        retrieve_and_format = RunnableLambda(retrieve_compress_format)

        self.rag_chain = (
            {
                "context":  retrieve_and_format,
                "question": RunnablePassthrough()
            }
            | LEGAL_QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        mode = "ITER-RETGEN" if self.iter_retgen_enabled else "Single-pass"
        logger.info(f"✅ HỆ THỐNG RAG SẴN SÀNG | Mode: {mode}")
        if self.iter_retgen_enabled:
            logger.info(f"   🔄 ITER-RETGEN: {self.max_iterations} vòng lặp retrieve-generate")

    def _ask_iterative(self, question: str) -> str:
        """
        ITER-RETGEN: Iterative Retrieval-Generation.

        Vòng 1: retrieve(query) → generate → answer_1
        Vòng 2: retrieve(query + answer_1) → dedup → generate(all context) → answer_2
        → Trả về answer cuối cùng (tốt nhất)
        """
        seen_hashes: set = set()      # dedup content hash giữa các vòng
        accumulated_docs: list = []   # tất cả docs dùy từ mọi vòng
        previous_answer: str = ""

        for iteration in range(self.max_iterations):
            logger.info(f"\n🔄 ITER-RETGEN Vòng {iteration + 1}/{self.max_iterations}")

            # Xây dựng retrieval query
            if iteration == 0 or not previous_answer:
                retrieval_query = question
            else:
                # Dùng câu trả lời trước làm gợi ý — cắt ngắn để không làm nhiễu query
                hint = previous_answer[:250].replace("\n", " ")
                retrieval_query = f"{question}\n\n[Gợi ý từ vòng trước]: {hint}"

            # Retrieve
            new_docs = self.retriever.get_relevant_laws(retrieval_query)

            # Deduplicate: chỉ thêm docs thực sự mới
            added = 0
            for doc in new_docs:
                h = hash(doc.page_content[:150])   # hash 150 ky tu dau
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    accumulated_docs.append(doc)
                    added += 1
            logger.info(f"   📄 +{added} docs mới | Tổng tích lũy: {len(accumulated_docs)} docs")

            # Chọn top_k từ accumulated
            docs_to_use = accumulated_docs[:self.retriever.top_k]

            # Compress nếu bật
            if self.compressor:
                docs_to_use = self.compressor.compress(question, docs_to_use)

            # Generate
            context = format_docs(docs_to_use)
            previous_answer = self.mini_chain.invoke(
                {"context": context, "question": question}
            )
            logger.info(f"   ✍️  Answer_{iteration + 1}: {previous_answer[:80].replace(chr(10), ' ')}...")

        return previous_answer

    def ask(self, question: str) -> str:
        """Gửi câu hỏi và nhận câu trả lời từ LLM."""
        logger.info(f"\n👤 KHÁCH HÀNG Hỏi: '{question}'")
        logger.info("⏳ Đang lục lọi bộ luật và soạn câu trả lời...")

        if self.iter_retgen_enabled:
            response = self._ask_iterative(question)
        else:
            response = self.rag_chain.invoke(question)

        logger.info("\n" + "=" * 60)
        logger.info("⚖️  LUẬT SƯ AI TRẢ LỜI:")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)

        return response


# ==========================================
# CHẠY THỬ NGHIỆM CHATBOT (END-TO-END)
# ==========================================
if __name__ == "__main__":
    # Đảm bảo Ollama đang chạy: ollama serve
    # Và đã pull model: ollama pull llama3.2:3b
    chatbot = LawChatbot(config_path="config.yaml")

    cau_hoi = "Hành vi lừa dối khách hàng thu lợi bất chính 10 triệu đồng bị phạt như thế nào?"
    chatbot.ask(cau_hoi)