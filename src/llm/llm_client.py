import yaml
import os
import sys

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Đảm bảo project root trong sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import retriever and reranker from retrieval module
from src.retrieval.retriever import LawRetriever
from src.retrieval.reranker import CrossEncoderReranker
# ✅ FIX 3 & 4: Dùng format_docs đã có sẵn, không viết lại
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
        print(f"🤖 Đang đánh thức Luật sư AI (Ollama: {llm_model})...")
        # temperature=0.1 → AI nghiêm túc, bớt sáng tạo, phù hợp pháp lý
        self.llm = Ollama(model=llm_model, base_url=llm_url, temperature=0.1)

        # 2. Cross-encoder Reranker (OPTIONAL)
        # ⚠️ BAAI/bge-reranker-base chủ yếu train trên tiếng Anh/Trung.
        # Với corpus tiếng Việt, reranker làm GIẢM MRR từ 0.81 → 0.67.
        # → Để None = dùng RRF thuần (tốt hơn cho tiếng Việt hiện tại).
        # TODO: Bật lại khi có Vietnamese cross-encoder tốt hơn.
        reranker = None
        # reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-base",
        #     top_k=5
        # )

        # 3. Khởi tạo Retriever
        self.retriever = LawRetriever(config_path, reranker=reranker)


        # 3. Lắp ráp chuỗi RAG
        # ✅ FIX 2: Dùng RunnableLambda để gọi get_relevant_laws() đúng cách
        # Luồng:
        #   query → get_relevant_laws() → format_docs() → {context}
        #   query → RunnablePassthrough() → {question}
        #   {context, question} → LEGAL_QA_PROMPT → LLM → StrOutputParser → str
        retrieve_and_format = RunnableLambda(
            lambda q: format_docs(self.retriever.get_relevant_laws(q))
        )

        self.rag_chain = (
            {
                "context":  retrieve_and_format,
                "question": RunnablePassthrough()
            }
            | LEGAL_QA_PROMPT
            | self.llm
            | StrOutputParser()
        )
        print("✅ HỆ THỐNG RAG ĐÃ SẴN SÀNG NHẬN CÂU HỎI!")

    def ask(self, question: str) -> str:
        """Gửi câu hỏi và nhận câu trả lời từ LLM."""
        print(f"\n👤 KHÁCH HÀNG HỎI: '{question}'")
        print("⏳ Đang lục lọi bộ luật và soạn câu trả lời...")

        response = self.rag_chain.invoke(question)

        print("\n" + "=" * 60)
        print("⚖️  LUẬT SƯ AI TRẢ LỜI:")
        print("=" * 60)
        print(response)
        print("=" * 60)

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