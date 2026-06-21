"""
ragas_eval.py — Đánh giá end-to-end pipeline RAG bằng RAGAS 0.1.21.

Metrics:
    faithfulness:       LLM có bịa thêm ngoài context không? (1.0 = không bịa gì)
    answer_relevancy:   Câu trả lời có đúng trọng tâm câu hỏi? (1.0 = hoàn toàn liên quan)
    context_precision:  Context có liên quan đến câu hỏi? (1.0 = toàn bộ context đều liên quan)
    context_recall:     Ground truth được cover bởi context? (1.0 = đủ thông tin)

Chạy:
    cd /Users/doanngocthanh/ChatBotLawFinal
    source .venv/bin/activate
    python evaluation/ragas_eval.py
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "evaluation/eval_dataset.json"
CONFIG_PATH  = "config.yaml"
OUTPUT_PATH  = "evaluation/ragas_results.json"
MAX_SAMPLES  = None          # None = tất cả 15 câu; đổi thành 5 để test nhanh
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

# ── Bước 1: Thu thập output của pipeline ──────────────────────────────────────
def collect_pipeline_outputs():
    from langchain_community.llms import Ollama as OllamaLLM
    from src.retrieval.retriever import LawRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.compressor import LegalContextCompressor
    from src.prompts.prompt_templates import LEGAL_QA_PROMPT, format_docs
    from langchain_core.output_parsers import StrOutputParser

    # ── Cache: tránh chạy lại nếu đã có kết quả ──
    cache_path = "evaluation/.ragas_cache.json"
    if os.path.exists(cache_path):
        print(f"💾 Đọc cache từ {cache_path} (bỏ qua collection)")
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return data["questions"], data["answers"], data["contexts_list"], data["ground_truths"]

    print("🔧 Khởi tạo pipeline (Reranker + Retriever + Compressor + LLM)...")
    reranker  = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", top_k=5)
    retriever = LawRetriever(CONFIG_PATH, reranker=reranker)
    compressor = LegalContextCompressor(CONFIG_PATH)
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.1)
    chain = LEGAL_QA_PROMPT | llm | StrOutputParser()

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    if MAX_SAMPLES:
        dataset = dataset[:MAX_SAMPLES]

    print(f"📋 Sẽ chạy {len(dataset)} câu hỏi\n")

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, sample in enumerate(dataset, 1):
        q  = sample["question"]
        gt = sample["ground_truth"]
        print(f"[{i:02d}/{len(dataset)}] {q[:58]}...")

        t0 = time.time()
        docs      = retriever.get_relevant_laws(q)
        docs_cmp  = compressor.compress(q, docs)

        ctx_strings  = [d.page_content for d in docs_cmp[:5]]
        context_text = format_docs(docs_cmp[:5])
        answer = chain.invoke({"context": context_text, "question": q})

        print(f"   ⏱  {time.time()-t0:.1f}s | {answer[:70].replace(chr(10),' ')}...")

        questions.append(q)
        answers.append(answer)
        contexts_list.append(ctx_strings)
        ground_truths.append(gt)

    # Lưu cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers,
                   "contexts_list": contexts_list, "ground_truths": ground_truths}, f,
                  ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu cache → {cache_path}")
    return questions, answers, contexts_list, ground_truths


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.llms import LangchainLLMWrapper          # ← tên đúng trong 0.1.21
    from langchain_community.llms import Ollama as OllamaLLM
    from langchain_community.embeddings import OllamaEmbeddings

    print("\n🔧 Cấu hình RAGAS với Ollama LLM...")

    evaluator_llm = LangchainLLMWrapper(
        langchain_llm=OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)
    )
    evaluator_emb = OllamaEmbeddings(
        model="nomic-embed-text", base_url=OLLAMA_URL
    )

    faithfulness.llm       = evaluator_llm
    answer_relevancy.llm   = evaluator_llm
    context_recall.llm     = evaluator_llm
    context_precision.llm  = evaluator_llm
    answer_relevancy.embeddings = evaluator_emb

    hf_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    print("🧮 Đang tính metrics (mỗi câu LLM gọi 3-4 lần để evaluate)...")
    print("   ⏳ Ước tính: ~5-10 phút\n")

    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result



# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("🎯 RAGAS END-TO-END EVALUATION")
    print(f"   Dataset : {DATASET_PATH}  ({MAX_SAMPLES or 'all'} samples)")
    print(f"   LLM     : {OLLAMA_MODEL} via Ollama")
    print(f"   Reranker: BAAI/bge-reranker-v2-m3")
    print("=" * 65)

    # Bước 1
    questions, answers, contexts_list, ground_truths = collect_pipeline_outputs()

    # Bước 2
    result = run_ragas_evaluation(questions, answers, contexts_list, ground_truths)

    # ── In kết quả ──
    print("\n" + "=" * 65)
    print("📊 KẾT QUẢ RAGAS:")
    print("=" * 65)

    import math

    descriptions = {
        "faithfulness":      "LLM không bịa ngoài context",
        "answer_relevancy":  "Câu trả lời đúng trọng tâm",
        "context_precision": "Context truy xuất liên quan",
        "context_recall":    "Ground truth được cover",
    }

    result_dict = {}
    for key, desc in descriptions.items():
        val = result.get(key, None)
        if val is None:
            print(f"   {desc:<38} N/A (không đo được)")
            continue
        val = float(val)
        result_dict[key] = round(val, 4)
        if math.isnan(val):
            print(f"   {desc:<38} NaN (Ollama timeout)")
        else:
            stars = "⭐" * round(val * 5)
            print(f"   {desc:<38} {val:.1%}  {stars}")

    valid_vals = [v for v in result_dict.values() if not math.isnan(v)]
    if len(valid_vals) >= 2:
        estimated_accuracy = sum(valid_vals) / len(valid_vals)
        print(f"\n   Trung bình các metrics hợp lệ: ~{estimated_accuracy:.1%}")
    print("=" * 65)

    save_data = {
        "config": {"llm_model": OLLAMA_MODEL, "reranker": "BAAI/bge-reranker-v2-m3",
                   "n_samples": len(questions)},
        "ragas_metrics": {k: (None if math.isnan(v) else v) for k, v in result_dict.items()},
        "per_question": [
            {"question": q, "answer": a[:300], "ground_truth": gt}
            for q, a, gt in zip(questions, answers, ground_truths)
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả chi tiết → {OUTPUT_PATH}")
