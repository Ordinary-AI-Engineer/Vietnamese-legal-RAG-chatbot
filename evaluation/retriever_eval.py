"""
Retriever Evaluation Script — đo chất lượng retrieval không cần LLM.

Metrics:
    - MRR@K  (Mean Reciprocal Rank): Rank trung bình của kết quả đúng đầu tiên
    - Recall@K: Tỷ lệ câu hỏi có ít nhất 1 kết quả đúng trong Top K
    - Precision@K: Tỷ lệ kết quả đúng trong Top K

Chạy:
    cd /Users/doanngocthanh/ChatBotLawFinal
    source .venv/bin/activate
    python evaluation/retriever_eval.py
"""

import json
import os
import sys
import time
from typing import List, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.retriever import LawRetriever
from langchain_core.documents import Document


def is_relevant(doc: Document, sample: Dict) -> bool:
    """
    Kiểm tra xem document có relevant với sample không.
    NOTE: dùng NFC normalize để tránh NFD/NFC mismatch (macOS filesystem = NFD, JSON = NFC).
    """
    import re
    import unicodedata

    def nfc(s: str) -> str:
        return unicodedata.normalize("NFC", s).lower()

    src = nfc(os.path.basename(doc.metadata.get("source", "")))
    art = nfc(doc.metadata.get("article", ""))

    # Kiểm tra source file (NFC normalized)
    source_match = any(
        nfc(exp) in src or src in nfc(exp)
        for exp in sample["relevant_sources"]
    )
    if not source_match:
        return False

    # Nếu không có relevant_articles → chỉ cần source khớp
    if not sample.get("relevant_articles"):
        return True

    # Kiểm tra article — dùng số điều thay vì so sánh Unicode
    for exp_art in sample["relevant_articles"]:
        if not exp_art:
            continue
        nums = re.findall(r'\d+', exp_art)
        for num in nums:
            # Dùng non-digit boundary để "200" không match "2000"
            if re.search(r'(?<!\d)' + num + r'(?!\d)', art):
                return True

    return False





def compute_mrr(retrieved: List[Document], sample: Dict, k: int = 10) -> float:
    """Tính MRR@K cho một câu hỏi."""
    for rank, doc in enumerate(retrieved[:k], start=1):
        if is_relevant(doc, sample):
            return 1.0 / rank
    return 0.0


def compute_recall(retrieved: List[Document], sample: Dict, k: int = 5) -> float:
    """Tính Recall@K: 1 nếu có ít nhất 1 relevant doc trong top K, 0 nếu không."""
    for doc in retrieved[:k]:
        if is_relevant(doc, sample):
            return 1.0
    return 0.0


def compute_precision(retrieved: List[Document], sample: Dict, k: int = 5) -> float:
    """Tính Precision@K: tỷ lệ relevant docs trong top K."""
    relevant_count = sum(1 for doc in retrieved[:k] if is_relevant(doc, sample))
    return relevant_count / min(k, len(retrieved)) if retrieved else 0.0


def run_retriever_eval(
    dataset_path: str = "evaluation/eval_dataset.json",
    config_path: str = "config.yaml",
    k_mrr: int = 10,
    k_recall: int = 5,
    use_reranker: bool = False,
):
    """Chạy evaluation toàn bộ dataset."""
    print("=" * 65)
    print("📊 RETRIEVER EVALUATION")
    print(f"   Dataset: {dataset_path}")
    print(f"   MRR@{k_mrr} | Recall@{k_recall} | Precision@{k_recall}")
    print(f"   Reranker: {'✅ ON' if use_reranker else '❌ OFF'}")
    print("=" * 65)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Khởi tạo retriever
    reranker = None
    if use_reranker:
        from src.retrieval.reranker import CrossEncoderReranker
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        r_cfg = cfg.get('reranker', {})
        model_name = r_cfg.get('model_name', "BAAI/bge-reranker-v2-m3")
        top_k = r_cfg.get('top_k', 5)
        reranker = CrossEncoderReranker(model_name=model_name, top_k=top_k)

    retriever = LawRetriever(config_path=config_path, reranker=reranker)

    # Chạy evaluation
    mrr_scores, recall_scores, precision_scores = [], [], []
    results = []

    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        print(f"\n[{i:02d}/{len(dataset)}] {question[:60]}...")

        start = time.time()
        docs = retriever.get_relevant_laws(question)
        elapsed = time.time() - start

        mrr    = compute_mrr(docs, sample, k=k_mrr)
        recall = compute_recall(docs, sample, k=k_recall)
        prec   = compute_precision(docs, sample, k=k_recall)

        mrr_scores.append(mrr)
        recall_scores.append(recall)
        precision_scores.append(prec)

        status = "✅" if recall > 0 else "❌"
        print(f"   {status} MRR={mrr:.3f} | Recall@{k_recall}={recall:.0f} | Prec@{k_recall}={prec:.3f} | {elapsed:.1f}s")

        # Log top 3 retrieved
        for rank, doc in enumerate(docs[:3], 1):
            src  = os.path.basename(doc.metadata.get("source", ""))
            art  = doc.metadata.get("article", "")[:50]
            rel  = "✓" if is_relevant(doc, sample) else "✗"
            score = doc.metadata.get("rerank_score", "N/A")
            score_str = f"{score:.3f}" if isinstance(score, float) else score
            print(f"      Rank {rank} [{rel}] {src} | {art} (score={score_str})")

        results.append({
            "question":   question,
            "mrr":        mrr,
            "recall_k5":  recall,
            "precision_k5": prec,
            "latency_s":  elapsed,
        })

    # Tổng kết
    avg_mrr     = sum(mrr_scores) / len(mrr_scores)
    avg_recall  = sum(recall_scores) / len(recall_scores)
    avg_prec    = sum(precision_scores) / len(precision_scores)

    print("\n" + "=" * 65)
    print("📈 KẾT QUẢ TỔNG HỢP:")
    print(f"   MRR@{k_mrr}:           {avg_mrr:.4f}")
    print(f"   Recall@{k_recall}:         {avg_recall:.4f}  ({sum(r>0 for r in recall_scores)}/{len(recall_scores)} câu tìm được)")
    print(f"   Precision@{k_recall}:      {avg_prec:.4f}")
    print("=" * 65)

    # Lưu kết quả chi tiết
    out_path = "evaluation/retriever_results.json"
    summary = {
        "config": {
            "dataset": dataset_path,
            "use_reranker": use_reranker,
            "k_mrr": k_mrr,
            "k_recall": k_recall,
        },
        "summary": {
            f"MRR@{k_mrr}": round(avg_mrr, 4),
            f"Recall@{k_recall}": round(avg_recall, 4),
            f"Precision@{k_recall}": round(avg_prec, 4),
        },
        "per_question": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Kết quả chi tiết lưu tại: {out_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retriever Evaluation")
    parser.add_argument("--dataset", default="evaluation/eval_dataset.json")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--reranker", action="store_true", help="Bật cross-encoder reranker")
    args = parser.parse_args()

    run_retriever_eval(
        dataset_path=args.dataset,
        config_path=args.config,
        use_reranker=args.reranker,
    )
