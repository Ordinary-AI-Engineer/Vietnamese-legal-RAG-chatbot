"""
Run Eval — pipeline evaluation đầy đủ (Retriever + LLM end-to-end).

Chế độ:
    1. retriever-only: nhanh, không cần LLM, dùng trong CI
    2. e2e: chạy toàn bộ pipeline RAG, đo Faithfulness đơn giản

Chạy:
    # Chỉ đo retriever (CI mode):
    python evaluation/run_eval.py --mode retriever

    # Chạy full end-to-end (thêm LLM faithfulness check):
    python evaluation/run_eval.py --mode e2e

    # So sánh có/không reranker:
    python evaluation/run_eval.py --mode retriever --reranker
"""

import json
import os
import sys
import time
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ===================== CI THRESHOLDS (fail nếu thấp hơn) =====================
CI_THRESHOLDS = {
    "MRR@10":      0.60,   # Minimum acceptable MRR
    "Recall@5":    0.70,   # Minimum acceptable Recall
    "Precision@5": 0.30,   # Minimum acceptable Precision
}


def run_retriever_mode(dataset_path: str, config_path: str, use_reranker: bool) -> dict:
    """Chạy evaluation chỉ đo Retriever metrics."""
    from evaluation.retriever_eval import run_retriever_eval
    return run_retriever_eval(
        dataset_path=dataset_path,
        config_path=config_path,
        use_reranker=use_reranker,
    )


def run_e2e_mode(dataset_path: str, config_path: str) -> dict:
    """
    Chạy evaluation end-to-end: Retriever + LLM.
    Đo Faithfulness đơn giản: kiểm tra LLM có trích dẫn nguồn không.
    """
    from src.llm.llm_client import LawChatbot

    print("\n" + "=" * 65)
    print("🔄 END-TO-END EVALUATION (Retriever + LLM)")
    print("=" * 65)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    chatbot = LawChatbot(config_path)
    results = []
    citation_hits = 0

    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        print(f"\n[{i:02d}/{len(dataset)}] {question[:55]}...")

        start = time.time()
        response = chatbot.ask(question)
        elapsed = time.time() - start

        # Faithfulness heuristic: LLM có nói "Căn cứ" / "Điều" không?
        has_citation = any(kw in response for kw in ["Căn cứ", "Điều", "📌", "Chương"])
        # Hallucination flag: LLM có nói "Xin lỗi" nhưng vẫn trả lời dài không?
        said_no_info = "không tìm thấy" in response.lower()
        answered_anyway = len(response.split()) > 50 and said_no_info

        if has_citation:
            citation_hits += 1

        status = "✅" if has_citation and not answered_anyway else "⚠️"
        print(f"   {status} Citation={has_citation} | Halluc={answered_anyway} | {elapsed:.1f}s")

        results.append({
            "question":       question,
            "has_citation":   has_citation,
            "hallucination":  answered_anyway,
            "latency_s":      elapsed,
        })

    citation_rate = citation_hits / len(dataset)
    halluc_count  = sum(r["hallucination"] for r in results)

    print("\n" + "=" * 65)
    print("📈 E2E KẾT QUẢ:")
    print(f"   Citation Rate:    {citation_rate:.2%}  ({citation_hits}/{len(dataset)})")
    print(f"   Hallucinations:   {halluc_count}/{len(dataset)}")
    print("=" * 65)

    out_path = "evaluation/e2e_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"citation_rate": citation_rate,
                               "hallucination_count": halluc_count},
                   "per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"💾 Kết quả lưu tại: {out_path}")

    return {"citation_rate": citation_rate, "hallucination_count": halluc_count}


def check_ci_thresholds(metrics: dict) -> bool:
    """Kiểm tra CI thresholds. Trả về True nếu pass."""
    print("\n" + "=" * 65)
    print("🚦 CI GATE CHECK:")
    all_pass = True
    for metric, threshold in CI_THRESHOLDS.items():
        # Try both key formats: "MRR@10" and "MRR@10"
        value = metrics.get("summary", {}).get(metric, metrics.get(metric))
        if value is None:
            continue
        passed = value >= threshold
        icon = "✅" if passed else "❌"
        print(f"   {icon} {metric}: {value:.4f} (threshold: {threshold:.2f})")
        if not passed:
            all_pass = False
    if all_pass:
        print("   🎉 ALL CHECKS PASSED — CI GATE: GREEN")
    else:
        print("   🛑 SOME CHECKS FAILED — CI GATE: RED")
    print("=" * 65)
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--mode",     choices=["retriever", "e2e"], default="retriever",
                        help="retriever = fast CI mode | e2e = full pipeline")
    parser.add_argument("--dataset",  default="evaluation/eval_dataset.json")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--reranker", action="store_true",
                        help="Bật cross-encoder reranker (chỉ áp dụng cho mode=retriever)")
    parser.add_argument("--no-ci",   action="store_true",
                        help="Bỏ qua CI gate check")
    args = parser.parse_args()

    if args.mode == "retriever":
        result = run_retriever_mode(args.dataset, args.config, args.reranker)
        if not args.no_ci:
            passed = check_ci_thresholds(result)
            sys.exit(0 if passed else 1)  # Exit code 1 = CI fail
    else:
        run_retriever_mode(args.dataset, args.config, False)
        run_e2e_mode(args.dataset, args.config)
