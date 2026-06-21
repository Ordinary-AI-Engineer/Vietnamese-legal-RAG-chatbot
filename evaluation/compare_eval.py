"""
compare_eval.py — So sánh metrics trước/sau khi thêm kỹ thuật nâng cao.

Chạy:
    cd /Users/doanngocthanh/ChatBotLawFinal
    source .venv/bin/activate
    python evaluation/compare_eval.py
"""
import json
import os
import sys
import time
import unicodedata

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.retriever import LawRetriever
from langchain_core.documents import Document

DATASET_PATH = "evaluation/eval_dataset.json"
CONFIG_PATH  = "config.yaml"
K_MRR        = 10
K_RECALL     = 5


# ── Metric helpers ────────────────────────────────────────────────────────────

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()

def is_relevant(doc: Document, sample: dict) -> bool:
    src = nfc(os.path.basename(doc.metadata.get("source", "")))
    art = nfc(doc.metadata.get("article", ""))
    source_match = any(nfc(e) in src or src in nfc(e) for e in sample["relevant_sources"])
    if not source_match:
        return False
    if not sample.get("relevant_articles"):
        return True
    return any(nfc(e) in art for e in sample["relevant_articles"])

def compute_mrr(docs, sample, k=10):
    for rank, doc in enumerate(docs[:k], 1):
        if is_relevant(doc, sample):
            return 1.0 / rank
    return 0.0

def compute_recall(docs, sample, k=5):
    return float(any(is_relevant(d, sample) for d in docs[:k]))

def compute_precision(docs, sample, k=5):
    hits = sum(is_relevant(d, sample) for d in docs[:k])
    return hits / k


# ── Core eval function ────────────────────────────────────────────────────────

def run_mode(label: str, multi_query: bool, auto_merging: bool, reranker=None):
    print(f"\n{'='*65}")
    print(f"MODE: {label}")
    print(f"  multi_query={multi_query}  auto_merging={auto_merging}  reranker={reranker is not None}")
    print(f"{'='*65}")

    retriever = LawRetriever(config_path=CONFIG_PATH, reranker=reranker)
    # Override flags without touching config file
    retriever.multi_query_enabled  = multi_query
    retriever.auto_merging_enabled = auto_merging


    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    mrr_scores, recall_scores, precision_scores = [], [], []
    total_start = time.time()

    for i, sample in enumerate(dataset, 1):
        q = sample["question"]
        print(f"  [{i:02d}/{len(dataset)}] {q[:55]}...")

        t0   = time.time()
        docs = retriever.get_relevant_laws(q)
        elapsed = time.time() - t0

        mrr     = compute_mrr(docs, sample, k=K_MRR)
        recall  = compute_recall(docs, sample, k=K_RECALL)
        prec    = compute_precision(docs, sample, k=K_RECALL)

        mrr_scores.append(mrr)
        recall_scores.append(recall)
        precision_scores.append(prec)

        status = "✅" if recall > 0 else "❌"
        print(f"    {status} MRR={mrr:.3f} | Recall@5={recall:.0f} | {elapsed:.1f}s")

    total_time = time.time() - total_start
    avg_mrr  = sum(mrr_scores)  / len(mrr_scores)
    avg_rec  = sum(recall_scores) / len(recall_scores)
    avg_prec = sum(precision_scores) / len(precision_scores)

    print(f"\n  📈 {label}")
    print(f"     MRR@{K_MRR}:      {avg_mrr:.4f}")
    print(f"     Recall@{K_RECALL}:   {avg_rec:.4f}  ({sum(r>0 for r in recall_scores)}/{len(recall_scores)})")
    print(f"     Precision@{K_RECALL}: {avg_prec:.4f}")
    print(f"     Total time:  {total_time:.1f}s")

    return {
        "label":     label,
        "mrr":       round(avg_mrr, 4),
        "recall":    round(avg_rec, 4),
        "precision": round(avg_prec, 4),
        "time_s":    round(total_time, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.retrieval.reranker import CrossEncoderReranker

    results = []

    # Mode 1: Baseline (RRF only, no new features)
    results.append(run_mode(
        "Baseline — RRF + BM25 + Targeted",
        multi_query=False, auto_merging=False,
        reranker=None
    ))

    # Mode 2: + bge-reranker-v2-m3
    print("\n⏳ Loading bge-reranker-v2-m3 (first run ~568MB download)...")
    reranker_v2 = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", top_k=5)
    results.append(run_mode(
        "+ bge-reranker-v2-m3",
        multi_query=False, auto_merging=False,
        reranker=reranker_v2
    ))

    # Mode 3: Full pipeline (Multi-Query + Auto Merge + Reranker-v2)
    results.append(run_mode(
        "Full: MultiQuery + AutoMerge + Reranker-v2",
        multi_query=True, auto_merging=True,
        reranker=reranker_v2
    ))

    # ── Comparison table ──
    print("\n\n" + "="*72)
    print("📊 COMPARISON TABLE")
    print(f"{'Mode':<42} {'MRR@10':>8} {'Rec@5':>8} {'P@5':>8} {'Time':>8}")
    print("-"*72)

    baseline = results[0]
    for r in results:
        dmrr = r['mrr']    - baseline['mrr']
        drec = r['recall'] - baseline['recall']
        tag  = "" if r is baseline else f"  Δ MRR={dmrr:+.3f}"
        print(f"{r['label']:<42} {r['mrr']:>8.4f} {r['recall']:>8.4f} {r['precision']:>8.4f} {r['time_s']:>6.0f}s{tag}")

    print("="*72)

    out = "evaluation/reranker_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved to {out}")

