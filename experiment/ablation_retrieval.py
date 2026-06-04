#!/usr/bin/env python3
"""
ablation_retrieval.py — Grid search over α (hierarchical blending) and k (top-k)
for conditions C2 and E2, measuring retrieval recall against LoCoMo evidence.

Does NOT require GPU. Runs on CPU only.

Usage:
    python ablation_retrieval.py [--data-dir data] [--output ablation_results.csv]

Results are saved as a CSV with columns:
    condition, alpha, k, recall@k, precision@k, f1@k, n_questions
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    ENCODER_MODEL,
    QUERY_INSTRUCTION,
    extract_nodes,
    extract_html_nodes,
    compute_hierarchical_embeddings,
)

BASE      = Path(__file__).parent.resolve()
DATA      = BASE / "data"
QUESTIONS = BASE / "questions"

ALPHA_GRID = [0.3, 0.5, 0.7, 0.9]
K_GRID     = [3, 5, 10]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output",   default="ablation_results.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

def extract_evidence_texts(qa: dict) -> list:
    """
    Return a list of evidence text strings from the QA evidence field.
    LoCoMo evidence items may be dicts with a 'text' key, or plain strings.
    """
    texts = []
    for ev in qa.get("evidence", []):
        if isinstance(ev, dict):
            t = ev.get("text") or ev.get("content") or ""
            if t:
                texts.append(str(t))
        elif isinstance(ev, str) and ev.strip():
            texts.append(ev.strip())
    return texts


def word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def evidence_recall(retrieved_texts: list, evidence_texts: list,
                    threshold: float = 0.5) -> float:
    """Fraction of evidence items covered by at least one retrieved node."""
    if not evidence_texts:
        return None  # unevaluable
    covered = sum(
        1 for ev in evidence_texts
        if any(word_overlap(ev, ret) >= threshold for ret in retrieved_texts)
    )
    return covered / len(evidence_texts)


def evidence_precision(retrieved_texts: list, evidence_texts: list,
                        threshold: float = 0.5) -> float:
    """Fraction of retrieved nodes that match at least one evidence item."""
    if not retrieved_texts or not evidence_texts:
        return None
    relevant = sum(
        1 for ret in retrieved_texts
        if any(word_overlap(ev, ret) >= threshold for ev in evidence_texts)
    )
    return relevant / len(retrieved_texts)


# ---------------------------------------------------------------------------
# Index building with variable alpha
# ---------------------------------------------------------------------------

def build_index_for_alpha(nodes: list, encoder, alpha: float) -> faiss.Index:
    texts      = [n["text_content"] if n["text_content"] else n["tag"] for n in nodes]
    local_embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    hier_embs  = np.zeros_like(local_embs)
    for node in nodes:
        nid, pid = node["node_id"], node["parent_id"]
        hier_embs[nid] = (local_embs[nid] if pid == -1
                          else alpha * local_embs[nid] + (1 - alpha) * hier_embs[pid])
    norms = np.linalg.norm(hier_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embs  = (hier_embs / norms).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve(query: str, nodes: list, index: faiss.Index, encoder, k: int) -> list:
    q = encoder.encode([QUERY_INSTRUCTION + query],
                       normalize_embeddings=True).astype("float32")
    _, idxs = index.search(q, k)
    return [nodes[i]["text_content"] for i in idxs[0] if nodes[i]["text_content"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    data_dir = BASE / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)

    # Load QA pairs with evidence
    qa_path = QUESTIONS / "locomo_qa.jsonl"
    if not qa_path.exists():
        print(f"ERROR: {qa_path} not found. Run Phase 1 first.")
        sys.exit(1)
    qa_pairs = [json.loads(l) for l in qa_path.open()]
    # keep only QA pairs that have non-empty evidence
    qa_with_ev = [qa for qa in qa_pairs if extract_evidence_texts(qa)]
    print(f"QA pairs with evidence: {len(qa_with_ev)}/{len(qa_pairs)}")
    if not qa_with_ev:
        print("WARNING: no evaluable evidence found. Check LoCoMo evidence field format.")
        print("Running recall estimation via top-answer-overlap proxy instead.")
        qa_with_ev = qa_pairs  # fall back: use reference answer as proxy evidence

    conv_ids = sorted({qa["conversation_id"] for qa in qa_with_ev})

    print(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL, device="cpu")

    conditions = {
        "C2": ("condition_C2/nodes", "xml", extract_nodes),
        "E2": ("condition_E2/nodes", "html", extract_html_nodes),
    }

    rows = []
    for cond_name, (node_subdir, fmt, extractor) in conditions.items():
        node_dir = data_dir / node_subdir
        if not node_dir.exists():
            print(f"  Skipping {cond_name}: {node_dir} not found")
            continue
        print(f"\n{'='*50}")
        print(f"Condition {cond_name}  ({fmt.upper()} nodes)")
        print(f"{'='*50}")

        for alpha in ALPHA_GRID:
            print(f"\n  α={alpha}")

            # Build per-conversation indices for this alpha
            conv_nodes  = {}
            conv_indices = {}
            for cid in tqdm(conv_ids, desc=f"  Building α={alpha} indices", leave=False):
                node_path = node_dir / f"{cid}.json"
                if not node_path.exists():
                    continue
                nodes = json.loads(node_path.read_text())
                index = build_index_for_alpha(nodes, encoder, alpha)
                conv_nodes[cid]   = nodes
                conv_indices[cid] = index

            for k in K_GRID:
                recalls, precisions = [], []
                for qa in qa_with_ev:
                    cid = qa["conversation_id"]
                    if cid not in conv_indices:
                        continue
                    ev_texts  = extract_evidence_texts(qa)
                    if not ev_texts:
                        # proxy: use reference answer
                        ev_texts = [qa["answer"]]
                    ret_texts = retrieve(qa["question"], conv_nodes[cid],
                                         conv_indices[cid], encoder, k)
                    r = evidence_recall(ret_texts, ev_texts)
                    p = evidence_precision(ret_texts, ev_texts)
                    if r is not None:
                        recalls.append(r)
                    if p is not None:
                        precisions.append(p)

                mean_r = float(np.mean(recalls))   if recalls    else 0.0
                mean_p = float(np.mean(precisions)) if precisions else 0.0
                f1_rp  = (2 * mean_p * mean_r / (mean_p + mean_r)
                           if (mean_p + mean_r) > 0 else 0.0)

                print(f"    k={k:2d}  recall={mean_r:.4f}  precision={mean_p:.4f}  F1={f1_rp:.4f}  "
                      f"(n={len(recalls)})")
                rows.append({
                    "condition":    cond_name,
                    "alpha":        alpha,
                    "k":            k,
                    "recall_at_k":  round(mean_r, 4),
                    "precision_at_k": round(mean_p, 4),
                    "f1_at_k":      round(f1_rp, 4),
                    "n_questions":  len(recalls),
                })

    # Save results
    out_path = BASE / args.output
    import csv
    fieldnames = ["condition", "alpha", "k", "recall_at_k",
                  "precision_at_k", "f1_at_k", "n_questions"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {out_path}")

    # Print summary table
    print("\n── Ablation Summary ────────────────────────────────────────")
    print(f"{'Cond':<5} {'α':<5} {'k':<4} {'Recall':<9} {'Precision':<11} {'F1':<7}")
    print("-" * 45)
    for r in rows:
        print(f"{r['condition']:<5} {r['alpha']:<5} {r['k']:<4} "
              f"{r['recall_at_k']:<9.4f} {r['precision_at_k']:<11.4f} {r['f1_at_k']:<7.4f}")


if __name__ == "__main__":
    main()
