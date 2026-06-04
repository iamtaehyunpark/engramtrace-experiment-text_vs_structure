#!/usr/bin/env python3
"""
run_engramtrace.py — LoCoMo benchmark evaluation of the EngramTrace system.

Condition ET: LLM-generated HTML knowledge base + EngramTrace hierarchical
retrieval (semantic + keyword), drift detection, and stage consolidation.

Uses the same models and encoder as run_experiment.py so results are
directly comparable.  Condition order for efficiency:
    Phase 1 (KB build)  : one LLM + encoder call per conversation session
    Phase 2 (QA)        : one LLM call per QA pair + encoder for drift detection
    Phase 3 (metrics)   : CPU only
    Phase 4 (report)    : prints comparison table against existing results

Usage:
    python run_engramtrace.py --gpu h200x4 --models 72B 7B
    python run_engramtrace.py --gpu a100x8 --rebuild

Requires beautifulsoup4 and lxml in addition to the main experiment deps:
    pip install beautifulsoup4 lxml
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import gc
import json
import logging
import re
import subprocess as _sp
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ─── Directories ─────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.resolve()
DATA      = BASE / "data"
QUESTIONS = BASE / "questions"
RESULTS   = BASE / "results"
EVAL      = BASE / "evaluation"

# ─── Experiment constants ─────────────────────────────────────────────────────
MODEL_IDS = {
    "72B": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "7B":  "Qwen/Qwen2.5-7B-Instruct",
}
CONDITION       = "ET"
ENCODER_MODEL   = "BAAI/bge-base-en-v1.5"
CATEGORIES      = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
CATEGORY_MAP    = {1: "single_hop", 2: "multi_hop", 3: "temporal",
                   4: "open_domain", 5: "adversarial"}

# GPU profiles → (tensor_parallel_size, gpu_memory_utilization)
GPU_PROFILES = {
    "h200x4":  (4, 0.90),
    "a100x2":  (2, 0.95),
    "a100x4":  (4, 0.90),
    "a100x6":  (4, 0.90),   # 6 GPUs but tp=4 to satisfy divisibility
    "a100x8":  (8, 0.90),
    "a6000x4": (4, 0.90),
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ─── Directory scaffolding ────────────────────────────────────────────────────

def ensure_dirs():
    for d in [
        DATA / "condition_ET",
        QUESTIONS,
        RESULTS / "condition_ET",
        EVAL / "scores",
        EVAL / "tables",
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ─── LoCoMo data helpers ──────────────────────────────────────────────────────

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def download_data():
    qa_path   = QUESTIONS / "locomo_qa.jsonl"
    raw_path  = DATA / "raw" / "locomo10.json"
    (DATA / "raw").mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        log(f"Downloading LoCoMo dataset → {raw_path}")
        urllib.request.urlretrieve(LOCOMO_URL, raw_path)

    if qa_path.exists():
        return

    log("Extracting QA pairs from LoCoMo...")
    with open(raw_path) as f:
        raw = json.load(f)

    dataset = list(raw.values()) if isinstance(raw, dict) else raw

    # Normalise per-conversation QA
    qa_pairs = []
    for conv in dataset:
        cid = conv.get("conversation_id", conv.get("id", "unknown"))
        for qa in conv.get("qa", []):
            cat_raw = qa.get("type") or qa.get("category") or 1
            cat     = CATEGORY_MAP.get(int(cat_raw), "single_hop") if str(cat_raw).isdigit() else str(cat_raw)
            qa_pairs.append({
                "question_id":     qa.get("id", len(qa_pairs)),
                "conversation_id": cid,
                "question":        qa.get("question", ""),
                "answer":          qa.get("answer",   ""),
                "category":        cat,
                "evidence":        qa.get("evidence", []),
            })

    with open(qa_path, "w") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")
    log(f"Extracted {len(qa_pairs)} QA pairs → {qa_path}")


def load_qa_pairs():
    return [json.loads(l) for l in (QUESTIONS / "locomo_qa.jsonl").open()]


def load_dataset():
    raw_path = DATA / "raw" / "locomo10.json"
    with open(raw_path) as f:
        raw = json.load(f)
    return list(raw.values()) if isinstance(raw, dict) else raw


def build_session_text(session: dict) -> str:
    """Concatenates a single session's turns into plain text for the atomizer."""
    date    = session.get("date", "Unknown date")
    lines   = [f"=== Session on {date} ==="]
    for turn in session.get("turns", []):
        speaker = turn.get("speaker", "?")
        ts      = turn.get("timestamp", "")
        content = re.sub(r"<[^>]+>", "", turn.get("content", ""))
        lines.append(f"[{speaker}, {ts}]: {content}")
    return "\n".join(lines)


def build_conversation_text(conversation: dict) -> str:
    """Joins all sessions of a conversation into one text blob."""
    parts = []
    for session in conversation.get("sessions", []):
        parts.append(build_session_text(session))
    return "\n\n".join(parts)


# ─── Tensor-parallel validation ───────────────────────────────────────────────

def _valid_tp(model_tag: str, requested: int) -> int:
    num_heads  = {"72B": 64, "7B": 28}
    vocab_size = {"72B": 152064, "7B": 152064}
    heads = num_heads.get(model_tag, requested)
    vocab = vocab_size.get(model_tag, 152064)
    tp = requested
    while tp > 1 and (heads % tp != 0 or vocab % tp != 0):
        tp -= 1
    return tp


# ─── Phase 1 — Build EngramTrace KBs ─────────────────────────────────────────

def _load_client(model_tag: str, tensor_parallel_size: int,
                 gpu_memory_utilization: float):
    """Load vLLM + tokenizer + encoder once; return (client, tp)."""
    from vllm import LLM
    from transformers import AutoTokenizer
    from engramtrace.llm_client import LocalLLMClient

    model_id = MODEL_IDS[model_tag]
    tp       = _valid_tp(model_tag, tensor_parallel_size)

    log(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log(f"  Loading vLLM: {model_id} (tp={tp})")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        distributed_executor_backend="mp",
    )
    client = LocalLLMClient(
        llm=llm,
        tokenizer=tokenizer,
        encoder_model=ENCODER_MODEL,
        encoder_device="cpu",
        max_tokens_html=4096,
        max_tokens_answer=256,
    )
    return client, tp


def phase1_build_kbs(client, dataset: list):
    """
    For each conversation, feed the full text to the EngramTrace atomizer to
    produce an LLM-structured HTML knowledge base with hierarchical embeddings.
    Accepts a pre-loaded LocalLLMClient so vLLM is loaded only once per subprocess.
    Token usage is saved per-conversation to kb_dir/kb_token_counts.json for
    amortized cost accounting in the efficiency table.
    """
    from engramtrace.memory import MemoryManager

    skipped = 0
    for conv in tqdm(dataset, desc="  Building KBs"):
        cid        = conv.get("conversation_id", conv.get("id"))
        kb_dir     = DATA / "condition_ET" / f"conv_{cid}"
        kb_path    = kb_dir / "knowledge_base.html"
        tok_path   = kb_dir / "kb_token_counts.json"

        if kb_path.exists() and tok_path.exists():
            skipped += 1
            continue

        kb_dir.mkdir(parents=True, exist_ok=True)
        mem = MemoryManager(
            kb_path                    = str(kb_dir / "knowledge_base.html"),
            p_embeddings_path          = str(kb_dir / "p_embeddings.json"),
            structural_embeddings_path = str(kb_dir / "structural_embeddings.json"),
        )

        raw_text = build_conversation_text(conv)
        client.reset_token_counts()
        try:
            active_ids = mem.atomizer(client, raw_text=raw_text)
            counts = client.get_token_counts()
            tok_path.write_text(json.dumps(counts))
            log(f"  conv_{cid}: KB built — {len(active_ids)} p-nodes, "
                f"{counts['input_tokens']}→{counts['output_tokens']} tokens")
        except Exception as e:
            log(f"  conv_{cid}: ERROR during atomizer — {e}", "WARN")

    n_built = len(dataset) - skipped
    log(f"[Phase 1] Done. {n_built} built, {skipped} skipped.")


# ─── Phase 2 — QA inference ───────────────────────────────────────────────────

def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def phase2_qa_inference(client, model_tag: str, qa_pairs: list, dataset: list):
    """
    For each QA pair, load the pre-built KB for its conversation and call
    brain.run_inference() to get the answer.

    Design notes vs original EngramTrace usage:
    - no_memorize=True: answers don't write to the stage log (evaluation is read-only)
    - current_trace reset per QA: prevents accumulated hits from earlier questions
      contaminating context for later ones (keeps evaluation stateless like other conditions)
    - stage/session wipe per conversation: clean slate for each conversation's QA block
    - Accepts pre-loaded LocalLLMClient so vLLM is loaded only once per subprocess.
    """
    from engramtrace.memory import MemoryManager
    from engramtrace.brain import Brain

    model_id = MODEL_IDS[model_tag]
    out_path = RESULTS / "condition_ET" / f"{model_tag}.jsonl"
    n_done   = jsonl_line_count(out_path)
    if n_done == len(qa_pairs):
        log(f"[Phase 2] [{model_tag}] Already complete ({n_done} answers). Skipping.")
        return

    log(f"[Phase 2] QA inference with {model_id}, {len(qa_pairs)-n_done} remaining...")

    # Build a map from conversation_id → dataset dict for quick lookup
    conv_map  = {str(c.get("conversation_id", c.get("id"))): c for c in dataset}

    # Track which Brain objects are active (one per conversation, reset between convs)
    current_cid  = None
    brain        = None

    if n_done > 0:
        log(f"  Resuming from line {n_done}...")

    with open(out_path, "a", buffering=1) as out_f:  # line-buffered for crash safety
        for i, qa in enumerate(tqdm(qa_pairs, desc=f"  QA [{model_tag}]")):
            if i < n_done:
                continue

            cid = str(qa["conversation_id"])

            # Load/reset Brain when the conversation changes
            if cid != current_cid:
                current_cid = cid
                kb_dir      = DATA / "condition_ET" / f"conv_{cid}"

                if not (kb_dir / "knowledge_base.html").exists():
                    log(f"  WARNING: KB missing for conv_{cid} — skipping remaining QA for this conv", "WARN")
                    brain = None

                else:
                    (kb_dir / "sessions").mkdir(exist_ok=True)
                    mem = MemoryManager(
                        kb_path                    = str(kb_dir / "knowledge_base.html"),
                        p_embeddings_path          = str(kb_dir / "p_embeddings.json"),
                        structural_embeddings_path = str(kb_dir / "structural_embeddings.json"),
                    )
                    brain = Brain(
                        memory_manager   = mem,
                        llm_client       = client,
                        base_dir         = str(kb_dir),
                        stage_threshold  = 0.83,
                        search_threshold = 0.80,
                    )
                    # Wipe ephemeral state so evaluation QA pairs don't cross-contaminate
                    brain.engram_trace.wipe(wipe_stage=True, wipe_session=True, wipe_trace=True)

            t0 = time.perf_counter()
            if brain is None:
                answer        = ""
                in_tokens     = 0
                out_tokens    = 0
            else:
                # Reset current_trace before each QA so hits don't accumulate across
                # independent questions (with no_memorize=True the stage log stays empty,
                # making `if len(stage_log)==0` always True → keyword search always fires
                # and hits pile up on current_trace indefinitely without this reset).
                brain.engram_trace.current_trace = set()
                # Reset token counter so we capture only THIS QA pair's LLM usage
                # (includes any consolidation calls if drift triggers, plus the response).
                client.reset_token_counts()
                try:
                    answer = brain.run_inference(qa["question"], no_memorize=True)
                except Exception as e:
                    log(f"  conv_{cid} q{qa['question_id']}: inference error — {e}", "WARN")
                    answer = ""
                counts     = client.get_token_counts()
                in_tokens  = counts["input_tokens"]
                out_tokens = counts["output_tokens"]
            elapsed_ms = (time.perf_counter() - t0) * 1000

            out_f.write(json.dumps({
                "question_id":      qa["question_id"],
                "conversation_id":  cid,
                "condition":        CONDITION,
                "model":            model_id,
                "category":         qa["category"],
                "question":         qa["question"],
                "reference_answer": qa["answer"],
                "predicted_answer": answer,
                "input_tokens":     in_tokens,   # LLM tokens for this QA (response only normally;
                "output_tokens":    out_tokens,  # more if consolidation fires)
                "inference_ms":     round(elapsed_ms, 1),
            }) + "\n")

    log(f"[Phase 2] [{model_tag}] Done. {jsonl_line_count(out_path)} answers written.")


# ─── Phase 3 — Evaluation metrics ────────────────────────────────────────────

def compute_f1(pred: str, ref: str) -> float:
    pred_toks = pred.lower().split()
    ref_toks  = ref.lower().split()
    common    = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if not common:
        return 0.0
    p = common / len(pred_toks)
    r = common / len(ref_toks)
    return 2 * p * r / (p + r)


def phase3_evaluate(model_tags: list):
    """Compute F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT-sim for condition ET."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rouge_lib
    from sentence_transformers import SentenceTransformer, util as st_util

    log("[Phase 3] Computing evaluation metrics...")
    rouge  = rouge_lib.RougeScorer(["rougeL", "rouge2"], use_stemmer=True)
    sbert  = SentenceTransformer("all-MiniLM-L6-v2")
    smooth = SmoothingFunction().method1

    for model_tag in model_tags:
        out_path  = RESULTS / "condition_ET" / f"{model_tag}.jsonl"
        eval_path = EVAL / "scores" / f"ET_{model_tag}.jsonl"

        if not out_path.exists():
            log(f"  No results for {model_tag} — skipping.", "WARN")
            continue

        if eval_path.exists() and sum(1 for _ in eval_path.open()) == sum(1 for _ in out_path.open()):
            log(f"  [{model_tag}] Evaluation already complete. Skipping.")
            continue

        records = [json.loads(l) for l in out_path.open()]
        scored  = []
        for rec in tqdm(records, desc=f"  Eval [{model_tag}]"):
            pred = rec.get("predicted_answer", "")
            ref  = rec.get("reference_answer", "")
            r    = rouge.score(ref, pred)
            pe   = sbert.encode(pred, convert_to_tensor=True)
            re_  = sbert.encode(ref,  convert_to_tensor=True)
            entry = dict(rec)
            entry.update({
                "f1":        compute_f1(pred, ref),
                "bleu1":     sentence_bleu([ref.lower().split()], pred.lower().split(),
                                            weights=(1, 0, 0, 0), smoothing_function=smooth),
                "rougeL":    r["rougeL"].fmeasure,
                "rouge2":    r["rouge2"].fmeasure,
                "meteor":    meteor_score([ref.split()], pred.split()),
                "sbert_sim": float(st_util.cos_sim(pe, re_)),
            })
            scored.append(entry)

        with open(eval_path, "w") as f:
            for s in scored:
                f.write(json.dumps(s) + "\n")
        log(f"  [{model_tag}] Evaluation done. {len(scored)} records.")


# ─── Phase 4 — Report ────────────────────────────────────────────────────────

def _load_et_token_totals(model_tags: list) -> dict:
    """
    Sum Phase 1 (KB structuring) + Phase 2 (QA response) LLM token usage per model.
    Returns {model_tag: {"phase1_in": int, "phase1_out": int,
                          "phase2_in": int, "phase2_out": int, "n_qa": int}}
    """
    totals = {}
    for model_tag in model_tags:
        # Phase 1: per-conversation KB structuring tokens
        p1_in = p1_out = 0
        for tok_file in (DATA / "condition_ET").glob("conv_*/kb_token_counts.json"):
            try:
                c = json.loads(tok_file.read_text())
                p1_in  += c.get("input_tokens",  0)
                p1_out += c.get("output_tokens", 0)
            except Exception:
                pass

        # Phase 2: per-QA response tokens (from the results JSONL)
        p2_in = p2_out = n_qa = 0
        res_path = RESULTS / "condition_ET" / f"{model_tag}.jsonl"
        if res_path.exists():
            for line in res_path.open():
                try:
                    r       = json.loads(line)
                    p2_in  += r.get("input_tokens",  0)
                    p2_out += r.get("output_tokens", 0)
                    n_qa   += 1
                except Exception:
                    pass

        totals[model_tag] = {
            "phase1_in":  p1_in,
            "phase1_out": p1_out,
            "phase2_in":  p2_in,
            "phase2_out": p2_out,
            "n_qa":       n_qa,
        }
    return totals


def phase4_report(model_tags: list):
    """Build per-category and overall tables, token efficiency, and comparison."""
    log("[Phase 4] Generating report...")

    rows_et = []
    for model_tag in model_tags:
        eval_path = EVAL / "scores" / f"ET_{model_tag}.jsonl"
        if not eval_path.exists():
            continue
        for line in eval_path.open():
            r = json.loads(line)
            r["model_tag"] = model_tag
            rows_et.append(r)

    if not rows_et:
        log("[Phase 4] No evaluated results found.", "WARN")
        return

    df          = pd.DataFrame(rows_et)
    token_totals = _load_et_token_totals(model_tags)

    print("\n" + "=" * 70)
    print("EngramTrace (ET) Results — LoCoMo Benchmark")
    print("=" * 70)

    metric_cols = ["f1", "bleu1", "rougeL", "rouge2", "meteor", "sbert_sim"]

    for model_tag in model_tags:
        sub = df[df["model_tag"] == model_tag]
        if sub.empty:
            continue
        print(f"\n── Model: {model_tag} ──")
        overall = sub[metric_cols].mean()
        print(f"  Overall:   " + "  ".join(f"{m}={overall[m]:.4f}" for m in metric_cols))

        print(f"\n  {'Category':<14} " + " ".join(f"{m:<8}" for m in metric_cols))
        for cat in CATEGORIES:
            cat_sub = sub[sub["category"] == cat]
            if cat_sub.empty:
                continue
            means = cat_sub[metric_cols].mean()
            print(f"  {cat:<14} " + " ".join(f"{means[m]:<8.4f}" for m in metric_cols))

    # ── Token efficiency (all LLM usage: KB build + QA response) ─────────
    print("\n── Token Efficiency (ALL LLM usage: KB structuring + QA response) ──")
    print(f"  {'Model':<8} {'Phase1-in':>10} {'Phase1-out':>11} "
          f"{'Phase2-in':>10} {'Phase2-out':>11} "
          f"{'Amort/QA':>9} {'F1':>7} {'F1/1k':>8}")
    print("  " + "-" * 72)

    for model_tag in model_tags:
        sub = df[df["model_tag"] == model_tag]
        if sub.empty:
            continue
        t    = token_totals.get(model_tag, {})
        n_qa = t.get("n_qa", max(len(sub), 1))
        # Total input tokens (the "cost" side — input drives compute cost at inference)
        total_in  = t.get("phase1_in", 0) + t.get("phase2_in", 0)
        total_out = t.get("phase1_out", 0) + t.get("phase2_out", 0)
        # Amortized input cost per QA pair (total ÷ #QA pairs)
        amort_in  = total_in / n_qa if n_qa else 0
        mean_f1   = sub["f1"].mean()
        f1_per_1k = mean_f1 / (amort_in / 1000) if amort_in > 0 else float("nan")
        print(f"  {model_tag:<8} {t.get('phase1_in',0):>10,} {t.get('phase1_out',0):>11,} "
              f"{t.get('phase2_in',0):>10,} {t.get('phase2_out',0):>11,} "
              f"{amort_in:>9.1f} {mean_f1:>7.4f} {f1_per_1k:>8.4f}")

    print()
    print("  Notes:")
    print("  - Phase1-in : input tokens for LLM-based KB HTML structuring (amortized over all QA pairs)")
    print("  - Phase2-in : input tokens for QA response generation (includes any consolidation calls)")
    print("  - Amort/QA  : (Phase1-in + Phase2-in) / n_qa  — total per-query LLM cost")
    print("  - F1/1k     : F1 per 1k amortized input tokens  (higher = more efficient)")
    print("  - Compare with other conditions in RESULTS.md Section 3 (encoder-only cost not included there)")

    # ── F1 comparison table against other conditions ───────────────────────
    other_conds = ["A", "B", "C", "C2", "D", "E", "E2"]
    all_rows    = list(rows_et)

    for cond in other_conds:
        for model_tag in model_tags:
            score_path = EVAL / "scores" / f"{cond}_{model_tag}.jsonl"
            if not score_path.exists():
                continue
            for line in score_path.open():
                r = json.loads(line)
                r["model_tag"] = model_tag
                r["condition"]  = cond
                all_rows.append(r)

    if len(all_rows) > len(rows_et):
        df_all = pd.DataFrame(all_rows)
        print("\n── F1 Comparison Table (Overall) ──")
        pivot = df_all.groupby(["condition", "model_tag"])["f1"].mean().unstack("model_tag")
        cond_order = other_conds + [CONDITION]
        pivot = pivot.reindex([c for c in cond_order if c in pivot.index])
        print(pivot.to_string(float_format="%.4f"))

        # ET vs B (flat RAG) and ET vs C2/E2 (hierarchical RAG)
        print("\n── Statistical Tests (paired t-test on F1) ──")
        comparisons = [("ET", "B", "flat RAG"), ("ET", "C2", "hier. XML RAG"), ("ET", "E2", "hier. HTML RAG")]
        for cond_a, cond_b, label in comparisons:
            for model_tag in model_tags:
                a_scores = df_all[(df_all["condition"] == cond_a) & (df_all["model_tag"] == model_tag)]["f1"].values
                b_scores = df_all[(df_all["condition"] == cond_b) & (df_all["model_tag"] == model_tag)]["f1"].values
                if len(a_scores) == 0 or len(b_scores) == 0:
                    continue
                n    = min(len(a_scores), len(b_scores))
                diff = a_scores[:n] - b_scores[:n]
                t, p = stats.ttest_rel(a_scores[:n], b_scores[:n])
                d    = diff.mean() / diff.std() if diff.std() > 0 else float("nan")
                sig  = "**" if p < 0.05 else "(~)" if p < 0.10 else ""
                print(f"  [{model_tag}] ET vs {cond_b} ({label}): "
                      f"ΔF1={diff.mean():+.4f}  p={p:.3f}{sig}  d={d:.3f}")

    print("\n" + "=" * 70)


# ─── Subprocess isolation ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EngramTrace LoCoMo evaluation")
    p.add_argument("--gpu",          default="h200x4",
                   choices=list(GPU_PROFILES),
                   help="Default GPU profile for all models")
    p.add_argument("--gpu-72b",      default=None,
                   choices=list(GPU_PROFILES),
                   help="GPU profile for 72B model (overrides --gpu). e.g. a100x8")
    p.add_argument("--gpu-7b",       default=None,
                   choices=list(GPU_PROFILES),
                   help="GPU profile for 7B model (overrides --gpu). e.g. a100x4")
    p.add_argument("--models",       nargs="+", default=["72B", "7B"],
                   choices=["72B", "7B"],
                   help="Which model(s) to run")
    p.add_argument("--rebuild",      action="store_true",
                   help="Delete all ET data and rebuild from scratch")
    p.add_argument("--skip-build",   action="store_true",
                   help="Skip Phase 1 KB building (use existing KBs)")
    p.add_argument("--skip-qa",      action="store_true",
                   help="Skip Phase 2 QA inference (re-evaluate existing answers)")
    p.add_argument("--inference-only", action="store_true",
                   help="INTERNAL: run one model's phases 1+2 then exit (subprocess mode)")
    return p.parse_args()


def _model_gpu(args, model_tag: str) -> str:
    """Return the GPU profile for a specific model, respecting per-model overrides."""
    # argparse lowercases flag names: --gpu-72b → args.gpu_72b (not gpu_72B)
    attr     = f"gpu_{model_tag.replace('-', '_').lower()}"
    override = getattr(args, attr, None)
    return override if override else args.gpu


def main():
    args = parse_args()

    # ── Subprocess mode: Phases 1+2 for one model then exit ─────────────────
    if args.inference_only:
        assert len(args.models) == 1, "--inference-only requires exactly one --models argument"
        model_tag = args.models[0]
        gpu_profile = _model_gpu(args, model_tag)
        tp, gpu_mem = GPU_PROFILES[gpu_profile]

        download_data()
        ensure_dirs()
        dataset  = load_dataset()
        qa_pairs = load_qa_pairs()

        # Load vLLM ONCE for both phases — avoids CUDA worker leak between loads.
        # (Phase 1 and Phase 2 share the same LocalLLMClient instance.)
        need_llm = (not args.skip_build) or (not args.skip_qa)
        client   = None

        if need_llm:
            log(f"[Subprocess {model_tag}] Loading models...")
            client, _ = _load_client(model_tag, tp, gpu_mem)

        if not args.skip_build:
            log(f"[Phase 1] Building EngramTrace KBs...")
            phase1_build_kbs(client, dataset)

        if not args.skip_qa:
            phase2_qa_inference(client, model_tag, qa_pairs, dataset)

        return

    # ── Orchestrator: spawn one subprocess per model ─────────────────────────
    download_data()
    ensure_dirs()

    if args.rebuild:
        import shutil
        et_dir = DATA / "condition_ET"
        if et_dir.exists():
            shutil.rmtree(et_dir)
            log("Rebuilt: removed data/condition_ET")
        et_res = RESULTS / "condition_ET"
        if et_res.exists():
            shutil.rmtree(et_res)
            log("Rebuilt: removed results/condition_ET")
        et_scores = EVAL / "scores"
        for f in et_scores.glob("ET_*.jsonl"):
            f.unlink()
            log(f"Rebuilt: removed {f.name}")
        ensure_dirs()

    this_script = str(Path(__file__).resolve())

    for model_tag in args.models:
        gpu_profile = _model_gpu(args, model_tag)
        log(f"[Orchestrator] Launching subprocess for {model_tag} on {gpu_profile}...")
        cmd = [
            sys.executable, this_script,
            "--gpu",    gpu_profile,
            "--models", model_tag,
            "--inference-only",
        ]
        if args.skip_build:
            cmd.append("--skip-build")
        if args.skip_qa:
            cmd.append("--skip-qa")

        ret = _sp.run(cmd, check=False)
        if ret.returncode != 0:
            log(f"[Orchestrator] Subprocess for {model_tag} exited with code {ret.returncode}.", "ERROR")
            sys.exit(ret.returncode)
        log(f"[Orchestrator] {model_tag} subprocess complete.")

    # ── Phase 3 & 4: metrics and report (CPU, runs in orchestrator process) ──
    phase3_evaluate(args.models)
    phase4_report(args.models)


if __name__ == "__main__":
    main()
