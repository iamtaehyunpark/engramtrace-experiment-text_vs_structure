#!/usr/bin/env python3
"""
run_slm_retrieval.py — SLM-as-retriever experiment.

Two-stage pipeline:
  Stage 1 (retrieval): 7B model reads the full context + question and returns
                       relevant sentences/paragraphs verbatim. It never answers.
  Stage 2 (inference): LLM (72B or 7B) answers using the retrieved passages.

Conditions:
  F1 — Plain text     → 7B retrieval → 72B inference
  F2 — Plain text     → 7B retrieval → 7B  inference
  F3 — XML context    → 7B retrieval → 72B inference
  F4 — XML context    → 7B retrieval → 7B  inference
  F5 — Enriched text  → 7B retrieval → 72B inference
  F6 — Enriched text  → 7B retrieval → 7B  inference

Enriched text embeds session/speaker/timestamp metadata inline as natural
language (no XML tags or escaping), giving the SLM retriever richer context
while keeping prose intact.

Usage:
    python run_slm_retrieval.py --gpu h200x4
    python run_slm_retrieval.py --gpu h200x1 --models 7B --conditions F5 F6
"""

import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

BASE      = Path(__file__).parent.resolve()
DATA      = BASE / "data"
QUESTIONS = BASE / "questions"
RESULTS   = BASE / "results"
EVAL      = BASE / "evaluation"

MODEL_IDS = {
    "72B": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "7B":  "Qwen/Qwen2.5-7B-Instruct",
}

GPU_PROFILES = {
    "h200x1":  (1, 0.90),
    "h200x2":  (2, 0.90),
    "h200x4":  (4, 0.90),
    "a100x2":  (2, 0.95),
    "a100x4":  (4, 0.90),
    "a100x8":  (8, 0.90),
    "a6000x4": (4, 0.90),
}

# Retrieval source for each condition.
RETRIEVAL_SOURCE = {
    "F1": "linear",
    "F2": "linear",
    "F3": "xml",
    "F4": "xml",
    "F5": "enriched",
    "F6": "enriched",
}
# Inference model for each condition.
INFERENCE_MODEL = {
    "F1": "72B",
    "F2": "7B",
    "F3": "72B",
    "F4": "7B",
    "F5": "72B",
    "F6": "7B",
}


def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def ensure_dirs():
    for cond in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        (RESULTS / f"condition_{cond}").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)  # for slm_retrieved_*.jsonl
    (DATA / "condition_enriched").mkdir(parents=True, exist_ok=True)
    (EVAL / "scores").mkdir(parents=True, exist_ok=True)
    (EVAL / "tables").mkdir(parents=True, exist_ok=True)


def _valid_tp(model_tag: str, requested: int) -> int:
    num_heads  = {"72B": 64, "7B": 28}
    vocab_size = {"72B": 152064, "7B": 152064}
    heads = num_heads.get(model_tag, requested)
    vocab = vocab_size.get(model_tag, 152064)
    tp = requested
    while tp > 1 and (heads % tp != 0 or vocab % tp != 0):
        tp -= 1
    return tp


def _teardown_vllm(llm):
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    time.sleep(30)
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


# ─── Representations ──────────────────────────────────────────────────────────

import re as _re


def load_locomo_dataset() -> dict:
    """
    Load locomo10.json and normalize into {conv_id: {conversation_id, sessions}}.
    Raw format: list of {sample_id, conversation: {session_1: [...], session_1_date_time: ..., ...}}
    """
    raw_path = DATA / "raw" / "locomo10.json"
    raw = json.loads(raw_path.read_text())
    result = {}
    for idx, item in enumerate(raw):
        conv    = item["conversation"]
        conv_id = str(item.get("sample_id", idx))
        sessions = []
        s = 1
        while f"session_{s}" in conv:
            date_str = conv.get(f"session_{s}_date_time", f"Session {s}")
            turns = [
                {
                    "speaker":   t["speaker"],
                    "timestamp": t.get("dia_id", ""),
                    "content":   t["text"],
                }
                for t in conv[f"session_{s}"]
            ]
            sessions.append({"date": date_str, "turns": turns})
            s += 1
        result[conv_id] = {"conversation_id": conv_id, "sessions": sessions}
    return result


def build_enriched_text(conv: dict) -> str:
    """
    Metadata-rich plain text: session header + one line per turn.
    Speaker metadata (date, participants, turn count, timestamp) preserved
    inline without any XML tags or escaping.

    Example:
        [Session 1 | May 7 2023 | Alice & Bob | 12 turns]
        Alice (0_3): I went to the gym today.
        Bob (0_4): How was it?
    """
    lines = []
    for s_idx, session in enumerate(conv["sessions"]):
        speakers = sorted({t["speaker"] for t in session["turns"]})
        date = session.get("date", "unknown date")
        n_turns = len(session["turns"])
        header = (
            f"[Session {s_idx + 1} | {date} | "
            f"{' & '.join(speakers)} | {n_turns} turns]"
        )
        lines.append(header)
        for turn in session["turns"]:
            content = _re.sub(r"<[^>]+>", "", turn["content"])
            ts = turn.get("timestamp", "")
            label = f"{turn['speaker']} ({ts})" if ts else turn["speaker"]
            lines.append(f"{label}: {content}")
        lines.append("")
    return "\n".join(lines)


def build_enriched_representations(dataset: dict):
    """Write enriched plain-text files to data/condition_enriched/{cid}.txt."""
    out_dir = DATA / "condition_enriched"
    for cid, conv in tqdm(dataset.items(), desc="  Build enriched reps", leave=False):
        path = out_dir / f"{cid}.txt"
        if not path.exists():
            path.write_text(build_enriched_text(conv))
    log(f"  Enriched representations ready in {out_dir}")


def load_representations(conv_ids: list, sources: set) -> dict:
    """Load only the representation sources actually needed by the requested conditions."""
    log(f"Loading representations {sources}...")
    reps = {}
    for cid in tqdm(conv_ids, desc="  Load reps", leave=False):
        reps[cid] = {}
        if "linear" in sources:
            reps[cid]["linear"] = (DATA / "condition_A" / f"{cid}.txt").read_text()
        if "xml" in sources:
            reps[cid]["xml"] = (DATA / "condition_C" / f"{cid}.xml").read_text()
        if "enriched" in sources:
            reps[cid]["enriched"] = (DATA / "condition_enriched" / f"{cid}.txt").read_text()
    return reps


# ─── Prompts ──────────────────────────────────────────────────────────────────

RETRIEVAL_PROMPT_TEXT = """\
You are a document search assistant. Read the conversation below and the question carefully.
Your task is to find and return all relevant sentences and paragraphs from the conversation \
that could help answer the question.

Rules:
- Return passages VERBATIM (word for word from the conversation).
- Include complete sentences and enough surrounding context.
- Do NOT answer the question. Only retrieve relevant passages.
- If multiple parts of the conversation are relevant, return all of them.

Conversation:
{context}

Question: {question}

Relevant passages from the conversation:"""

RETRIEVAL_PROMPT_XML = """\
You are a document search assistant. Read the structured conversation record below \
and the question carefully.
Your task is to find and return all relevant sentences and paragraphs from the conversation \
that could help answer the question.

Rules:
- Return passages VERBATIM (word for word from the conversation content, not the XML tags).
- Include complete sentences and enough surrounding context (speaker, date if helpful).
- Do NOT answer the question. Only retrieve relevant passages.
- If multiple parts of the conversation are relevant, return all of them.

Conversation record (XML):
{context}

Question: {question}

Relevant passages from the conversation:"""

INFERENCE_PROMPT = """\
You are a helpful assistant. Answer the question based on the relevant passages \
retrieved from the conversation below.

Retrieved Passages:
{retrieved}

Question: {question}
Answer:"""


def make_retrieval_prompt(source: str, context: str, question: str) -> str:
    if source == "xml":
        return RETRIEVAL_PROMPT_XML.format(context=context, question=question)
    return RETRIEVAL_PROMPT_TEXT.format(context=context, question=question)


def make_inference_prompt(retrieved: str, question: str) -> str:
    return INFERENCE_PROMPT.format(retrieved=retrieved, question=question)


# ─── Stage 1: SLM Retrieval ───────────────────────────────────────────────────

def run_retrieval_stage(qa_pairs: list, reps: dict,
                        llm, tokenizer,
                        source: str,
                        out_path: Path,
                        max_input_tokens: int):
    """
    Run one retrieval pass (plain-text or XML source) with the 7B model.
    Saves a JSONL with question_id + retrieved_passages for each QA pair.
    Resume-safe.
    """
    n_done = jsonl_count(out_path)
    if n_done == len(qa_pairs):
        log(f"  Retrieval ({source}): already complete, skipping")
        return

    from vllm import SamplingParams
    retrieval_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["\n\nQuestion:", "\n\nAnswer:"],
    )

    log(f"  Retrieval ({source}): assembling {len(qa_pairs) - n_done} prompts...")
    records_to_run = qa_pairs[n_done:]
    prompts = []
    for qa in tqdm(records_to_run, desc=f"  Prompts ({source})", leave=False):
        ctx = reps[qa["conversation_id"]][source]
        raw = make_retrieval_prompt(source, ctx, qa["question"])
        ids = tokenizer.encode(raw, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[:max_input_tokens]
        prompts.append({"prompt_token_ids": ids})

    log(f"  Retrieval ({source}): running batch ({len(prompts)} prompts)...")
    outputs = llm.generate(prompts, retrieval_params)

    with out_path.open("a") as f:
        for qa, output in zip(records_to_run, outputs):
            f.write(json.dumps({
                "question_id":      qa["question_id"],
                "conversation_id":  qa["conversation_id"],
                "question":         qa["question"],
                "reference_answer": qa["answer"],
                "category":         qa["category"],
                "retrieved_passages": output.outputs[0].text.strip(),
                "retrieval_tokens":   len(output.outputs[0].token_ids),
            }) + "\n")

    log(f"  Retrieval ({source}): done — {jsonl_count(out_path)} records saved")


# ─── Stage 2: Inference ───────────────────────────────────────────────────────

def run_inference_stage(condition: str, model_tag: str,
                        retrieved_path: Path,
                        llm, tokenizer,
                        max_input_tokens: int):
    """
    Run inference over the SLM-retrieved passages for a given condition.
    Resume-safe.
    """
    out_path = RESULTS / f"condition_{condition}" / f"{model_tag}.jsonl"
    model_id = MODEL_IDS[model_tag]

    retrieved = [json.loads(l) for l in retrieved_path.open()]
    n_done = jsonl_count(out_path)
    if n_done == len(retrieved):
        log(f"  Inference {condition}/{model_tag}: already complete, skipping")
        return

    from vllm import SamplingParams
    inference_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["\n\nQuestion:", "\n\nAnswer:"],
    )

    log(f"  Inference {condition}/{model_tag}: assembling {len(retrieved) - n_done} prompts...")
    records_to_run = retrieved[n_done:]
    prompts = []
    input_token_counts = []
    for rec in tqdm(records_to_run, desc=f"  Prompts {condition}", leave=False):
        raw = make_inference_prompt(rec["retrieved_passages"], rec["question"])
        ids = tokenizer.encode(raw, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[:max_input_tokens]
        prompts.append({"prompt_token_ids": ids})
        input_token_counts.append(len(ids))

    log(f"  Inference {condition}/{model_tag}: running batch ({len(prompts)} prompts)...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, inference_params)
    ms_per = (time.perf_counter() - t0) / len(outputs) * 1000

    with out_path.open("a") as f:
        for rec, output, in_toks in zip(records_to_run, outputs, input_token_counts):
            f.write(json.dumps({
                "question_id":      rec["question_id"],
                "conversation_id":  rec["conversation_id"],
                "condition":        condition,
                "model":            model_id,
                "category":         rec["category"],
                "question":         rec["question"],
                "reference_answer": rec["reference_answer"],
                "predicted_answer": output.outputs[0].text.strip(),
                "input_tokens":     in_toks,
                "output_tokens":    len(output.outputs[0].token_ids),
                "inference_time_ms": round(ms_per, 2),
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "f1": None, "bleu1": None, "rougeL": None,
                "rouge2": None, "meteor": None, "sbert_sim": None,
            }) + "\n")

    log(f"  Inference {condition}/{model_tag}: done — {jsonl_count(out_path)} records "
        f"({ms_per:.1f} ms/query)")


# ─── Subprocess entry point ───────────────────────────────────────────────────

def run_subprocess(model_tag: str, tp: int, gpu_mem: float, conditions: list):
    """
    Called in subprocess mode. Loads one model, runs all assigned work, exits.

    7B subprocess: retrieval for both sources + inference for F2/F4.
    72B subprocess: inference for F1/F3 (retrieval already done by 7B subprocess).
    """
    from vllm import LLM

    model_id = MODEL_IDS[model_tag]
    tp = _valid_tp(model_tag, tp)

    log(f"[{model_tag}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log(f"[{model_tag}] Loading vLLM (tp={tp})...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=False,
        distributed_executor_backend="mp",
    )
    max_input_tokens = llm.llm_engine.model_config.max_model_len - 1024

    qa_pairs = [json.loads(l) for l in (QUESTIONS / "locomo_qa.jsonl").open()]
    conv_ids = sorted({qa["conversation_id"] for qa in qa_pairs})

    # Determine which retrieval sources are actually needed.
    needed_sources = {RETRIEVAL_SOURCE[c] for c in conditions}

    # Build enriched representations before loading if needed.
    if "enriched" in needed_sources:
        dataset = load_locomo_dataset()
        build_enriched_representations(dataset)

    reps = load_representations(conv_ids, needed_sources)

    # ── 7B subprocess: retrieval + SLM-only inference ─────────────────────
    if model_tag == "7B":
        for source in needed_sources:
            out = RESULTS / f"slm_retrieved_{source}.jsonl"
            run_retrieval_stage(qa_pairs, reps, llm, tokenizer,
                                source, out, max_input_tokens)

        for cond in [c for c in conditions if INFERENCE_MODEL[c] == "7B"]:
            source = RETRIEVAL_SOURCE[cond]
            retrieved_path = RESULTS / f"slm_retrieved_{source}.jsonl"
            run_inference_stage(cond, "7B", retrieved_path, llm, tokenizer,
                                max_input_tokens)

    # ── 72B subprocess: inference only (retrieval files already exist) ────
    elif model_tag == "72B":
        for cond in [c for c in conditions if INFERENCE_MODEL[c] == "72B"]:
            source = RETRIEVAL_SOURCE[cond]
            retrieved_path = RESULTS / f"slm_retrieved_{source}.jsonl"
            if not retrieved_path.exists():
                log(f"  [{cond}] Retrieval file missing: {retrieved_path} — skipping", "WARN")
                continue
            run_inference_stage(cond, "72B", retrieved_path, llm, tokenizer,
                                max_input_tokens)

    log(f"[{model_tag}] Tearing down vLLM...")
    _teardown_vllm(llm)
    log(f"[{model_tag}] Done. Process exiting.")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def run_evaluation(conditions: list, models: list):
    from collections import Counter
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rouge_lib
    from sentence_transformers import SentenceTransformer, util as st_util

    log("Running evaluation metrics...")
    rouge = rouge_lib.RougeScorer(["rougeL", "rouge2"], use_stemmer=True)
    sbert = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    smooth = SmoothingFunction().method1

    def f1(pred, ref):
        pt, rt = pred.lower().split(), ref.lower().split()
        common = sum((Counter(pt) & Counter(rt)).values())
        if not common:
            return 0.0
        p, r = common / len(pt), common / len(rt)
        return 2 * p * r / (p + r)

    for cond in conditions:
        for model_tag in models:
            model_id = MODEL_IDS[model_tag]
            path = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
            if not path.exists():
                continue
            records = [json.loads(l) for l in path.open()]
            if all(r.get("f1") is not None for r in records):
                log(f"  {cond}/{model_tag}: already scored")
                continue
            log(f"  Scoring {cond}/{model_tag} ({len(records)} records)...")
            for r in tqdm(records, desc=f"  {cond}/{model_tag}", leave=False):
                if r.get("f1") is not None:
                    continue
                pred, ref = r["predicted_answer"], r["reference_answer"]
                rg = rouge.score(ref, pred)
                pe = sbert.encode(pred, convert_to_tensor=True)
                re_ = sbert.encode(ref, convert_to_tensor=True)
                r.update({
                    "f1":        f1(pred, ref),
                    "bleu1":     sentence_bleu([ref.lower().split()], pred.lower().split(),
                                               weights=(1,0,0,0), smoothing_function=smooth),
                    "rougeL":    rg["rougeL"].fmeasure,
                    "rouge2":    rg["rouge2"].fmeasure,
                    "meteor":    meteor_score([ref.split()], pred.split()),
                    "sbert_sim": float(st_util.cos_sim(pe, re_)),
                })
            with path.open("w") as f_:
                for r in records:
                    f_.write(json.dumps(r) + "\n")
            log(f"  {cond}/{model_tag}: scored")

    del sbert
    gc.collect()


# ─── Judge ────────────────────────────────────────────────────────────────────

def _parse_verdict(text: str) -> int:
    t = text.strip().lower()
    if "incorrect" in t and "correct" not in t.replace("incorrect", ""):
        return 0
    return 1 if "correct" in t else 0


def run_judge(conditions: list, models: list,
              tensor_parallel_size: int = 1,
              gpu_memory_utilization: float = 0.90):
    from vllm import LLM, SamplingParams

    any_needed = any(
        (RESULTS / f"condition_{c}" / f"{m}.jsonl").exists()
        and any(r.get("llm_judge") is None
                for r in [json.loads(l) for l in
                           (RESULTS / f"condition_{c}" / f"{m}.jsonl").open()])
        for c in conditions for m in models
        if (RESULTS / f"condition_{c}" / f"{m}.jsonl").exists()
    )
    if not any_needed:
        log("[Judge] All records already judged, skipping.")
        return

    tp = _valid_tp("7B", tensor_parallel_size)
    log(f"[Judge] Loading Qwen2.5-7B judge (tp={tp})...")
    judge_llm = LLM(
        model=MODEL_IDS["7B"],
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        distributed_executor_backend="mp",
    )
    judge_params = SamplingParams(temperature=0.0, max_tokens=10)

    for cond in conditions:
        for model_tag in models:
            path = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
            if not path.exists():
                continue
            records = [json.loads(l) for l in path.open()]
            if all(r.get("llm_judge") is not None for r in records):
                log(f"  [Judge] {cond}/{model_tag}: already judged, skipping")
                continue

            pending = [(i, r) for i, r in enumerate(records) if r.get("llm_judge") is None]
            log(f"  [Judge] {cond}/{model_tag}: judging {len(pending)} records...")

            prompts = [
                "You are evaluating whether a predicted answer correctly answers a question.\n\n"
                f"Question: {r['question']}\n"
                f"Reference Answer: {r['reference_answer']}\n"
                f"Predicted Answer: {r['predicted_answer']}\n\n"
                "Does the predicted answer correctly answer the question? "
                "It is correct if it captures the key information, even if worded differently.\n"
                "Reply with exactly one word: Correct or Incorrect."
                for _, r in pending
            ]
            outputs = judge_llm.generate(prompts, judge_params)
            for (i, _), out in zip(pending, outputs):
                records[i]["llm_judge"] = _parse_verdict(out.outputs[0].text)

            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            n_correct = sum(r.get("llm_judge", 0) for r in records)
            log(f"  [Judge] {cond}/{model_tag}: {n_correct}/{len(records)} correct "
                f"({100*n_correct/len(records):.1f}%)")

    _teardown_vllm(judge_llm)
    log("[Judge] Done.")


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(conditions: list, models: list):
    import pandas as pd

    rows = []
    for cond in conditions:
        for model_tag in models:
            path = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
            if not path.exists():
                continue
            for l in path.open():
                r = json.loads(l)
                r["model_tag"] = model_tag
                rows.append(r)

    if not rows:
        log("No results to report.", "WARN")
        return

    df = pd.DataFrame(rows)
    metric_cols = ["f1", "bleu1", "rougeL", "rouge2", "meteor", "sbert_sim"]
    has_judge = "llm_judge" in df.columns and df["llm_judge"].notna().any()

    print("\n" + "=" * 72)
    print("SLM Retrieval Experiment — Results")
    print("=" * 72)

    cond_labels = {
        "F1": "Text→SLM-retrieve→LLM",
        "F2": "Text→SLM-retrieve→SLM",
        "F3": "XML→SLM-retrieve→LLM",
        "F4": "XML→SLM-retrieve→SLM",
        "F5": "Enriched→SLM-retrieve→LLM",
        "F6": "Enriched→SLM-retrieve→SLM",
    }

    for model_tag in models:
        print(f"\n── Model: {MODEL_IDS[model_tag]} ──")
        cols = metric_cols + (["llm_judge"] if has_judge else [])
        print(f"  {'Cond':<8} {'Description':<30} " +
              " ".join(f"{m:<10}" for m in cols))
        print("  " + "-" * 80)
        for cond in conditions:
            sub = df[(df["condition"] == cond) & (df["model_tag"] == model_tag)]
            if sub.empty:
                continue
            label = cond_labels.get(cond, cond)
            vals = " ".join(f"{sub[m].mean():<10.4f}" for m in cols)
            print(f"  {cond:<8} {label:<30} {vals}")

    # Judge accuracy summary
    if has_judge:
        print("\n── LLM Judge Accuracy ──")
        print(f"  {'Cond':<8} {'Description':<30} " +
              " ".join(f"{m:<8}" for m in models))
        print("  " + "-" * 56)
        for cond in conditions:
            label = cond_labels.get(cond, cond)
            row = f"  {cond:<8} {label:<30} "
            for model_tag in models:
                sub = df[(df["condition"] == cond) & (df["model_tag"] == model_tag)]
                if sub.empty or sub["llm_judge"].isna().all():
                    row += f"{'—':<8} "
                else:
                    row += f"{sub['llm_judge'].mean():<8.4f} "
            print(row)

    # Per-category F1
    print("\n── F1 by category ──")
    categories = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
    for model_tag in models:
        print(f"\n  {MODEL_IDS[model_tag]}")
        header = f"  {'Cond':<8} " + " ".join(f"{c:<14}" for c in categories)
        print(header)
        print("  " + "-" * (8 + 15 * len(categories)))
        for cond in conditions:
            sub = df[(df["condition"] == cond) & (df["model_tag"] == model_tag)]
            if sub.empty:
                continue
            row = f"  {cond:<8} "
            for cat in categories:
                cat_sub = sub[sub["category"] == cat]
                row += f"{cat_sub['f1'].mean() if not cat_sub.empty else float('nan'):<14.4f} "
            print(row)

    # Token efficiency
    print("\n── Token Efficiency ──")
    print(f"  {'Cond':<8} {'Model':<6} {'Avg Input Tokens':>18} {'F1':>8} {'F1/1k':>9}")
    print("  " + "-" * 54)
    for cond in conditions:
        for model_tag in models:
            sub = df[(df["condition"] == cond) & (df["model_tag"] == model_tag)]
            if sub.empty:
                continue
            avg_toks = sub["input_tokens"].mean()
            mean_f1 = sub["f1"].mean()
            f1k = mean_f1 / (avg_toks / 1000) if avg_toks > 0 else float("nan")
            print(f"  {cond:<8} {model_tag:<6} {avg_toks:>18,.1f} {mean_f1:>8.4f} {f1k:>9.4f}")

    print("\n" + "=" * 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SLM retrieval + LLM inference experiment")
    p.add_argument("--gpu", default="h200x4", choices=list(GPU_PROFILES))
    p.add_argument("--models", nargs="+", default=["72B", "7B"], choices=["72B", "7B"])
    p.add_argument("--conditions", nargs="+", default=["F1", "F2", "F3", "F4"],
                   choices=["F1", "F2", "F3", "F4", "F5", "F6"])
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-tp", type=int, default=1,
                   help="Tensor-parallel size for the judge model (default: 1)")
    p.add_argument("--inference-only", action="store_true",
                   help="INTERNAL: subprocess mode — one model then exit")
    return p.parse_args()


def main():
    args = parse_args()
    tp, gpu_mem = GPU_PROFILES[args.gpu]
    conditions = args.conditions

    ensure_dirs()

    this_script = str(Path(__file__).resolve())

    if args.inference_only:
        # Subprocess mode: one model, then exit to free GPU memory.
        assert len(args.models) == 1
        run_subprocess(args.models[0], tp, gpu_mem, conditions)
        return

    # ── Orchestrator: 7B first (retrieval + SLM inference), then 72B ─────
    # 7B must run before 72B so retrieval files exist when 72B needs them.
    ordered_models = []
    if "7B" in args.models:
        ordered_models.append("7B")
    if "72B" in args.models:
        ordered_models.append("72B")

    for model_tag in ordered_models:
        log(f"[Orchestrator] Launching subprocess for {model_tag}...")
        cmd = [
            sys.executable, this_script,
            "--gpu", args.gpu,
            "--models", model_tag,
            "--conditions", *conditions,
            "--inference-only",
        ]
        import subprocess as _sp
        ret = _sp.run(cmd, check=False)
        if ret.returncode != 0:
            log(f"[Orchestrator] Subprocess {model_tag} failed (exit {ret.returncode})", "ERROR")
            sys.exit(ret.returncode)
        log(f"[Orchestrator] {model_tag} subprocess complete.")

    # ── Evaluation ────────────────────────────────────────────────────────
    if not args.skip_eval:
        run_evaluation(conditions, args.models)

    # ── Judge ─────────────────────────────────────────────────────────────
    if not args.skip_judge:
        run_judge(conditions, args.models,
                  tensor_parallel_size=args.judge_tp,
                  gpu_memory_utilization=gpu_mem)

    # ── Report ────────────────────────────────────────────────────────────
    print_report(conditions, args.models)


if __name__ == "__main__":
    main()
