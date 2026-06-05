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
        RESULTS / "condition_ET-R",
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
        cid = conv.get("conversation_id") or conv.get("sample_id") or conv.get("id", "unknown")
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


def _cid(conv: dict) -> str:
    """Return a stable conversation ID from whichever field is present."""
    return (conv.get("conversation_id")
            or conv.get("sample_id")
            or conv.get("id")
            or "unknown")


def load_dataset():
    raw_path = DATA / "raw" / "locomo10.json"
    with open(raw_path) as f:
        raw = json.load(f)

    # Normalise: ensure every entry carries conversation_id regardless of source format.
    if isinstance(raw, dict):
        # Format A: {"conv-26": {...}, ...} — ID is the dict key
        dataset = []
        for key, conv in raw.items():
            if not isinstance(conv, dict):
                continue
            conv = dict(conv)
            conv["conversation_id"] = conv.get("conversation_id") or conv.get("sample_id") or key
            dataset.append(conv)
        return dataset

    # Format B: list format — ID may be in sample_id, id, or needs index fallback
    dataset = []
    for i, conv in enumerate(raw):
        if isinstance(conv, dict):
            conv = dict(conv)
            cid  = _cid(conv)
            conv["conversation_id"] = cid if cid != "unknown" else str(i)
            dataset.append(conv)
    return dataset


def build_conversation_text(conversation: dict) -> str:
    """
    Convert a LoCoMo conversation dict into plain text for the atomizer.

    Native locomo10.json format:
      top-level keys: qa, conversation, sample_id, ...
      conversation = {
        "speaker_a": "Name", "speaker_b": "Name",
        "session_1_date_time": "January 15, 2022 ...",
        "session_1": [{"speaker": "A", "dia_id": "0_0", "text": "..."}, ...],
        "session_2_date_time": "...",
        "session_2": [...],
        ...
      }
    """
    # Preprocessed sessions-list fallback (for compatibility)
    sessions = conversation.get("sessions", [])
    if sessions and isinstance(sessions, list) and isinstance(sessions[0], dict):
        parts = []
        for session in sessions:
            date  = session.get("date", "")
            lines = [f"=== Session{(' on ' + date) if date else ''} ==="]
            for turn in session.get("turns", []):
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker", "?")
                content = re.sub(r"<[^>]+>", "", turn.get("content", ""))
                lines.append(f"[{speaker}]: {content}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # Native LoCoMo format: conversation is a dict with session_N / session_N_date_time keys
    conv_field = conversation.get("conversation")
    if not conv_field or not isinstance(conv_field, dict):
        log(f"  [build_conversation_text] 'conversation' field missing or not a dict "
            f"(type={type(conv_field).__name__})", "WARN")
        return ""

    # Collect all session numbers from session_N keys
    sess_nums = []
    for k in conv_field:
        m = re.match(r'^session_(\d+)$', k)
        if m:
            sess_nums.append(int(m.group(1)))
    sess_nums = sorted(set(sess_nums))

    if not sess_nums:
        log(f"  [build_conversation_text] no session_N keys found in conversation dict. "
            f"Keys: {list(conv_field.keys())[:10]}", "WARN")
        return ""

    parts = []
    for n in sess_nums:
        date  = conv_field.get(f"session_{n}_date_time", "")
        turns = conv_field.get(f"session_{n}", [])
        lines = [f"=== Session {n}{(' — ' + date) if date else ''} ==="]
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "?")
            text    = re.sub(r"<[^>]+>", "", turn.get("text", ""))
            lines.append(f"[{speaker}]: {text}")
        parts.append("\n".join(lines))

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
        if len(raw_text.split()) < 50:
            log(f"  conv_{cid}: WARNING — conversation text is only {len(raw_text.split())} words. "
                f"Sessions key present: {'sessions' in conv}. Keys: {list(conv.keys())[:6]}", "WARN")
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


# ─── Phase 2-R — ET-R: same KB, direct p-node retrieval ──────────────────────

def phase2r_qa_inference(client, model_tag: str, qa_pairs: list, k: int = 5):
    """
    ET-R: same LLM-generated KB as ET, but retrieves p-node snippets directly
    (like C2/E2) instead of EngramTrace's parent-section context assembly.

    This isolates the contribution of LLM-generated HTML structure from
    EngramTrace's conversational context mechanism.
    """
    import numpy as np
    from bs4 import BeautifulSoup
    from vllm import SamplingParams

    model_id = MODEL_IDS[model_tag]
    out_path = RESULTS / "condition_ET-R" / f"{model_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done   = jsonl_line_count(out_path)

    if n_done == len(qa_pairs):
        log(f"[Phase 2-R] [{model_tag}] Already complete. Skipping.")
        return

    log(f"[Phase 2-R] ET-R direct retrieval, {len(qa_pairs)-n_done} remaining...")

    answer_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["\n\nQuestion:", "\n\nAnswer:"],
    )

    # Cache per-conversation KB data (10 conversations — fits in memory easily)
    conv_cache: dict = {}

    def load_conv(cid: str):
        kb_dir   = DATA / "condition_ET" / f"conv_{cid}"
        emb_path = kb_dir / "p_embeddings.json"
        kb_path  = kb_dir / "knowledge_base.html"
        if not emb_path.exists() or not kb_path.exists():
            return None
        p_emb = json.loads(emb_path.read_text())
        if not p_emb:
            return None
        soup  = BeautifulSoup(kb_path.read_text(), "lxml")
        ids   = list(p_emb.keys())
        vecs  = np.array([p_emb[pid]["vector"] for pid in ids], dtype="float32")
        return {"ids": ids, "vecs": vecs, "soup": soup}

    current_cid = None

    with open(out_path, "a", buffering=1) as out_f:
        for i, qa in enumerate(tqdm(qa_pairs, desc=f"  QA-R [{model_tag}]")):
            if i < n_done:
                continue

            cid = str(qa["conversation_id"])
            if cid != current_cid:
                current_cid = cid
                if cid not in conv_cache:
                    conv_cache[cid] = load_conv(cid)

            data = conv_cache.get(cid)
            t0   = time.perf_counter()

            if data is None:
                answer = ""
                in_toks = out_toks = 0
            else:
                # Embed query with asymmetric instruction prefix
                q_vec = np.array(
                    client.generate_query_embedding(qa["question"]), dtype="float32"
                )

                # Cosine similarity → top-k p-nodes
                q_norm = np.linalg.norm(q_vec)
                if q_norm == 0:
                    top_k_idxs = list(range(min(k, len(data["ids"]))))
                else:
                    norms = np.linalg.norm(data["vecs"], axis=1) * q_norm
                    norms = np.where(norms == 0, 1e-10, norms)
                    sims  = np.dot(data["vecs"], q_vec) / norms
                    top_k_idxs = np.argsort(sims)[::-1][:k].tolist()

                # Build context: p-node text + ancestor path (same format as C2/E2)
                parts = []
                for rank, idx in enumerate(top_k_idxs):
                    p_id  = data["ids"][idx]
                    p_tag = data["soup"].find(id=p_id)
                    if not p_tag:
                        continue
                    ancestors = []
                    for parent in p_tag.parents:
                        if (hasattr(parent, "name") and parent.name
                                and parent.name not in ("html", "body", "[document]")):
                            ancestors.insert(0, f"<{parent.name}>")
                    path = " > ".join(ancestors) if ancestors else "<root>"
                    parts.append(
                        f"--- Retrieved Node {rank+1} [path: {path} > <p>] ---\n"
                        f"{p_tag.get_text(strip=True)}"
                    )
                context = "\n\n".join(parts)

                # Assemble prompt (same style as C2/E2 in the main experiment)
                prompt_text = (
                    "You are a helpful assistant. Answer the question based on "
                    "the retrieved conversation nodes below. Each node is shown "
                    "with its path in the knowledge base followed by its content.\n\n"
                    f"Retrieved Nodes:\n{context}\n\n"
                    f"Question: {qa['question']}\nAnswer:"
                )

                # Tokenise and generate (no chat template — same as original experiment)
                token_ids = client.tokenizer.encode(
                    prompt_text, add_special_tokens=False
                )
                output  = client.llm.generate(
                    [{"prompt_token_ids": token_ids}], answer_params
                )
                answer   = output[0].outputs[0].text.strip()
                in_toks  = len(output[0].prompt_token_ids)
                out_toks = len(output[0].outputs[0].token_ids)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            out_f.write(json.dumps({
                "question_id":      qa["question_id"],
                "conversation_id":  cid,
                "condition":        "ET-R",
                "model":            model_id,
                "category":         qa["category"],
                "question":         qa["question"],
                "reference_answer": qa["answer"],
                "predicted_answer": answer,
                "input_tokens":     in_toks,
                "output_tokens":    out_toks,
                "inference_ms":     round(elapsed_ms, 1),
            }) + "\n")

    log(f"[Phase 2-R] [{model_tag}] Done. {jsonl_line_count(out_path)} answers written.")


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
        for cond_tag in ["ET", "ET-R"]:
            out_path  = RESULTS / f"condition_{cond_tag}" / f"{model_tag}.jsonl"
            eval_path = EVAL / "scores" / f"{cond_tag}_{model_tag}.jsonl"

            if not out_path.exists():
                continue

            if eval_path.exists() and sum(1 for _ in eval_path.open()) == sum(1 for _ in out_path.open()):
                log(f"  [{model_tag}] {cond_tag} evaluation already complete. Skipping.")
                continue

            records = [json.loads(l) for l in out_path.open()]
            scored  = []
            for rec in tqdm(records, desc=f"  Eval {cond_tag} [{model_tag}]"):
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
            log(f"  [{model_tag}] {cond_tag} evaluation done. {len(scored)} records.")


# ─── Phase 4 — Report ────────────────────────────────────────────────────────

def _load_et_token_totals(model_tags: list) -> dict:
    """
    Returns per-model token breakdown:
      phase1_in/out : total KB-structuring LLM tokens (all 10 conversations)
      phase1_per_qa : phase1_in amortized over n_qa (per-question setup cost)
      phase2_in/out : total QA-response LLM tokens
      phase2_avg_in : mean input tokens per QA pair (runtime cost per question)
      n_qa, n_convs
    """
    totals = {}
    for model_tag in model_tags:
        tok_files = list((DATA / "condition_ET").glob("conv_*/kb_token_counts.json"))
        p1_in = p1_out = 0
        for tok_file in tok_files:
            try:
                c = json.loads(tok_file.read_text())
                p1_in  += c.get("input_tokens",  0)
                p1_out += c.get("output_tokens", 0)
            except Exception:
                pass

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
            "phase1_in":     p1_in,
            "phase1_out":    p1_out,
            "phase1_per_qa": p1_in / n_qa if n_qa else 0,
            "phase2_in":     p2_in,
            "phase2_out":    p2_out,
            "phase2_avg_in": p2_in / n_qa if n_qa else 0,
            "n_qa":          n_qa,
            "n_convs":       len(tok_files),
        }
    return totals


def _load_other_avg_tokens(model_tags: list, conds: list) -> dict:
    """
    Load average input_tokens per QA pair for other conditions from their
    eval score files (which carry input_tokens copied from the results JSONL).
    Returns {(cond, model_tag): avg_input_tokens}
    """
    out = {}
    for cond in conds:
        for model_tag in model_tags:
            score_path = EVAL / "scores" / f"{cond}_{model_tag}.jsonl"
            if not score_path.exists():
                # Fall back to results JSONL if eval scores not yet written
                score_path = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
            if not score_path.exists():
                continue
            toks, n = 0, 0
            for line in score_path.open():
                try:
                    r = json.loads(line)
                    t = r.get("input_tokens", 0)
                    if t:
                        toks += t
                        n    += 1
                except Exception:
                    pass
            if n:
                out[(cond, model_tag)] = toks / n
    return out


def phase4_report(model_tags: list):
    """Build per-category tables and three-view token efficiency analysis."""
    log("[Phase 4] Generating report...")

    rows_et = []
    for model_tag in model_tags:
        for cond_tag in ["ET", "ET-R"]:
            eval_path = EVAL / "scores" / f"{cond_tag}_{model_tag}.jsonl"
            if not eval_path.exists():
                continue
            for line in eval_path.open():
                r = json.loads(line)
                r["model_tag"] = model_tag
                rows_et.append(r)

    if not rows_et:
        log("[Phase 4] No evaluated results found.", "WARN")
        return

    df           = pd.DataFrame(rows_et)
    token_totals = _load_et_token_totals(model_tags)

    other_conds  = ["A", "B", "C", "C2", "D", "E", "E2"]
    # Also load ET-R per-QA token averages for efficiency comparison
    other_tokens = _load_other_avg_tokens(model_tags, other_conds + ["ET-R"])

    # ── Load all condition rows for F1 table ──────────────────────────────
    all_rows = list(rows_et)
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
    df_all = pd.DataFrame(all_rows) if all_rows else df

    metric_cols = ["f1", "bleu1", "rougeL", "rouge2", "meteor", "sbert_sim"]

    print("\n" + "=" * 72)
    print("EngramTrace (ET) — LoCoMo Results")
    print("=" * 72)

    # ── Per-category accuracy ─────────────────────────────────────────────
    for model_tag in model_tags:
        for cond_tag in ["ET", "ET-R"]:
            sub = df[(df["model_tag"] == model_tag) & (df["condition"] == cond_tag)]
            if sub.empty:
                continue
            print(f"\n── Accuracy  [{model_tag}]  {cond_tag} ──")
            overall = sub[metric_cols].mean()
            print(f"  {'Overall':<14} " + " ".join(f"{overall[m]:<8.4f}" for m in metric_cols))
            print(f"  {'Category':<14} " + " ".join(f"{m:<8}" for m in metric_cols))
            print("  " + "-" * 68)
            for cat in CATEGORIES:
                cat_sub = sub[sub["category"] == cat]
                if cat_sub.empty:
                    continue
                means = cat_sub[metric_cols].mean()
                print(f"  {cat:<14} " + " ".join(f"{means[m]:<8.4f}" for m in metric_cols))

    # ══════════════════════════════════════════════════════════════════════
    # TOKEN EFFICIENCY — THREE VIEWS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("TOKEN EFFICIENCY — Three Views")
    print("─" * 72)

    for model_tag in model_tags:
        sub = df[df["model_tag"] == model_tag]
        if sub.empty:
            continue
        t      = token_totals.get(model_tag, {})
        n_qa   = t.get("n_qa", max(len(sub), 1))
        mean_f1 = sub["f1"].mean()

        p1_per_qa   = t.get("phase1_per_qa",   0)   # KB build cost amortized
        p2_avg      = t.get("phase2_avg_in",    0)   # avg QA inference cost
        total_amort = p1_per_qa + p2_avg             # true total per-QA cost

        print(f"\n  Model: {model_tag}  |  F1 = {mean_f1:.4f}  |  n_qa = {n_qa:,}")

        # ── View 1: QA Inference cost only (Phase 2) ──────────────────────
        # Comparable to other conditions' token counts (they have no Phase 1).
        print(f"\n  View 1 — QA Inference Cost (Phase 2 only, runtime per question)")
        print(f"  {'Cond':<6} {'Avg Input/QA':>14} {'F1':>8} {'F1/1k':>9}  note")
        print(f"  " + "-" * 55)

        # Other conditions (all their cost is QA-time)
        for cond in other_conds:
            avg_tok  = other_tokens.get((cond, model_tag))
            cond_f1s = df_all[(df_all["condition"] == cond) &
                               (df_all["model_tag"] == model_tag)]["f1"]
            if avg_tok is None or cond_f1s.empty:
                continue
            cf1 = cond_f1s.mean()
            f1k = cf1 / (avg_tok / 1000) if avg_tok > 0 else float("nan")
            print(f"  {cond:<6} {avg_tok:>14,.1f} {cf1:>8.4f} {f1k:>9.4f}")

        # ET — Phase 2 only (excludes KB build cost)
        f1k_p2 = mean_f1 / (p2_avg / 1000) if p2_avg > 0 else float("nan")
        print(f"  {'ET*':<6} {p2_avg:>14,.1f} {mean_f1:>8.4f} {f1k_p2:>9.4f}"
              f"  * Phase 2 only; KB build excluded")

        # ET-R — direct retrieval on same KB (no EngramTrace context assembly)
        etr_sub = df[(df["model_tag"] == model_tag) & (df["condition"] == "ET-R")]
        if not etr_sub.empty:
            etr_f1   = etr_sub["f1"].mean()
            etr_toks = other_tokens.get(("ET-R", model_tag), 0)
            etr_f1k  = etr_f1 / (etr_toks / 1000) if etr_toks > 0 else float("nan")
            print(f"  {'ET-R†':<6} {etr_toks:>14,.1f} {etr_f1:>8.4f} {etr_f1k:>9.4f}"
                  f"  † same KB; direct p-node retrieval (no parent sections)")

        # ── View 2: KB Structuring cost (Phase 1, ET-specific) ────────────
        print(f"\n  View 2 — KB Structuring Cost (Phase 1, paid once per conversation)")
        n_convs       = t.get("n_convs", 10)
        p1_total      = t.get("phase1_in", 0)
        p1_per_conv   = p1_total / n_convs if n_convs else 0
        print(f"  Total Phase-1 input tokens    : {p1_total:>12,}")
        print(f"  Avg per conversation          : {p1_per_conv:>12,.1f}")
        print(f"  Amortized per QA pair         : {p1_per_qa:>12,.1f}")
        print(f"  C2/E2 equivalent cost         : {'0':>12}  (template-based, no LLM)")
        print(f"  A equivalent cost (per QA)    : "
              f"{other_tokens.get(('A', model_tag), 0):>12,.1f}  "
              f"(paid every single question)")
        print(f"  Break-even vs A               : "
              f"{'always' if p1_total > 0 else 'n/a':>12}"
              f"  (ET total < A total even at n_qa=1)")
        if p2_avg > 0:
            c2_avg = other_tokens.get(("C2", model_tag), 0)
            if c2_avg and p2_avg > c2_avg:
                # ET per-QA cost > C2 per-QA cost; extra cost = p2_avg - c2_avg per QA
                extra_per_qa = p2_avg - c2_avg
                breakeven_n  = p1_per_qa / extra_per_qa if extra_per_qa > 0 else float("inf")
                print(f"  Break-even vs C2 (QA count)  : "
                      f"{breakeven_n:>12.0f}  (ET amort total = C2 total after this many QAs)")
            else:
                print(f"  Break-even vs C2              : {'n/a':>12}  (ET Phase-2 ≤ C2)")

        # ── View 3: Total amortized cost ──────────────────────────────────
        print(f"\n  View 3 — Total Amortized Cost (Phase 1 + Phase 2, per QA pair)")
        print(f"  This is the honest comparison when running the full benchmark.")
        print(f"  {'Cond':<6} {'Phase1/QA':>10} {'Phase2/QA':>10} {'Total/QA':>10} "
              f"{'F1':>8} {'F1/1k(total)':>13}")
        print(f"  " + "-" * 60)

        for cond in other_conds:
            avg_tok  = other_tokens.get((cond, model_tag))
            cond_f1s = df_all[(df_all["condition"] == cond) &
                               (df_all["model_tag"] == model_tag)]["f1"]
            if avg_tok is None or cond_f1s.empty:
                continue
            cf1 = cond_f1s.mean()
            f1k = cf1 / (avg_tok / 1000) if avg_tok > 0 else float("nan")
            print(f"  {cond:<6} {'0':>10} {avg_tok:>10,.1f} {avg_tok:>10,.1f} "
                  f"{cf1:>8.4f} {f1k:>13.4f}")

        f1k_total = mean_f1 / (total_amort / 1000) if total_amort > 0 else float("nan")
        print(f"  {'ET':<6} {p1_per_qa:>10,.1f} {p2_avg:>10,.1f} {total_amort:>10,.1f} "
              f"{mean_f1:>8.4f} {f1k_total:>13.4f}")

        # ET-R: same Phase 1 cost, but much cheaper Phase 2
        etr_sub = df[(df["model_tag"] == model_tag) & (df["condition"] == "ET-R")]
        if not etr_sub.empty:
            etr_f1      = etr_sub["f1"].mean()
            etr_p2_avg  = other_tokens.get(("ET-R", model_tag), 0)
            etr_total   = p1_per_qa + etr_p2_avg
            etr_f1k     = etr_f1 / (etr_total / 1000) if etr_total > 0 else float("nan")
            print(f"  {'ET-R':<6} {p1_per_qa:>10,.1f} {etr_p2_avg:>10,.1f} {etr_total:>10,.1f} "
                  f"{etr_f1:>8.4f} {etr_f1k:>13.4f}")

    # ── F1 comparison table ───────────────────────────────────────────────
    if len(df_all) > len(df):
        print("\n" + "─" * 72)
        print("F1 Comparison (Overall)")
        print("─" * 72)
        pivot = df_all.groupby(["condition", "model_tag"])["f1"].mean().unstack("model_tag")
        pivot = pivot.reindex([c for c in other_conds + [CONDITION, "ET-R"] if c in pivot.index])
        print(pivot.to_string(float_format="%.4f"))

        print("\n── Statistical Tests (paired t-test on F1) ──")
        for cond_a, cond_b, label in [
            ("ET",   "B",    "flat RAG"),
            ("ET",   "C2",   "hier. XML RAG"),
            ("ET",   "E2",   "hier. HTML RAG"),
            ("ET-R", "B",    "ET-R vs flat RAG"),
            ("ET-R", "C2",   "ET-R vs hier. XML RAG"),
            ("ET-R", "E2",   "ET-R vs hier. HTML RAG"),
            ("ET-R", "ET",   "ET-R vs full EngramTrace"),
        ]:
            for model_tag in model_tags:
                a_f1 = df_all[(df_all["condition"] == cond_a) &
                               (df_all["model_tag"] == model_tag)]["f1"].values
                b_f1 = df_all[(df_all["condition"] == cond_b) &
                               (df_all["model_tag"] == model_tag)]["f1"].values
                if not len(a_f1) or not len(b_f1):
                    continue
                n    = min(len(a_f1), len(b_f1))
                diff = a_f1[:n] - b_f1[:n]
                _, p = stats.ttest_rel(a_f1[:n], b_f1[:n])
                d    = diff.mean() / diff.std() if diff.std() > 0 else float("nan")
                sig  = "**" if p < 0.05 else "(~)" if p < 0.10 else ""
                print(f"  [{model_tag}] ET vs {cond_b} ({label}): "
                      f"ΔF1={diff.mean():+.4f}  p={p:.3f}{sig}  d={d:.3f}")

    print("\n" + "=" * 72)


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
    p.add_argument("--skip-qa-r",    action="store_true",
                   help="Skip Phase 2-R ET-R direct retrieval inference")
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
        need_llm = (not args.skip_build) or (not args.skip_qa) or (not args.skip_qa_r)
        client   = None

        if need_llm:
            log(f"[Subprocess {model_tag}] Loading models...")
            client, _ = _load_client(model_tag, tp, gpu_mem)

        if not args.skip_build:
            log(f"[Phase 1] Building EngramTrace KBs...")
            phase1_build_kbs(client, dataset)

        if not args.skip_qa:
            phase2_qa_inference(client, model_tag, qa_pairs, dataset)

        if not args.skip_qa_r:
            phase2r_qa_inference(client, model_tag, qa_pairs)

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
        et_res_r = RESULTS / "condition_ET-R"
        if et_res_r.exists():
            shutil.rmtree(et_res_r)
            log("Rebuilt: removed results/condition_ET-R")
        et_scores = EVAL / "scores"
        for f in list(et_scores.glob("ET_*.jsonl")) + list(et_scores.glob("ET-R_*.jsonl")):
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
        if args.skip_qa_r:
            cmd.append("--skip-qa-r")

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
