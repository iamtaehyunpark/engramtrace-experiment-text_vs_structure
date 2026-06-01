#!/usr/bin/env python3
"""
run_experiment.py — EngramTrace Concept Verification Experiment
Complete pipeline: data prep → 72B inference → 7B inference → evaluation → report.

Run once on an NVIDIA H200:
    python run_experiment.py

All outputs land in ./  (relative to this file). Resume-safe: every phase
checks whether its output already exists and skips if complete.

Install first:
    pip install vllm transformers datasets sentence-transformers faiss-gpu \\
                rouge-score nltk scipy pandas tqdm
    python -m nltk.downloader punkt wordnet averaged_perceptron_tagger
"""

# ─── stdlib ─────────────────────────────────────────────────────────────────
import gc
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

# ─── third-party (must be installed before running) ─────────────────────────
import faiss
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rouge_scorer_lib
from scipy import stats
from sentence_transformers import SentenceTransformer, util as st_util
from tqdm import tqdm
from transformers import AutoTokenizer

# ─── Directories ────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.resolve()
DATA      = BASE / "data"
QUESTIONS = BASE / "questions"
RESULTS   = BASE / "results"
EVAL      = BASE / "evaluation"

# ─── Experiment constants ────────────────────────────────────────────────────
MODEL_IDS = {
    "72B": "Qwen/Qwen2.5-72B-Instruct",
    "7B":  "Qwen/Qwen2.5-7B-Instruct",
}
CONDITION_ORDER = ["D", "B", "C2", "C", "A"]   # shortest → longest prompt
CONDITIONS      = ["A", "B", "C", "C2", "D"]
MODELS          = ["72B", "7B"]
CATEGORIES      = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
ALPHA           = 0.7   # hierarchical embedding blending coefficient
K               = 5     # retrieval top-k

# Integer category labels used in locomo10.json → string names
CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}

KEY_COMPARISONS = [
    ("A",  "C",  "C  vs A  (H1 — structure vs full linear)"),
    ("B",  "C2", "C2 vs B  (H2 — hierarchical vs flat RAG)"),
    ("A",  "C2", "C2 vs A  (efficiency — retrieval vs full context)"),
    ("C",  "C2", "C2 vs C  (retrieval over XML vs full XML)"),
    ("A",  "B",  "B  vs A  (flat RAG vs full context, lit replication)"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Directory scaffolding
# ═══════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    for d in [
        DATA / "raw", DATA / "condition_A",
        DATA / "condition_B" / "chunks", DATA / "condition_B" / "embeddings",
        DATA / "condition_C",
        DATA / "condition_C2" / "nodes", DATA / "condition_C2" / "embeddings",
        QUESTIONS,
        RESULTS / "condition_A", RESULTS / "condition_B",
        RESULTS / "condition_C", RESULTS / "condition_C2", RESULTS / "condition_D",
        EVAL / "scores", EVAL / "tables",
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── Condition A  (Linear text) ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_linear_text(conv: dict) -> str:
    lines = []
    for session in conv["sessions"]:
        date = session.get("date", "Unknown date")
        lines.append(f"--- Session: {date} ---")
        for turn in session["turns"]:
            content = re.sub(r"<[^>]+>", "", turn["content"])
            lines.append(f"[{turn['speaker']}, {turn.get('timestamp','')}]: {content}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ── Condition B  (Chunked RAG) ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_chunks(conv: dict) -> list:
    chunks = []
    for s_idx, session in enumerate(conv["sessions"]):
        for t_idx, turn in enumerate(session["turns"]):
            chunks.append({
                "text":        f"[{turn['speaker']}, {session['date']}, Session {s_idx+1}]\n{turn['content']}",
                "speaker":     turn["speaker"],
                "session_idx": s_idx,
                "turn_idx":    t_idx,
                "date":        session["date"],
                "timestamp":   turn.get("timestamp", ""),
            })
    return chunks


def build_faiss_flat_index(chunks: list, encoder) -> faiss.Index:
    embs  = encoder.encode([c["text"] for c in chunks],
                           normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve_chunks(query: str, chunks: list, index: faiss.Index, encoder) -> list:
    q = encoder.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = index.search(q, K)
    hits = [chunks[i] for i in idxs[0]]
    hits.sort(key=lambda x: (x["session_idx"], x["turn_idx"]))
    return hits


# ═══════════════════════════════════════════════════════════════════════════
# ── Condition C  (Structured XML) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def escape_xml(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def build_xml(conv: dict) -> str:
    import xml.etree.ElementTree as ET   # stdlib
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<conversation>"]
    for s_idx, session in enumerate(conv["sessions"]):
        date = session.get("date", f"session-{s_idx+1}")
        parts.append(f'  <session id="{s_idx+1}" date="{escape_xml(str(date))}">')
        for turn in session["turns"]:
            speaker = escape_xml(turn["speaker"])
            ts      = turn.get("timestamp", "")
            content = re.sub(r"<[^>]+>", "", turn["content"])
            parts.append(f'    <turn speaker="{speaker}" timestamp="{ts}">')
            for sent in re.split(r"(?<=[.!?])\s+", content.strip()):
                if sent.strip():
                    parts.append(f"      <utterance>{escape_xml(sent.strip())}</utterance>")
            parts.append("    </turn>")
        parts.append("  </session>")
    parts.append("</conversation>")
    return "\n".join(parts)


def validate_xml(conv: dict, xml_str: str) -> bool:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        log(f"  XML parse error: {e}", "ERROR")
        return False
    # Normalize all whitespace (incl. \n) to single spaces before comparing,
    # because build_xml splits sentences and rejoins with spaces.
    all_text = " ".join(
        " ".join((u.text or "").split()) for u in root.iter("utterance")
    )
    for session in conv["sessions"]:
        for turn in session["turns"]:
            raw = re.sub(r"<[^>]+>", "", turn["content"])
            key = " ".join(raw.split())[:30]   # normalize whitespace in key too
            if key and key not in all_text:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# ── Condition C2  (Hierarchical XML Retrieval) ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def extract_nodes(xml_str: str) -> list:
    import xml.etree.ElementTree as ET
    root  = ET.fromstring(xml_str)
    nodes = []

    def recurse(el, parent_id: int, depth: int, path: list):
        node_id     = len(nodes)
        direct_text = (el.text or "").strip()
        for child in el:
            if child.tail:
                direct_text += " " + child.tail.strip()
        attribs    = " ".join(f'{k}="{v}"' for k, v in el.attrib.items())
        path_entry = f"<{el.tag} {attribs}>".strip()
        nodes.append({
            "node_id":      node_id,
            "tag":          el.tag,
            "text_content": direct_text,
            "full_path":    path + [path_entry],
            "depth":        depth,
            "parent_id":    parent_id,
            "xml_snippet":  ET.tostring(el, encoding="unicode"),
        })
        for child in el:
            recurse(child, node_id, depth + 1, path + [path_entry])

    recurse(root, -1, 0, [])
    return nodes


def compute_hierarchical_embeddings(nodes: list, encoder) -> np.ndarray:
    texts      = [n["text_content"] if n["text_content"] else n["tag"] for n in nodes]
    local_embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    hier_embs  = np.zeros_like(local_embs)
    for node in nodes:
        nid, pid = node["node_id"], node["parent_id"]
        hier_embs[nid] = (local_embs[nid] if pid == -1
                          else ALPHA * local_embs[nid] + (1 - ALPHA) * hier_embs[pid])
    norms = np.linalg.norm(hier_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (hier_embs / norms).astype("float32")


def build_c2_index(nodes: list, encoder) -> faiss.Index:
    hier_embs = compute_hierarchical_embeddings(nodes, encoder)
    index = faiss.IndexFlatIP(hier_embs.shape[1])
    index.add(hier_embs)
    return index


def retrieve_nodes_hierarchical(query: str, nodes: list, index: faiss.Index, encoder) -> list:
    q = encoder.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = index.search(q, K)
    results = []
    for idx in idxs[0]:
        node, ancestors = nodes[idx], []
        pid = node["parent_id"]
        while pid != -1:
            ancestors.insert(0, nodes[pid])
            pid = nodes[pid]["parent_id"]
        results.append({"node": node, "ancestors": ancestors})
    return results


def format_c2_context(retrieved: list) -> str:
    parts = []
    for i, item in enumerate(retrieved):
        node = item["node"]
        path = " > ".join(f"<{a['tag']}>" for a in item["ancestors"]) if item["ancestors"] else "<root>"
        parts.append(
            f"--- Retrieved Node {i+1} [path: {path} > <{node['tag']}>] ---\n"
            f"{node['xml_snippet']}"
        )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# ── Load and normalize LoCoMo ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_locomo() -> list:
    """
    Download locomo10.json from GitHub (cached to data/raw/) and normalize
    into the internal format used by all builders:
      conversation_id, sessions[{date, turns[{speaker, timestamp, content}]}],
      qa_pairs[{id, question, answer, category, evidence}]
    """
    cache = DATA / "raw" / "locomo10.json"
    if not cache.exists():
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        log(f"  Downloading LoCoMo from GitHub...")
        urllib.request.urlretrieve(url, cache)
        log(f"  Saved to {cache}")
    else:
        log(f"  Using cached LoCoMo: {cache}")

    raw = json.loads(cache.read_text())

    conversations = []
    for idx, item in enumerate(raw):
        conv     = item["conversation"]
        conv_id  = str(item.get("sample_id", idx))

        # Sessions are stored as session_1, session_2, ... keys
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

        # QA pairs — filter unanswerable, map integer category to string
        qa_pairs = []
        for q_idx, qa in enumerate(item.get("qa", [])):
            answer = qa.get("answer")          # missing key = treat as unanswerable
            if answer is None:
                continue
            answer = str(answer).strip()
            if answer.lower() == "unanswerable":
                continue
            cat_int = qa.get("category", 1)
            qa_pairs.append({
                "id":       f"{conv_id}_q{q_idx}",
                "question": qa["question"],
                "answer":   answer,
                "category": CATEGORY_MAP.get(cat_int, f"cat_{cat_int}"),
                "evidence": qa.get("evidence", []),
            })

        conversations.append({
            "conversation_id": conv_id,
            "sessions":        sessions,
            "qa_pairs":        qa_pairs,
        })

    total_qa = sum(len(c["qa_pairs"]) for c in conversations)
    log(f"  {len(conversations)} conversations, {total_qa} answerable QA pairs loaded")
    return conversations


# ═══════════════════════════════════════════════════════════════════════════
# ── Phase 1: Build all representations ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def phase1_build(dataset: list):
    log("═" * 60)
    log("PHASE 1 — Building representations")
    log("═" * 60)

    # QA file — already filtered (unanswerable removed) during load_locomo()
    qa_path = QUESTIONS / "locomo_qa.jsonl"
    if not qa_path.exists():
        records = []
        for conv in dataset:
            for qa in conv["qa_pairs"]:
                records.append({
                    "question_id":     qa["id"],
                    "conversation_id": conv["conversation_id"],
                    "category":        qa["category"],
                    "question":        qa["question"],
                    "answer":          qa["answer"],
                    "evidence":        qa.get("evidence", []),
                })
        with qa_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        log(f"  QA file: {len(records)} pairs saved")
    else:
        log(f"  QA file: exists, skipping")

    # ── Condition A ──────────────────────────────────────────────────────
    log("  Building Condition A (linear text)...")
    skipped = 0
    for conv in tqdm(dataset, desc="  Cond A", leave=False):
        p = DATA / "condition_A" / f"{conv['conversation_id']}.txt"
        if p.exists():
            skipped += 1
        else:
            p.write_text(build_linear_text(conv))
    log(f"  Condition A: {len(dataset)-skipped} built, {skipped} skipped")

    # ── Condition C ──────────────────────────────────────────────────────
    log("  Building Condition C (XML + validation)...")
    failures, skipped = [], 0
    for conv in tqdm(dataset, desc="  Cond C", leave=False):
        p = DATA / "condition_C" / f"{conv['conversation_id']}.xml"
        if p.exists():
            skipped += 1
            continue
        xml_str = build_xml(conv)
        if not validate_xml(conv, xml_str):
            failures.append(conv["conversation_id"])
            log(f"  VALIDATION FAILED: {conv['conversation_id']}", "ERROR")
        else:
            p.write_text(xml_str)
    if failures:
        log(f"XML validation failed for: {failures}", "ERROR")
        sys.exit(1)
    log(f"  Condition C: {len(dataset)-skipped} built, {skipped} skipped — all validated")

    # ── Encoder for B and C2 ──────────────────────────────────────────────
    log("  Loading sentence encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # ── Condition B ──────────────────────────────────────────────────────
    log("  Building Condition B (chunks + FAISS)...")
    skipped = 0
    for conv in tqdm(dataset, desc="  Cond B", leave=False):
        cid  = conv["conversation_id"]
        cp   = DATA / "condition_B" / "chunks"    / f"{cid}.json"
        ip   = DATA / "condition_B" / "embeddings" / f"{cid}.index"
        if cp.exists() and ip.exists():
            skipped += 1
            continue
        chunks = build_chunks(conv)
        index  = build_faiss_flat_index(chunks, encoder)
        cp.write_text(json.dumps(chunks))
        faiss.write_index(index, str(ip))
    log(f"  Condition B: {len(dataset)-skipped} built, {skipped} skipped")

    # ── Condition C2 ─────────────────────────────────────────────────────
    log("  Building Condition C2 (hierarchical nodes + FAISS)...")
    skipped = 0
    for conv in tqdm(dataset, desc="  Cond C2", leave=False):
        cid  = conv["conversation_id"]
        np_  = DATA / "condition_C2" / "nodes"      / f"{cid}.json"
        ip   = DATA / "condition_C2" / "embeddings"  / f"{cid}.index"
        if np_.exists() and ip.exists():
            skipped += 1
            continue
        xml_str = (DATA / "condition_C" / f"{cid}.xml").read_text()
        nodes   = extract_nodes(xml_str)
        index   = build_c2_index(nodes, encoder)
        np_.write_text(json.dumps(nodes))
        faiss.write_index(index, str(ip))
    log(f"  Condition C2: {len(dataset)-skipped} built, {skipped} skipped")

    # Summary check
    n_convs = len(dataset)
    for label, pat in [
        ("A",          "condition_A/*.txt"),
        ("B chunks",   "condition_B/chunks/*.json"),
        ("B indices",  "condition_B/embeddings/*.index"),
        ("C",          "condition_C/*.xml"),
        ("C2 nodes",   "condition_C2/nodes/*.json"),
        ("C2 indices", "condition_C2/embeddings/*.index"),
    ]:
        n = len(list(DATA.glob(pat)))
        ok = "✓" if n == n_convs else f"WARNING {n}/{n_convs}"
        log(f"    {label}: {n} files [{ok}]")

    log("PHASE 1 complete.")
    del encoder
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# ── Prompt assembly ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_representations(conv_ids: list) -> dict:
    log("  Loading pre-built representations into memory...")
    reps = {}
    for cid in tqdm(conv_ids, desc="  Load reps", leave=False):
        reps[cid] = {
            "linear":   (DATA / "condition_A" / f"{cid}.txt").read_text(),
            "xml":      (DATA / "condition_C" / f"{cid}.xml").read_text(),
            "chunks":   json.loads((DATA / "condition_B" / "chunks" / f"{cid}.json").read_text()),
            "nodes":    json.loads((DATA / "condition_C2" / "nodes" / f"{cid}.json").read_text()),
            "b_index":  faiss.read_index(str(DATA / "condition_B" / "embeddings" / f"{cid}.index")),
            "c2_index": faiss.read_index(str(DATA / "condition_C2" / "embeddings" / f"{cid}.index")),
        }
    return reps


def assemble_prompt(cond: str, question: str, conv_id: str,
                    reps: dict, encoder, tokenizer) -> dict:
    rep = reps[conv_id]
    if cond == "A":
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation history below.\n\n"
            f"Conversation History:\n{rep['linear']}\n\n"
            f"Question: {question}\nAnswer:"
        )
    elif cond == "B":
        hits     = retrieve_chunks(question, rep["chunks"], rep["b_index"], encoder)
        excerpts = "\n\n".join(
            f"--- Excerpt {i+1} [{c['speaker']}, {c['date']}] ---\n{c['text']}"
            for i, c in enumerate(hits)
        )
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation excerpts below.\n\n"
            f"Retrieved Excerpts (most relevant to the question):\n{excerpts}\n\n"
            f"Question: {question}\nAnswer:"
        )
    elif cond == "C":
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation record below. The record is formatted as XML. "
            "Use the tag structure to understand the speaker, chronological, "
            "and topical organization of the conversation.\n\n"
            f"{rep['xml']}\n\n"
            f"Question: {question}\nAnswer:"
        )
    elif cond == "C2":
        hits    = retrieve_nodes_hierarchical(question, rep["nodes"], rep["c2_index"], encoder)
        context = format_c2_context(hits)
        prompt  = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation nodes below. Each node is shown with "
            "its hierarchical path (ancestors) for context, followed by its "
            "XML content.\n\n"
            f"Retrieved Nodes:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
    elif cond == "D":
        prompt = (
            "You are a helpful assistant. Answer the question as best you can.\n\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        raise ValueError(f"Unknown condition: {cond}")
    return {
        "prompt":       prompt,
        "input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ── Phase 2 & 3: Inference ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def run_inference_for_model(model_tag: str, qa_pairs: list, reps: dict,
                            encoder, conditions=None):
    from vllm import LLM, SamplingParams

    model_id   = MODEL_IDS[model_tag]
    conditions = conditions or CONDITION_ORDER
    log(f"  Loading tokenizer ({model_id})...")
    tokenizer  = AutoTokenizer.from_pretrained(model_id)

    log(f"  Loading vLLM ({model_id})...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=1,   # set to 2 if OOM on 72B
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["\n\nQuestion:", "\n\nAnswer:"],
    )

    for cond in conditions:
        out_path   = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
        n_qa_total = len(qa_pairs)
        if jsonl_line_count(out_path) == n_qa_total:
            log(f"  [{model_tag}] Condition {cond}: already complete, skipping")
            continue

        log(f"  [{model_tag}] Condition {cond}: assembling {len(qa_pairs)} prompts...")
        assembled = []
        for qa in tqdm(qa_pairs, desc=f"  Prompts {cond}", leave=False):
            item = assemble_prompt(cond, qa["question"], qa["conversation_id"],
                                   reps, encoder, tokenizer)
            assembled.append({"qa": qa, **item})

        log(f"  [{model_tag}] Condition {cond}: running batch inference...")
        t0      = time.perf_counter()
        outputs = llm.generate([a["prompt"] for a in assembled], sampling_params)
        ms_per  = (time.perf_counter() - t0) / len(outputs) * 1000

        results = []
        for item, out in zip(assembled, outputs):
            results.append({
                "question_id":       item["qa"]["question_id"],
                "conversation_id":   item["qa"]["conversation_id"],
                "condition":         cond,
                "model":             model_id,
                "category":          item["qa"]["category"],
                "question":          item["qa"]["question"],
                "reference_answer":  item["qa"]["answer"],
                "predicted_answer":  out.outputs[0].text.strip(),
                "input_tokens":      item["input_tokens"],
                "output_tokens":     len(out.outputs[0].token_ids),
                "inference_time_ms": round(ms_per, 2),
                "timestamp":         datetime.utcnow().isoformat(),
                "f1": None, "bleu1": None, "rougeL": None,
                "rouge2": None, "meteor": None, "sbert_sim": None,
            })

        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        log(f"  [{model_tag}] Condition {cond}: {len(results)} records saved ({ms_per:.1f} ms/query)")

    return llm   # caller decides when to unload


def unload_llm(llm):
    del llm
    gc.collect()
    # Do NOT call torch.cuda.empty_cache() here — it leaves CUDA initialized
    # in the parent process, which causes vLLM's next fork to fail with
    # "Cannot re-initialize CUDA in forked subprocess".
    log("  LLM unloaded.")


# ═══════════════════════════════════════════════════════════════════════════
# ── Phase 4: Evaluation ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def compute_f1(pred: str, ref: str) -> float:
    pt = pred.lower().split()
    rt = ref.lower().split()
    common = sum((Counter(pt) & Counter(rt)).values())
    if not common:
        return 0.0
    p = common / len(pt)
    r = common / len(rt)
    return 2 * p * r / (p + r)


def compute_metrics(pred: str, ref: str, rouge, sbert) -> dict:
    r    = rouge.score(ref, pred)
    pe   = sbert.encode(pred, convert_to_tensor=True)
    re_  = sbert.encode(ref,  convert_to_tensor=True)
    return {
        "f1":        compute_f1(pred, ref),
        "bleu1":     sentence_bleu([ref.lower().split()], pred.lower().split(),
                                   weights=(1,0,0,0),
                                   smoothing_function=SmoothingFunction().method1),
        "rougeL":    r["rougeL"].fmeasure,
        "rouge2":    r["rouge2"].fmeasure,
        "meteor":    meteor_score([ref.split()], pred.split()),
        "sbert_sim": float(st_util.cos_sim(pe, re_)),
    }


def phase4_evaluate() -> pd.DataFrame:
    log("═" * 60)
    log("PHASE 4 — Computing evaluation metrics")
    log("═" * 60)

    rouge = rouge_scorer_lib.RougeScorer(["rougeL", "rouge2"], use_stemmer=True)
    log("  Loading SBERT for semantic similarity...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    all_records = []
    for cond in CONDITIONS:
        for model_tag in MODELS:
            path = RESULTS / f"condition_{cond}" / f"{model_tag}.jsonl"
            if not path.exists():
                log(f"  Missing: {path}, skipping", "WARN")
                continue
            records = [json.loads(l) for l in path.open()]
            needs_scoring = any(r.get("f1") is None for r in records)
            if needs_scoring:
                log(f"  Scoring {cond}/{model_tag} ({len(records)} records)...")
                for r in tqdm(records, desc=f"  {cond}/{model_tag}", leave=False):
                    if r.get("f1") is None:
                        m = compute_metrics(r["predicted_answer"], r["reference_answer"],
                                            rouge, sbert)
                        r.update(m)
                with path.open("w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
            else:
                log(f"  {cond}/{model_tag}: already scored, loading")
            all_records.extend(records)

    del sbert
    gc.collect()

    df = pd.DataFrame(all_records)
    df = df[df["f1"].notna()].copy()
    log(f"  Total records with scores: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# ── Phase 5: Report ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_tables(df: pd.DataFrame):
    # Main accuracy table
    main = (
        df.groupby(["model", "condition", "category"])
        .agg(f1=("f1","mean"), bleu1=("bleu1","mean"),
             rougeL=("rougeL","mean"), rouge2=("rouge2","mean"),
             meteor=("meteor","mean"), sbert_sim=("sbert_sim","mean"),
             n=("f1","count"))
        .round(4).reset_index()
    )
    overall = (
        df.groupby(["model", "condition"])
        .agg(f1=("f1","mean"), bleu1=("bleu1","mean"),
             rougeL=("rougeL","mean"), rouge2=("rouge2","mean"),
             meteor=("meteor","mean"), sbert_sim=("sbert_sim","mean"),
             n=("f1","count"))
        .round(4).reset_index()
    )
    overall["category"] = "overall"
    main = pd.concat([main, overall], ignore_index=True)

    # Efficiency table
    eff = (
        df.groupby(["model", "condition"])
        .agg(mean_f1=("f1","mean"),
             mean_input_tokens=("input_tokens","mean"),
             total_input_tokens=("input_tokens","sum"),
             mean_output_tokens=("output_tokens","mean"),
             mean_time_ms=("inference_time_ms","mean"),
             total_time_ms=("inference_time_ms","sum"))
        .reset_index()
    )
    eff["total_time_min"] = eff["total_time_ms"] / 60000
    for m in eff["model"].unique():
        base = eff.loc[(eff["model"]==m)&(eff["condition"]=="A"), "mean_input_tokens"].values
        if len(base):
            mask = eff["model"]==m
            eff.loc[mask, "token_reduction_vs_A"] = (1 - eff.loc[mask,"mean_input_tokens"]/base[0])*100
    eff["f1_per_1k_tokens"] = eff["mean_f1"] / (eff["mean_input_tokens"] / 1000)
    eff = eff.round(4)

    # Significance table
    sig_rows = []
    for m in df["model"].unique():
        for cat in ["overall"] + CATEGORIES:
            sub = df[df["model"]==m] if cat=="overall" else df[(df["model"]==m)&(df["category"]==cat)]
            for base_c, test_c, label in KEY_COMPARISONS:
                b = sub[sub["condition"]==base_c]["f1"].values
                t = sub[sub["condition"]==test_c]["f1"].values
                if not len(b) or not len(t):
                    continue
                t_stat, p = stats.ttest_rel(t, b)
                d = (np.mean(t)-np.mean(b))/(np.std(b)+1e-9)
                sig_rows.append({"model":m, "category":cat, "label":label,
                                  "base":base_c, "test":test_c,
                                  "base_f1":round(float(np.mean(b)),4),
                                  "test_f1":round(float(np.mean(t)),4),
                                  "delta_f1":round(float(np.mean(t)-np.mean(b)),4),
                                  "t_stat":round(float(t_stat),3),
                                  "p_value":round(float(p),4),
                                  "cohen_d":round(float(d),3),
                                  "sig_005":bool(p<0.05)})
    sig = pd.DataFrame(sig_rows)

    # Save CSVs
    main.to_csv(EVAL/"tables"/"main_results.csv", index=False)
    eff.to_csv(EVAL/"tables"/"efficiency.csv",    index=False)
    sig.to_csv(EVAL/"tables"/"significance.csv",  index=False)
    log("  Saved: main_results.csv, efficiency.csv, significance.csv")
    return main, eff, sig


def format_results_for_report(main: pd.DataFrame, eff: pd.DataFrame,
                               sig: pd.DataFrame) -> str:
    lines = []

    lines.append("── F1 Scores by Condition and Category ─────────────────────────")
    for model_tag in MODELS:
        model_id = MODEL_IDS[model_tag]
        lines.append(f"\nModel: {model_id}")
        pivot = main[(main["model"]==model_id)].pivot_table(
            index="condition", columns="category", values="f1"
        ).round(4)
        lines.append(pivot.to_string())

    lines.append("\n── Efficiency ────────────────────────────────────────────────────")
    lines.append(eff[["model","condition","mean_f1","mean_input_tokens",
                       "f1_per_1k_tokens","token_reduction_vs_A"]].to_string(index=False))

    lines.append("\n── Statistical Significance (overall + multi_hop) ───────────────")
    for cat in ["overall","multi_hop"]:
        sub = sig[sig["category"]==cat]
        if sub.empty:
            continue
        lines.append(f"\nCategory: {cat}")
        for _, row in sub.iterrows():
            mark = "**" if row["sig_005"] else "  "
            lines.append(
                f"  {mark} [{row['model'].split('/')[-1][:3]}] {row['label']}: "
                f"base={row['base_f1']:.4f} test={row['test_f1']:.4f} "
                f"Δ={row['delta_f1']:+.4f} p={row['p_value']:.4f} d={row['cohen_d']:.3f}"
            )

    return "\n".join(lines)


def generate_llm_narrative(results_text: str, llm, sampling_params) -> str:
    """Use the loaded LLM (on H200) to write the paper section based on results."""
    prompt = (
        "You are an expert NLP researcher writing an academic paper on EngramTrace, "
        "a hierarchical XML memory architecture for LLM agents.\n\n"
        "Below are results from a controlled concept verification experiment comparing "
        "five memory representation conditions on the LoCoMo conversational QA benchmark.\n\n"
        "Conditions:\n"
        "  A  — Full conversation as plain linear text (accuracy ceiling)\n"
        "  B  — Top-5 chunks retrieved by cosine similarity (flat RAG baseline)\n"
        "  C  — Full conversation as structured XML (tests H1: does structure alone help?)\n"
        "  C2 — Top-5 XML nodes retrieved by hierarchical embeddings + ancestral path "
              "(tests H2: does structured retrieval beat flat RAG?)\n"
        "  D  — No memory, question only (floor baseline)\n\n"
        "Hypotheses:\n"
        "  H1: Hierarchical XML structure improves accuracy over flat linear text of identical content.\n"
        "  H2: Hierarchical XML retrieval with ancestral context outperforms flat chunk RAG "
              "at comparable token cost.\n\n"
        f"Experimental Results:\n{results_text}\n\n"
        "Write Section 4.1 of the paper: \"Validating the Structural Representation Hypothesis\".\n"
        "The section should:\n"
        "1. Briefly restate H1 and H2\n"
        "2. Present the key quantitative findings with specific numbers\n"
        "3. Highlight multi-hop and temporal question performance as the critical categories\n"
        "4. Discuss the efficiency finding (C2 vs A token cost)\n"
        "5. Note any unexpected results honestly\n"
        "6. Conclude with what the results imply for the EngramTrace system design\n\n"
        "Write in the style of an ACL/EMNLP paper. Length: 450-650 words.\n\n"
        "Section 4.1: Validating the Structural Representation Hypothesis\n\n"
    )
    from vllm import SamplingParams as SP
    narrative_params = SP(temperature=0.3, max_tokens=1024,
                          stop=["\n\nSection 4.2", "\n\nSection 5"])
    outputs = llm.generate([prompt], narrative_params)
    return outputs[0].outputs[0].text.strip()


def phase5_report(df: pd.DataFrame, llm_7b=None, sampling_params=None):
    log("═" * 60)
    log("PHASE 5 — Generating final report")
    log("═" * 60)

    main, eff, sig = build_tables(df)
    results_text = format_results_for_report(main, eff, sig)

    # ── LLM narrative (uses loaded 7B on H200) ────────────────────────────
    narrative = ""
    if llm_7b is not None:
        log("  Generating paper section narrative with Qwen2.5-7B...")
        try:
            narrative = generate_llm_narrative(results_text, llm_7b, sampling_params)
            log("  Narrative generated.")
        except Exception as e:
            log(f"  Narrative generation failed: {e}", "WARN")
            narrative = "(Narrative generation failed — run manually with the results above.)"

    # ── Assemble full report ──────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        "  EngramTrace Concept Verification Experiment — Final Report",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
        "",
        "EXPERIMENT SUMMARY",
        "  Benchmark : LoCoMo (snap-research/LoCoMo, test split)",
        f"  QA pairs  : {len(df[df['condition']=='A'][df['model']==MODEL_IDS['72B']]) if 'A' in df['condition'].values else 'N/A'} (per condition per model)",
        "  Models    : Qwen2.5-72B-Instruct, Qwen2.5-7B-Instruct",
        "  Conditions: A (Full Linear) | B (Flat RAG) | C (Full XML) | C2 (Hier. XML Retrieval) | D (No Memory)",
        "  Metrics   : F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT-sim",
        "",
        results_text,
    ]

    if narrative:
        report_lines += [
            "",
            "─" * 70,
            "LLM-GENERATED PAPER SECTION (Qwen2.5-7B-Instruct)",
            "─" * 70,
            "",
            "Section 4.1: Validating the Structural Representation Hypothesis",
            "",
            narrative,
        ]

    report_lines += [
        "",
        "─" * 70,
        "OUTPUT FILES",
        f"  evaluation/tables/main_results.csv  — accuracy metrics per condition/category/model",
        f"  evaluation/tables/efficiency.csv    — token counts, inference time, F1/1k tokens",
        f"  evaluation/tables/significance.csv  — paired t-tests, Cohen's d",
        f"  evaluation/REPORT.txt               — this report",
        "─" * 70,
    ]

    report_str = "\n".join(report_lines)

    # Print to console
    print("\n" + report_str)

    # Save to disk
    report_path = EVAL / "REPORT.txt"
    report_path.write_text(report_str)
    log(f"  Full report saved → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()
    log("EngramTrace Concept Verification Experiment — starting")
    log(f"Base directory: {BASE}")

    ensure_dirs()

    # ── Load LoCoMo ─────────────────────────────────────────────────────
    log("Loading LoCoMo dataset...")
    dataset = load_locomo()

    qa_path = QUESTIONS / "locomo_qa.jsonl"

    # ── Phase 1: Build representations ─────────────────────────────────
    phase1_build(dataset)

    # ── Load QA pairs + shared encoder ─────────────────────────────────
    qa_pairs = [json.loads(l) for l in qa_path.open()]
    log(f"QA pairs: {len(qa_pairs)}")
    conv_ids = sorted({qa["conversation_id"] for qa in qa_pairs})

    log("Loading sentence encoder for inference...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    reps    = load_representations(conv_ids)

    # ── Phase 2: Inference — 72B ────────────────────────────────────────
    log("═" * 60)
    log("PHASE 2 — Inference (Qwen2.5-72B-Instruct)")
    log("═" * 60)
    llm_72b = run_inference_for_model("72B", qa_pairs, reps, encoder)
    unload_llm(llm_72b)

    # ── Phase 3: Inference — 7B ─────────────────────────────────────────
    log("═" * 60)
    log("PHASE 3 — Inference (Qwen2.5-7B-Instruct)")
    log("═" * 60)
    llm_7b = run_inference_for_model("7B", qa_pairs, reps, encoder)
    # Keep llm_7b loaded for report generation in Phase 5

    del encoder, reps
    gc.collect()

    # ── Phase 4: Evaluation ─────────────────────────────────────────────
    df = phase4_evaluate()

    # ── Phase 5: Report (uses loaded 7B) ────────────────────────────────
    from vllm import SamplingParams
    sp_report = SamplingParams(temperature=0.3, max_tokens=1024)
    phase5_report(df, llm_7b=llm_7b, sampling_params=sp_report)
    unload_llm(llm_7b)

    elapsed = (time.perf_counter() - t_start) / 3600
    log(f"Experiment complete. Total wall time: {elapsed:.2f} hours")
    log(f"Final report: {EVAL / 'REPORT.txt'}")


if __name__ == "__main__":
    # Set BEFORE main() so it is in effect before vLLM imports multiprocessing.
    # 'spawn' starts child processes fresh, avoiding CUDA re-init errors when
    # vLLM forks its EngineCore after the parent process has touched CUDA.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
