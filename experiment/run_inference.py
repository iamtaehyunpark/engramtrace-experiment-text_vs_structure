"""
Phase 2 & 3 — Run vLLM inference for all conditions.

Requires an NVIDIA GPU with enough VRAM (H200 recommended).
Run Phase 1 (build_representations.py) first.

Usage:
    # 72B model
    python run_inference.py --model 72B

    # 7B model
    python run_inference.py --model 7B

    # Single condition (for resume / re-run)
    python run_inference.py --model 72B --condition C2
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    retrieve_chunks,
    retrieve_nodes_hierarchical,
    format_c2_context,
)

MODEL_IDS = {
    '72B': 'Qwen/Qwen2.5-72B-Instruct',
    '7B':  'Qwen/Qwen2.5-7B-Instruct',
}

# Execution order: shortest prompts first (warm CUDA graph cache)
CONDITION_ORDER = ['D', 'B', 'C2', 'C', 'A']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['72B', '7B'])
    p.add_argument('--condition', choices=CONDITION_ORDER,
                   help='Run a single condition (default: all, shortest-first)')
    p.add_argument('--data-dir',    default='data')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--k',           type=int, default=5, help='Retrieval top-k')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load pre-built representations from disk
# ---------------------------------------------------------------------------

def load_all_representations(data_dir: Path, conv_ids: list) -> dict:
    print("Loading pre-built representations...")
    reps = {}
    for cid in tqdm(conv_ids, desc='Loading reps'):
        reps[cid] = {
            'linear':   (data_dir / 'condition_A' / f'{cid}.txt').read_text(),
            'xml':      (data_dir / 'condition_C' / f'{cid}.xml').read_text(),
            'chunks':   json.loads((data_dir / 'condition_B' / 'chunks' / f'{cid}.json').read_text()),
            'nodes':    json.loads((data_dir / 'condition_C2' / 'nodes' / f'{cid}.json').read_text()),
            'b_index':  faiss.read_index(str(data_dir / 'condition_B' / 'embeddings' / f'{cid}.index')),
            'c2_index': faiss.read_index(str(data_dir / 'condition_C2' / 'embeddings' / f'{cid}.index')),
        }
    return reps


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(condition: str, question: str, conv_id: str,
                    reps: dict, encoder, tokenizer, k: int = 5) -> dict:
    rep = reps[conv_id]

    if condition == 'A':
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation history below.\n\n"
            f"Conversation History:\n{rep['linear']}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'B':
        retrieved = retrieve_chunks(question, rep['chunks'], rep['b_index'],
                                    encoder, k=k)
        excerpts  = '\n\n'.join([
            f"--- Excerpt {i+1} [{c['speaker']}, {c['date']}] ---\n{c['text']}"
            for i, c in enumerate(retrieved)
        ])
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation excerpts below.\n\n"
            f"Retrieved Excerpts:\n{excerpts}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'C':
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation record below. The record is formatted as XML. "
            "Use the tag structure to understand the speaker, chronological, "
            "and topical organization of the conversation.\n\n"
            f"{rep['xml']}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'C2':
        retrieved = retrieve_nodes_hierarchical(
            question, rep['nodes'], rep['c2_index'], encoder, k=k
        )
        context = format_c2_context(retrieved)
        prompt  = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation nodes below. Each node is shown with "
            "its hierarchical path (ancestors) for context, followed by its "
            "XML content.\n\n"
            f"Retrieved Nodes:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'D':
        prompt = (
            "You are a helpful assistant. Answer the question as best you can.\n\n"
            f"Question: {question}\nAnswer:"
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return {
        'prompt':       prompt,
        'input_tokens': len(tokenizer.encode(prompt, add_special_tokens=False)),
    }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def result_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def run_condition(condition: str, model_id: str, model_tag: str,
                  qa_pairs: list, reps: dict,
                  encoder, tokenizer, llm, sampling_params,
                  results_dir: Path, k: int):
    out_path = results_dir / f'condition_{condition}' / f'{model_tag}.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = result_count(out_path)
    if existing == 1540:
        print(f"  Condition {condition}: already complete (1,540 records), skipping")
        return
    if existing > 0:
        print(f"  Condition {condition}: {existing}/1,540 records found, re-running from scratch")

    print(f"  Assembling {len(qa_pairs)} prompts for condition {condition}...")
    assembled = []
    for qa in tqdm(qa_pairs, desc=f'  Prompts {condition}', leave=False):
        item = assemble_prompt(condition, qa['question'], qa['conversation_id'],
                               reps, encoder, tokenizer, k=k)
        assembled.append({'qa': qa, **item})

    prompts = [a['prompt'] for a in assembled]
    print(f"  Running vLLM batch inference ({len(prompts)} prompts)...")
    t_start = time.perf_counter()
    outputs  = llm.generate(prompts, sampling_params)
    t_total  = time.perf_counter() - t_start
    ms_per   = (t_total / len(outputs)) * 1000

    results = []
    for item, output in zip(assembled, outputs):
        results.append({
            'question_id':       item['qa']['question_id'],
            'conversation_id':   item['qa']['conversation_id'],
            'condition':         condition,
            'model':             model_id,
            'category':          item['qa']['category'],
            'question':          item['qa']['question'],
            'reference_answer':  item['qa']['answer'],
            'predicted_answer':  output.outputs[0].text.strip(),
            'input_tokens':      item['input_tokens'],
            'output_tokens':     len(output.outputs[0].token_ids),
            'inference_time_ms': ms_per,
            'timestamp':         datetime.utcnow().isoformat(),
            # metrics filled in by evaluate.py
            'f1': None, 'bleu1': None, 'rougeL': None,
            'rouge2': None, 'meteor': None, 'sbert_sim': None,
        })

    with out_path.open('w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved {len(results)} records → {out_path}  ({ms_per:.1f} ms/query)")


def main():
    args      = parse_args()
    model_id  = MODEL_IDS[args.model]
    base_dir  = Path(__file__).parent
    data_dir  = base_dir / args.data_dir
    res_dir   = base_dir / args.results_dir
    conditions = [args.condition] if args.condition else CONDITION_ORDER

    # Load QA pairs
    qa_path  = base_dir / 'questions' / 'locomo_qa.jsonl'
    qa_pairs = [json.loads(l) for l in qa_path.open()]
    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Unique conversation IDs
    conv_ids = sorted({qa['conversation_id'] for qa in qa_pairs})

    # Load encoder (needed for B and C2 prompt assembly)
    print("Loading sentence encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load tokenizer for token counting
    print(f"Loading tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load pre-built representations
    reps = load_all_representations(data_dir, conv_ids)

    # Load vLLM model
    print(f"\nLoading vLLM model: {model_id}")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_id,
        dtype='bfloat16',
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=['\n\nQuestion:', '\n\nAnswer:'],
    )

    print(f"\nRunning conditions: {conditions} (model={args.model})")
    for cond in conditions:
        print(f"\n--- Condition {cond} ---")
        run_condition(cond, model_id, args.model, qa_pairs, reps,
                      encoder, tokenizer, llm, sampling_params,
                      res_dir, k=args.k)

    print("\n=== Inference complete ===")
    for cond in conditions:
        path  = res_dir / f'condition_{cond}' / f'{args.model}.jsonl'
        count = result_count(path)
        status = '✓' if count == 1540 else f'WARNING: {count}/1540'
        print(f"  {cond}/{args.model}: {count} records  [{status}]")


if __name__ == '__main__':
    main()
