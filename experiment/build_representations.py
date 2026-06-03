"""
Phase 1 — Build all data representations.

Run on CPU before any inference. Outputs are saved to disk so that all
inference conditions read from identical pre-built files.

Usage:
    python build_representations.py [--data-dir DATA_DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow running from the experiment/ directory or the project root
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_linear_text,
    build_chunks,
    build_faiss_flat_index,
    build_xml,
    validate_xml,
    extract_nodes,
    build_c2_index,
    build_html,
    validate_html,
    extract_html_nodes,
    build_e2_index,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data',
                   help='Root data directory (default: data/)')
    return p.parse_args()


def load_locomo():
    print("Loading LoCoMo dataset (snap-research/LoCoMo, test split)...")
    dataset = load_dataset('snap-research/LoCoMo', split='test')
    print(f"  Loaded {len(dataset)} conversations")
    return dataset


def build_qa_file(dataset, out_path: Path):
    if out_path.exists():
        count = sum(1 for _ in out_path.open())
        print(f"  QA file exists ({count} records), skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for conv in dataset:
        for qa in conv['qa_pairs']:
            if qa['answer'] != 'unanswerable':
                records.append({
                    'question_id':     qa['id'],
                    'conversation_id': conv['conversation_id'],
                    'category':        qa['category'],
                    'question':        qa['question'],
                    'answer':          qa['answer'],
                    'evidence':        qa.get('evidence', []),
                })
    with out_path.open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved {len(records)} QA pairs → {out_path}")
    assert len(records) == 1540, f"Expected 1,540 answerable QA pairs, got {len(records)}"


def build_condition_a(dataset, data_dir: Path):
    out_dir = data_dir / 'condition_A'
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for conv in tqdm(dataset, desc='Condition A'):
        cid  = conv['conversation_id']
        path = out_dir / f'{cid}.txt'
        if path.exists():
            skipped += 1
            continue
        path.write_text(build_linear_text(conv))
    print(f"  Condition A: {len(dataset) - skipped} built, {skipped} skipped")


def build_condition_b(dataset, data_dir: Path, encoder):
    chunk_dir = data_dir / 'condition_B' / 'chunks'
    emb_dir   = data_dir / 'condition_B' / 'embeddings'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for conv in tqdm(dataset, desc='Condition B'):
        cid        = conv['conversation_id']
        chunk_path = chunk_dir / f'{cid}.json'
        index_path = emb_dir   / f'{cid}.index'
        if chunk_path.exists() and index_path.exists():
            skipped += 1
            continue
        chunks = build_chunks(conv)
        index  = build_faiss_flat_index(chunks, encoder)
        with chunk_path.open('w') as f:
            json.dump(chunks, f)
        faiss.write_index(index, str(index_path))
    print(f"  Condition B: {len(dataset) - skipped} built, {skipped} skipped")


def build_condition_c(dataset, data_dir: Path) -> list:
    """Returns list of conversation_ids that failed validation."""
    out_dir = data_dir / 'condition_C'
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped  = 0
    for conv in tqdm(dataset, desc='Condition C (XML)'):
        cid  = conv['conversation_id']
        path = out_dir / f'{cid}.xml'
        if path.exists():
            skipped += 1
            continue
        xml_str = build_xml(conv)
        if not validate_xml(conv, xml_str):
            failures.append(cid)
            print(f"  VALIDATION FAILED: {cid}")
        else:
            path.write_text(xml_str)
    built = len(dataset) - skipped - len(failures)
    print(f"  Condition C: {built} built, {skipped} skipped, {len(failures)} failed")
    return failures


def build_condition_c2(dataset, data_dir: Path, encoder):
    node_dir  = data_dir / 'condition_C2' / 'nodes'
    emb_dir   = data_dir / 'condition_C2' / 'embeddings'
    xml_dir   = data_dir / 'condition_C'
    node_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    missing_xml = 0
    for conv in tqdm(dataset, desc='Condition C2'):
        cid        = conv['conversation_id']
        node_path  = node_dir / f'{cid}.json'
        index_path = emb_dir  / f'{cid}.index'
        if node_path.exists() and index_path.exists():
            skipped += 1
            continue
        xml_path = xml_dir / f'{cid}.xml'
        if not xml_path.exists():
            missing_xml += 1
            print(f"  Missing XML for {cid}, skipping C2")
            continue
        xml_str = xml_path.read_text()
        nodes   = extract_nodes(xml_str)
        index   = build_c2_index(nodes, encoder)
        with node_path.open('w') as f:
            json.dump(nodes, f)
        faiss.write_index(index, str(index_path))
    built = len(dataset) - skipped - missing_xml
    print(f"  Condition C2: {built} built, {skipped} skipped, {missing_xml} missing XML")


def build_condition_e(dataset, data_dir: Path) -> list:
    """Returns list of conversation_ids that failed validation."""
    out_dir = data_dir / 'condition_E'
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped  = 0
    for conv in tqdm(dataset, desc='Condition E (HTML)'):
        cid  = conv['conversation_id']
        path = out_dir / f'{cid}.html'
        if path.exists():
            skipped += 1
            continue
        html_str = build_html(conv)
        if not validate_html(conv, html_str):
            failures.append(cid)
            print(f'  VALIDATION FAILED (HTML): {cid}')
        else:
            path.write_text(html_str)
    built = len(dataset) - skipped - len(failures)
    print(f'  Condition E: {built} built, {skipped} skipped, {len(failures)} failed')
    return failures


def build_condition_e2(dataset, data_dir: Path, encoder):
    node_dir  = data_dir / 'condition_E2' / 'nodes'
    emb_dir   = data_dir / 'condition_E2' / 'embeddings'
    html_dir  = data_dir / 'condition_E'
    node_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    missing_html = 0
    for conv in tqdm(dataset, desc='Condition E2'):
        cid        = conv['conversation_id']
        node_path  = node_dir / f'{cid}.json'
        index_path = emb_dir  / f'{cid}.index'
        if node_path.exists() and index_path.exists():
            skipped += 1
            continue
        html_path = html_dir / f'{cid}.html'
        if not html_path.exists():
            missing_html += 1
            print(f'  Missing HTML for {cid}, skipping E2')
            continue
        html_str = html_path.read_text()
        nodes    = extract_html_nodes(html_str)
        index    = build_e2_index(nodes, encoder)
        with node_path.open('w') as f:
            json.dump(nodes, f)
        faiss.write_index(index, str(index_path))
    built = len(dataset) - skipped - missing_html
    print(f'  Condition E2: {built} built, {skipped} skipped, {missing_html} missing HTML')


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    base_dir = Path(__file__).parent
    data_dir = base_dir / data_dir if not data_dir.is_absolute() else data_dir

    dataset = load_locomo()

    # QA file
    print("\n[1/7] Building QA file...")
    build_qa_file(dataset, base_dir / 'questions' / 'locomo_qa.jsonl')

    # Condition A — CPU only
    print("\n[2/7] Building Condition A (linear text)...")
    build_condition_a(dataset, data_dir)

    # Condition C — CPU only (must succeed 100% before C2 and before inference)
    print("\n[3/7] Building Condition C (XML)...")
    failures = build_condition_c(dataset, data_dir)
    if failures:
        print(f"\nERROR: {len(failures)} XML validation failures. "
              "Fix before proceeding.\n  Failed: " + ', '.join(failures))
        sys.exit(1)
    print("  All XML files passed validation.")

    # Condition E — HTML (analogous to C)
    print("\n[4/7] Building Condition E (HTML)...")
    failures = build_condition_e(dataset, data_dir)
    if failures:
        print(f"\nERROR: {len(failures)} HTML validation failures. "
              "Fix before proceeding.\n  Failed: " + ', '.join(failures))
        sys.exit(1)
    print("  All HTML files passed validation.")

    # Encoder needed for B, C2, E2
    print("\n[5/7] Building Condition B (chunked RAG)...")
    print("  Loading sentence encoder...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    build_condition_b(dataset, data_dir, encoder)

    # Condition C2 — reuses XML from Condition C
    print("\n[6/7] Building Condition C2 (hierarchical XML)...")
    build_condition_c2(dataset, data_dir, encoder)

    # Condition E2 — reuses HTML from Condition E
    print("\n[7/7] Building Condition E2 (hierarchical HTML)...")
    build_condition_e2(dataset, data_dir, encoder)

    # Final summary
    print("\n=== Phase 1 complete ===")
    for cond, subpath in [
        ('A', 'condition_A/*.txt'),
        ('B chunks', 'condition_B/chunks/*.json'),
        ('B indices', 'condition_B/embeddings/*.index'),
        ('C', 'condition_C/*.xml'),
        ('C2 nodes', 'condition_C2/nodes/*.json'),
        ('C2 indices', 'condition_C2/embeddings/*.index'),
        ('E', 'condition_E/*.html'),
        ('E2 nodes', 'condition_E2/nodes/*.json'),
        ('E2 indices', 'condition_E2/embeddings/*.index'),
    ]:
        count = len(list(data_dir.glob(subpath)))
        status = '✓' if count == 50 else f'WARNING: {count}/50'
        print(f"  {cond}: {count} files  [{status}]")
    qa_count = sum(1 for _ in (base_dir / 'questions' / 'locomo_qa.jsonl').open())
    print(f"  QA pairs: {qa_count}  [{'✓' if qa_count == 1540 else f'WARNING: {qa_count}/1540'}]")


if __name__ == '__main__':
    main()
