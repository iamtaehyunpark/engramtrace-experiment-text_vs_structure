"""
Phase 4 — Compute metrics, run significance tests, generate tables.

Run after all inference JSONL files are complete.

Usage:
    python evaluate.py
    python evaluate.py --model 72B   # single model only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy import stats
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import compute_all_metrics


CONDITIONS = ['A', 'B', 'C', 'C2', 'D']
MODELS     = ['72B', '7B']
CATEGORIES = ['single_hop', 'multi_hop', 'temporal', 'open_domain', 'adversarial']

KEY_COMPARISONS = [
    ('A',  'C',  'C  vs A  — structure vs full linear (H1)'),
    ('B',  'C2', 'C2 vs B  — hierarchical retrieval vs flat RAG (H2)'),
    ('A',  'C2', 'C2 vs A  — hierarchical retrieval vs full context (efficiency)'),
    ('C',  'C2', 'C2 vs C  — retrieval over XML vs full XML'),
    ('A',  'B',  'B  vs A  — flat RAG vs full context (lit replication)'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['72B', '7B'],
                   help='Evaluate a single model (default: both)')
    p.add_argument('--results-dir',    default='results')
    p.add_argument('--evaluation-dir', default='evaluation')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Score all records (fills metric fields in-place and resaves JSONL)
# ---------------------------------------------------------------------------

def score_jsonl(path: Path, rouge, sbert) -> list:
    """Load a JSONL, compute metrics for every record, return list of dicts."""
    records = [json.loads(l) for l in path.open()]
    for r in tqdm(records, desc=f'  Scoring {path.name}', leave=False):
        if r.get('f1') is not None:
            continue   # already scored
        metrics = compute_all_metrics(r['predicted_answer'],
                                      r['reference_answer'], rouge, sbert)
        r.update(metrics)
    # Rewrite with scores
    with path.open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return records


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def compare_conditions(df: pd.DataFrame, model: str, category: str,
                       base_cond: str, test_cond: str, label: str) -> dict:
    sub = df[(df['model'] == model) & (df['category'] == category)]
    base_scores = sub[sub['condition'] == base_cond]['f1'].values
    test_scores = sub[sub['condition'] == test_cond]['f1'].values
    if len(base_scores) == 0 or len(test_scores) == 0:
        return {}
    t_stat, p_value = stats.ttest_rel(test_scores, base_scores)
    cohen_d = (np.mean(test_scores) - np.mean(base_scores)) / (np.std(base_scores) + 1e-9)
    return {
        'model':    model,
        'category': category,
        'label':    label,
        'base':     base_cond,
        'test':     test_cond,
        'base_f1':  float(np.mean(base_scores)),
        'test_f1':  float(np.mean(test_scores)),
        'delta_f1': float(np.mean(test_scores) - np.mean(base_scores)),
        't_stat':   float(t_stat),
        'p_value':  float(p_value),
        'cohen_d':  float(cohen_d),
        'sig_005':  bool(p_value < 0.05),
    }


def run_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df['model'].unique():
        for category in ['all'] + CATEGORIES:
            sub = df[df['model'] == model] if category == 'all' else \
                  df[(df['model'] == model) & (df['category'] == category)]
            for base_cond, test_cond, label in KEY_COMPARISONS:
                b = sub[sub['condition'] == base_cond]['f1'].values
                t = sub[sub['condition'] == test_cond]['f1'].values
                if len(b) == 0 or len(t) == 0:
                    continue
                t_stat, p_value = stats.ttest_rel(t, b)
                cohen_d = (np.mean(t) - np.mean(b)) / (np.std(b) + 1e-9)
                rows.append({
                    'model':    model,
                    'category': category,
                    'label':    label,
                    'base':     base_cond,
                    'test':     test_cond,
                    'base_f1':  round(float(np.mean(b)), 4),
                    'test_f1':  round(float(np.mean(t)), 4),
                    'delta_f1': round(float(np.mean(t) - np.mean(b)), 4),
                    't_stat':   round(float(t_stat), 3),
                    'p_value':  round(float(p_value), 4),
                    'cohen_d':  round(float(cohen_d), 3),
                    'sig_005':  bool(p_value < 0.05),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args      = parse_args()
    base_dir  = Path(__file__).parent
    res_dir   = base_dir / args.results_dir
    eval_dir  = base_dir / args.evaluation_dir
    (eval_dir / 'scores').mkdir(parents=True, exist_ok=True)
    (eval_dir / 'tables').mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else MODELS

    print("Loading rouge scorer and SBERT...")
    rouge = rouge_scorer.RougeScorer(['rougeL', 'rouge2'], use_stemmer=True)
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Score all JSONL files
    all_records = []
    for condition in CONDITIONS:
        for model in models:
            path = res_dir / f'condition_{condition}' / f'{model}.jsonl'
            if not path.exists():
                print(f"  Skipping {path} (not found)")
                continue
            print(f"Scoring {condition}/{model}...")
            records = score_jsonl(path, rouge, sbert)
            all_records.extend(records)
            print(f"  {len(records)} records scored")

    if not all_records:
        print("No records found. Run inference first.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    # Drop rows where metrics are still None (shouldn't happen)
    df = df[df['f1'].notna()].copy()

    # -----------------------------------------------------------------------
    # Main accuracy table: condition × category × model
    # -----------------------------------------------------------------------
    main_table = (
        df.groupby(['model', 'condition', 'category'])
        .agg(
            f1=('f1', 'mean'),
            bleu1=('bleu1', 'mean'),
            rougeL=('rougeL', 'mean'),
            rouge2=('rouge2', 'mean'),
            meteor=('meteor', 'mean'),
            sbert_sim=('sbert_sim', 'mean'),
            n=('f1', 'count'),
        )
        .round(4)
        .reset_index()
    )
    # Add overall row per model × condition
    overall = (
        df.groupby(['model', 'condition'])
        .agg(
            f1=('f1', 'mean'),
            bleu1=('bleu1', 'mean'),
            rougeL=('rougeL', 'mean'),
            rouge2=('rouge2', 'mean'),
            meteor=('meteor', 'mean'),
            sbert_sim=('sbert_sim', 'mean'),
            n=('f1', 'count'),
        )
        .round(4)
        .reset_index()
    )
    overall['category'] = 'overall'
    main_table = pd.concat([main_table, overall], ignore_index=True)

    # -----------------------------------------------------------------------
    # Efficiency table
    # -----------------------------------------------------------------------
    eff = (
        df.groupby(['model', 'condition'])
        .agg(
            mean_f1=('f1', 'mean'),
            mean_input_tokens=('input_tokens', 'mean'),
            total_input_tokens=('input_tokens', 'sum'),
            mean_output_tokens=('output_tokens', 'mean'),
            mean_time_ms=('inference_time_ms', 'mean'),
        )
        .reset_index()
    )
    eff['f1_per_1k_tokens'] = eff['mean_f1'] / (eff['mean_input_tokens'] / 1000)
    for model in eff['model'].unique():
        baseline = eff.loc[(eff['model'] == model) & (eff['condition'] == 'A'),
                           'mean_input_tokens'].values
        if len(baseline) == 0:
            continue
        mask = eff['model'] == model
        eff.loc[mask, 'token_reduction_vs_A'] = (
            1 - eff.loc[mask, 'mean_input_tokens'] / baseline[0]
        ) * 100
    eff = eff.round(4)

    # -----------------------------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------------------------
    sig_table = run_significance_tests(df)

    # -----------------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------------
    main_table.to_csv(eval_dir / 'tables' / 'main_results.csv', index=False)
    eff.to_csv(eval_dir / 'tables' / 'efficiency.csv', index=False)
    sig_table.to_csv(eval_dir / 'tables' / 'significance.csv', index=False)
    print(f"\nTables saved to {eval_dir / 'tables'}/")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n=== MAIN RESULTS (F1, overall) ===")
    pivot = main_table[main_table['category'] == 'overall'].pivot_table(
        index='condition', columns='model', values='f1'
    ).round(4)
    print(pivot.to_string())

    print("\n=== EFFICIENCY ===")
    print(eff[['model', 'condition', 'mean_f1', 'mean_input_tokens',
               'f1_per_1k_tokens', 'token_reduction_vs_A']].to_string(index=False))

    print("\n=== SIGNIFICANCE TESTS (overall, multi_hop) ===")
    for cat in ['overall', 'multi_hop']:
        sub = sig_table[(sig_table['category'] == cat)]
        if sub.empty:
            continue
        print(f"\n  Category: {cat}")
        for _, row in sub.iterrows():
            sig = '**' if row['sig_005'] else '  '
            print(f"  {sig} [{row['model']}] {row['label']}: "
                  f"base={row['base_f1']:.4f}, test={row['test_f1']:.4f}, "
                  f"Δ={row['delta_f1']:+.4f}, p={row['p_value']:.4f}, "
                  f"d={row['cohen_d']:.3f}")


if __name__ == '__main__':
    main()
