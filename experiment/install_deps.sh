#!/bin/bash
# Install all dependencies for the EngramTrace experiment.
# vllm is installed first and alone — it pins compatible versions of torch
# and transformers. Everything else is installed after.

set -e

echo "=== Step 1: Install vLLM (pulls compatible torch + transformers) ==="
pip install vllm

echo ""
echo "=== Step 2: Install remaining dependencies ==="
pip install \
    datasets \
    sentence-transformers \
    faiss-cpu \
    rouge-score \
    nltk \
    scipy \
    pandas \
    tqdm

echo ""
echo "=== Step 3: Download NLTK data ==="
python -c "
import nltk
for pkg in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)
    print(f'  [done] {pkg}')
"

echo ""
echo "=== Verifying imports ==="
python -c "
import importlib, sys
libs = ['vllm','transformers','datasets','sentence_transformers',
        'faiss','rouge_score','nltk','scipy','pandas','numpy','tqdm','torch']
ok = True
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f'  OK  {lib}')
    except ImportError as e:
        print(f'  FAIL {lib}: {e}')
        ok = False
sys.exit(0 if ok else 1)
"
