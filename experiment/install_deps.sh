#!/bin/bash
# Install all dependencies for the EngramTrace experiment.
# Uses HuggingFace transformers for inference (compatible with CUDA 12.6).
# PyTorch cu126 is installed first so everything links against it.

set -e

echo "=== Step 1: PyTorch with CUDA 12.6 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== Step 2: ML and experiment libraries ==="
pip install \
    accelerate \
    transformers \
    datasets \
    sentence-transformers \
    faiss-cpu \
    rouge-score \
    nltk \
    scipy \
    pandas \
    tqdm

echo ""
echo "=== Step 3: NLTK data ==="
python -c "
import nltk
for pkg in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)
    print(f'  [done] {pkg}')
"

echo ""
echo "=== Verifying ==="
python -c "
import importlib, sys, torch
libs = ['torch','transformers','accelerate','datasets','sentence_transformers',
        'faiss','rouge_score','nltk','scipy','pandas','numpy','tqdm']
ok = True
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f'  OK  {lib}')
    except ImportError as e:
        print(f'  FAIL {lib}: {e}')
        ok = False
print(f'  CUDA available : {torch.cuda.is_available()}')
print(f'  GPU count      : {torch.cuda.device_count()}')
print(f'  PyTorch version: {torch.__version__}')
sys.exit(0 if ok else 1)
"
