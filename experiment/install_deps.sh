#!/bin/bash
# Installs each dependency one by one.
# Safe to re-run — skips libraries that are already importable.

set -e

LIBS=(
    "vllm"
    "transformers"
    "datasets"
    "sentence-transformers"
    "faiss-cpu"
    "rouge-score"
    "nltk"
    "scipy"
    "pandas"
    "numpy"
    "tqdm"
)

# Map pip package name -> python import name (only where they differ)
declare -A IMPORT_NAME
IMPORT_NAME["sentence-transformers"]="sentence_transformers"
IMPORT_NAME["faiss-cpu"]="faiss"
IMPORT_NAME["rouge-score"]="rouge_score"

echo "=== Installing dependencies ==="
for pkg in "${LIBS[@]}"; do
    import="${IMPORT_NAME[$pkg]:-$pkg}"
    if python -c "import $import" 2>/dev/null; then
        echo "  [skip] $pkg already installed"
    else
        echo "  [installing] $pkg ..."
        pip install "$pkg"
        echo "  [done] $pkg"
    fi
done

echo ""
echo "=== Downloading NLTK data ==="
python -c "
import nltk
for pkg in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        if pkg == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif pkg == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif pkg == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        print(f'  [skip] {pkg} already downloaded')
    except LookupError:
        print(f'  [downloading] {pkg} ...')
        nltk.download(pkg, quiet=True)
        print(f'  [done] {pkg}')
"

echo ""
echo "=== Verifying all imports ==="
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
