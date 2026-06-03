#!/usr/bin/env python3
"""
debug_vllm.py — Diagnostic script for vLLM + Qwen on this machine.
Prints every library version, CUDA detail, and full traceback at each step.
"""

import sys
import subprocess
import traceback

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        print(out.strip())
    except subprocess.CalledProcessError as e:
        print(f"[cmd failed] {e.output.strip()}")

# ── 1. System & driver info ──────────────────────────────────────────────────
section("1. System info")
run("uname -a")
run("nvidia-smi | head -20")
run("ldconfig -p | grep libcudart || echo 'ldconfig: no libcudart found'")
run("find /usr /data /opt -name 'libcudart.so*' 2>/dev/null || echo 'find: none'")

# ── 2. Python & package versions ────────────────────────────────────────────
section("2. Python & package versions")
print(f"Python: {sys.version}")
for pkg in ["torch", "vllm", "transformers", "accelerate", "numpy",
            "setuptools", "triton", "xformers"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  {pkg}: {ver}")
    except ImportError as e:
        print(f"  {pkg}: NOT INSTALLED ({e})")

# ── 3. PyTorch CUDA check ────────────────────────────────────────────────────
section("3. PyTorch CUDA")
try:
    import torch
    print(f"  torch.version.cuda   : {torch.version.cuda}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device count : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  VRAM={props.total_memory//1024**3}GB  "
                  f"compute={props.major}.{props.minor}")
    else:
        print("  [WARNING] CUDA not available to torch")
        print(f"  CUDA_HOME env: {__import__('os').environ.get('CUDA_HOME', 'not set')}")
        print(f"  LD_LIBRARY_PATH: {__import__('os').environ.get('LD_LIBRARY_PATH', 'not set')}")
except Exception:
    print("  [FATAL] torch import or CUDA check failed:")
    traceback.print_exc()

# ── 4. vLLM import ──────────────────────────────────────────────────────────
section("4. vLLM import")
try:
    import vllm
    print(f"  vllm version: {vllm.__version__}")
    print(f"  vllm path: {vllm.__file__}")
except Exception:
    print("  [FATAL] vllm import failed:")
    traceback.print_exc()
    sys.exit(1)

# ── 5. vLLM LLM + SamplingParams import ─────────────────────────────────────
section("5. vllm.LLM + SamplingParams import")
try:
    from vllm import LLM, SamplingParams
    print("  OK: LLM and SamplingParams imported")
except Exception:
    print("  [FATAL] from vllm import LLM, SamplingParams failed:")
    traceback.print_exc()
    sys.exit(1)

# ── 6. Tokenizer load ────────────────────────────────────────────────────────
section("6. Tokenizer load (Qwen2.5-7B-Instruct)")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  OK: tokenizer loaded, vocab_size={tok.vocab_size}")
except Exception:
    print("  [FATAL] tokenizer load failed:")
    traceback.print_exc()
    sys.exit(1)

# ── 7. LLM init ──────────────────────────────────────────────────────────────
section("7. LLM init (tensor_parallel_size=4, bfloat16)")
llm = None
try:
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
    )
    print("  OK: LLM initialized")
except Exception:
    print("  [ERROR] LLM init with tensor_parallel_size=4 failed:")
    traceback.print_exc()

# ── 7b. Fallback: single GPU ─────────────────────────────────────────────────
if llm is None:
    section("7b. Fallback: tensor_parallel_size=1")
    try:
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.80,
            enforce_eager=True,
        )
        print("  OK: LLM initialized (single GPU, eager mode)")
    except Exception:
        print("  [FATAL] fallback LLM init also failed:")
        traceback.print_exc()
        sys.exit(1)

# ── 8. Generate ──────────────────────────────────────────────────────────────
section("8. Test generation")
try:
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    prompt = "What is 2 + 2? Answer in one sentence."
    print(f"  Prompt: {prompt!r}")
    outputs = llm.generate([prompt], sp)
    print(f"  Response: {outputs[0].outputs[0].text.strip()!r}")
    print("  OK: generation successful")
except Exception:
    print("  [FATAL] generate failed:")
    traceback.print_exc()
    sys.exit(1)

section("DONE — all checks passed")
