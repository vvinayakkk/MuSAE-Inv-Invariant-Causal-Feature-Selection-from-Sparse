#!/usr/bin/env python3
"""
Download all datasets used in MuSAE-Inv experiments.

Downloads and caches:
    - HaluEval QA, Dialogue, Summarisation (from HuggingFace)
    - TruthfulQA multiple_choice (from HuggingFace)
    - Gemma-2-2B model weights (optional, requires HF_TOKEN)
    - Gemma Scope SAE weights (optional)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --include-model
    HF_TOKEN=hf_xxx python scripts/download_data.py --include-model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Download Datasets")
    parser.add_argument("--include-model", action="store_true",
                        help="Also download Gemma-2-2B model weights")
    parser.add_argument("--include-saes", action="store_true",
                        help="Also download Gemma Scope SAE weights")
    parser.add_argument("--output-dir", type=str, default="./data/raw",
                        help="Directory for downloaded data")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MuSAE-Inv Dataset Download")
    print("=" * 60)

    # ── HaluEval ──────────────────────────────────────────
    print("\n[1/4] Downloading HaluEval QA...")
    from datasets import load_dataset
    hq = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    print(f"  ✅ HaluEval QA: {len(hq)} examples")

    print("\n[2/4] Downloading HaluEval Dialogue...")
    hd = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
    print(f"  ✅ HaluEval Dialogue: {len(hd)} examples")

    print("\n[3/4] Downloading HaluEval Summarisation...")
    hs = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
    print(f"  ✅ HaluEval Summarisation: {len(hs)} examples")

    print("\n[4/4] Downloading TruthfulQA...")
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    print(f"  ✅ TruthfulQA: {len(tqa)} examples")

    # ── Model (optional) ─────────────────────────────────
    if args.include_model:
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("\n⚠️  Set HF_TOKEN environment variable for gated model access")
        else:
            print("\n[5] Downloading Gemma-2-2B model weights...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login
            login(token=token)
            AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
            AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b",
                torch_dtype="auto",
                token=token,
            )
            print("  ✅ Gemma-2-2B weights cached")

    if args.include_saes:
        print("\n[6] Downloading Gemma Scope SAEs...")
        from sae_lens import SAE
        for layer in [6, 12, 18, 25]:
            sae_id = f"layer_{layer}/width_16k/canonical"
            print(f"  Downloading SAE layer {layer}...")
            SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=sae_id,
                device="cpu",
            )
        print("  ✅ All SAEs cached")

    print("\n" + "=" * 60)
    print("  ✅ All downloads complete. Datasets are cached by HuggingFace.")
    print("=" * 60)


if __name__ == "__main__":
    main()
