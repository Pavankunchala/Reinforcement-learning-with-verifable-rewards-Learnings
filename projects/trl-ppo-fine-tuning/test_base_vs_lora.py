#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_base_vs_lora.py

Compare generations from a base HF chat model vs the same model + your LoRA adapters.
- Works great for Qwen/Qwen3 Instruct models (uses tokenizer.apply_chat_template).
- Runs prompts twice (base, then LoRA) to avoid double GPU memory usage.
- Saves CSV with prompt, base_output, lora_output.

Example:
  python compare_base_vs_lora.py ^
    --base-model Qwen/Qwen3-4B-Instruct-2507 ^
    --adapter-dir .\output ^
    --use-4bit ^
    --prompts "Explain GRPO in 2 sentences." "What is the capital of Japan?"

Or from a file:
  python compare_base_vs_lora.py ^
    --base-model Qwen/Qwen3-4B-Instruct-2507 ^
    --adapter-dir .\output ^
    --prompts-file .\prompts.txt
"""

import argparse
import csv
import os
import torch
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(
    base_model: str,
    use_4bit: bool,
    local_only: bool,
    attn_impl: str,
    dtype: torch.dtype,
):
    tok = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True, local_files_only=local_only
    )
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=local_only,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        quantization_config=bnb,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    return tok, model


def apply_chat(tok, prompt: str, device, disable_thinking: bool):
    # Qwen3 uses a chat template; we feed messages to get the correct formatting.
    messages = [{"role": "user", "content": prompt}]
    # Some Qwen3 templates support enable_thinking; if you want to disable, pass the flag.
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    if disable_thinking:
        # If the model’s chat template supports it, this will be honored; otherwise ignored.
        kwargs["enable_thinking"] = False
    inputs = tok.apply_chat_template(messages, **kwargs).to(device)
    return inputs


def generate(model, tok, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float, do_sample: bool, disable_thinking: bool):
    outs: List[str] = []
    for p in prompts:
        inputs = apply_chat(tok, p, model.device, disable_thinking)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        outs.append(text)
    return outs


def main():
    ap = argparse.ArgumentParser(description="Compare base vs LoRA-tuned outputs on the same prompts.")
    ap.add_argument("--base-model", required=True, help="HF model id or local path (e.g., Qwen/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--adapter-dir", required=True, help="Path to your saved LoRA adapters (e.g., .\\output)")
    ap.add_argument("--prompts", nargs="*", default=[], help="List of prompts (space-separated; quote each)")
    ap.add_argument("--prompts-file", default=None, help="Optional text file with one prompt per line")
    ap.add_argument("--out", default=r".\output\compare_base_vs_lora.csv", help="CSV output path")
    ap.add_argument("--use-4bit", action="store_true", help="Load both models in 4-bit")
    ap.add_argument("--local-only", action="store_true", help="Use local files only (offline)")
    ap.add_argument("--attn-impl", choices=["sdpa","eager"], default="sdpa")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2, help="Low temp for reproducible sanity checks")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling (off by default for determinism)")
    ap.add_argument("--disable-thinking", action="store_true", help="Try to disable Qwen 'thinking' if supported")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if not args.prompts and not args.prompts_file:
        # Default quick sanity set
        args.prompts = [
            "Explain GRPO in two sentences.",
            "List three safe ways to speed up PyTorch inference on Windows.",
            "You are given: 12 apples, you eat 5 and buy 4 more. How many now?",
            "Write a very short email subject for a job application follow-up.",
            "What's one surprising fact about the James Webb Space Telescope?",
        ]

    prompts = list(args.prompts)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend([ln.strip() for ln in f if ln.strip()])

    # Choose precision
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # -------- Pass 1: BASE --------
    tok, model = load_model_and_tokenizer(
        base_model=args.base_model,
        use_4bit=args.use_4bit,
        local_only=args.local_only,
        attn_impl=args.attn_impl,
        dtype=dtype,
    )
    base_outs = generate(
        model, tok, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample,
        disable_thinking=args.disable_thinking
    )
    del model
    torch.cuda.empty_cache()

    # -------- Pass 2: LoRA (same base + adapters) --------
    tok2, base2 = load_model_and_tokenizer(
        base_model=args.base_model,
        use_4bit=args.use_4bit,
        local_only=args.local_only,
        attn_impl=args.attn_impl,
        dtype=dtype,
    )
    lora_model = PeftModel.from_pretrained(base2, args.adapter_dir)
    lora_outs = generate(
        lora_model, tok2, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample,
        disable_thinking=args.disable_thinking
    )

    # Print a small table to console (truncated)
    def trunc(s, n=160):
        s = s.replace("\n", " ")
        return (s[:n] + "…") if len(s) > n else s

    print("\n=== COMPARISON (first 5) ===")
    for i, p in enumerate(prompts[:5]):
        print(f"\n[{i+1}] PROMPT: {p}")
        print("BASE:", trunc(base_outs[i]))
        print("LORA:", trunc(lora_outs[i]))

    # Save CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "base_output", "lora_output"])
        for p, b, l in zip(prompts, base_outs, lora_outs):
            w.writerow([p, b, l])
    print(f"\n[INFO] Wrote {len(prompts)} comparisons to: {args.out}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
