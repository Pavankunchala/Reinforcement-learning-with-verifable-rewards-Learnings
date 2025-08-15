#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_windows_setup_and_train.py  — TRL GRPO (datasets v3+), Windows-friendly

Fixes:
- Use `num_generations` (no `group_size` in GRPOConfig)
- Send `attn_implementation` to model.from_pretrained, not GRPOConfig
- Dynamic HF dataset split/column mapping; offline via DownloadConfig
Tested combos per HF cookbook: transformers≈4.47–4.48, trl≈0.14.x, datasets≈3.2.x. :contentReference[oaicite:3]{index=3}
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    get_dataset_config_names,
    get_dataset_split_names,
    DownloadConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

# ---------- helpers: precision ----------
def bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

def choose_precision(args) -> Dict[str, bool]:
    if args.fp16:
        return {"fp16": True, "bf16": False}
    if args.bf16:
        return {"fp16": False, "bf16": True}
    if bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}

# ---------- helpers: prompt mapping ----------
PROMPT_FIELDS_PRIMARY = [
    "prompt","question","query","instruction","text","input","context","ctx","document","source",
]
CHAT_FIELDS = ["messages","conversations","dialog","chat"]
REFERENCE_FIELDS = ["reference","response","answer","output","label","target","gold","completion","chosen"]

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def messages_to_prompt(msgs: Any) -> Optional[str]:
    try:
        if not isinstance(msgs, list) or not msgs:
            return None
        lines = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(p.get("text","")) if isinstance(p, dict) else str(p) for p in content)
                lines.append(f"{role}: {str(content).strip()}")
            else:
                lines.append(str(m))
        return "\n".join(lines).strip()
    except Exception:
        return None

def synthesize_prompt(ex: Dict[str, Any]) -> str:
    for k in PROMPT_FIELDS_PRIMARY:
        if k in ex and ex[k]:
            return str(ex[k])
    for k in CHAT_FIELDS:
        if k in ex and ex[k]:
            p = messages_to_prompt(ex[k])
            if p:
                return p
    instr = str(ex.get("instruction","")).strip()
    inp   = str(ex.get("input","")).strip()
    if instr and inp: return f"{instr}\n\nInput: {inp}"
    if instr: return instr
    if inp:   return inp
    ctx = str(ex.get("context","")).strip()
    q   = str(ex.get("question","")).strip()
    if ctx and q: return f"{ctx}\n\nQuestion: {q}"
    if q: return q
    try:
        return json.dumps(ex, ensure_ascii=False)
    except Exception:
        return str(ex)

def pick_reference(ex: Dict[str, Any]) -> Optional[str]:
    for k in REFERENCE_FIELDS:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    if "rejected" in ex and ex.get("chosen"):
        return str(ex["chosen"])
    return None

def map_to_prompt_reference(ds: Dataset) -> Dataset:
    def mapper(ex):
        return {"prompt": synthesize_prompt(ex), "reference": pick_reference(ex)}
    mapped = ds.map(mapper)
    for col in list(mapped.column_names):
        if col not in {"prompt","reference"}:
            try:
                mapped = mapped.remove_columns(col)
            except Exception:
                pass
    return mapped

# ---------- reward ----------
def _to_text(c: Any) -> str:
    # Handle plain text or chat-style structures
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # list of messages [{"role":..., "content":...}, ...]
        parts = []
        for m in c:
            if isinstance(m, dict):
                parts.append(str(m.get("content","")))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(c, dict):
        return str(c.get("content", c.get("text","")))
    return str(c)

def extract_number(s: str) -> Optional[str]:
    m = re.search(r"(?:####\s*)?(-?\d+(?:\.\d+)?)\s*$", (s or "").strip())
    return m.group(1) if m else None

def reward_function(completions: List[Any], references: Optional[List[Optional[str]]] = None, **kwargs) -> List[float]:
    rewards: List[float] = []
    for i, comp in enumerate(completions):
        txt = _to_text(comp) or ""
        r = 0.0
        # length heuristic (keep answers compact-ish)
        tokens_est = max(1, len(txt) // 4)
        if tokens_est <= 256: r += 0.2
        elif tokens_est <= 512: r += 0.1
        else: r -= 0.2
        # formatting bonus
        if re.search(r"(final answer|answer)\s*[:\-]", txt, flags=re.I):
            r += 0.2
        # reference-based
        ref = references[i] if references is not None and i < len(references) else None
        if ref:
            if normalize_text(txt) == normalize_text(ref):
                r += 1.0
            ns, nr = extract_number(txt), extract_number(ref)
            if ns is not None and nr is not None and ns == nr:
                r += 0.6
        # boilerplate penalty
        if "as an ai" in normalize_text(txt):
            r -= 0.3
        rewards.append(float(r))
    return rewards

# ---------- split auto-detection ----------
PREFERRED_TRAIN_SPLITS = ["train_sft","train","train_gen","train_prefs","train_all"]
PREFERRED_EVAL_SPLITS  = ["test_sft","validation","valid","test","test_gen","test_prefs"]

def pick_splits_from_dataset_dict(dd: DatasetDict) -> Tuple[str, Optional[str]]:
    keys = set(dd.keys())
    train = next((k for k in PREFERRED_TRAIN_SPLITS if k in keys), None)
    if train is None:
        train = next((k for k in keys if "train" in k), next(iter(keys)))
    eval_split = next((k for k in PREFERRED_EVAL_SPLITS if k in keys), None)
    if eval_split is None:
        for k in keys:
            if (("test" in k) or ("valid" in k)) and k != train:
                eval_split = k
                break
    if eval_split == train:
        eval_split = None
    return train, eval_split

def safe_load_dataset(name: str, cfg: Optional[str], split: Optional[str], local_only: bool) -> Union[Dataset, DatasetDict]:
    dlc = DownloadConfig(local_files_only=local_only) if local_only else None
    if split:
        return load_dataset(name, cfg, split=split, download_config=dlc)
    return load_dataset(name, cfg, download_config=dlc)

def auto_load_hf_splits(name: str, cfg: Optional[str], local_only: bool) -> Tuple[Dataset, Optional[Dataset], str, Optional[str]]:
    try:
        dd_or_ds = safe_load_dataset(name, cfg, None, local_only)
        if isinstance(dd_or_ds, DatasetDict):
            train_split, eval_split = pick_splits_from_dataset_dict(dd_or_ds)
            train_ds = map_to_prompt_reference(dd_or_ds[train_split])
            eval_ds  = map_to_prompt_reference(dd_or_ds[eval_split]) if (eval_split and eval_split in dd_or_ds) else None
            return train_ds, eval_ds, train_split, eval_split
        else:
            return map_to_prompt_reference(dd_or_ds), None, "train", None
    except Exception:
        try:
            dlc = DownloadConfig(local_files_only=local_only) if local_only else None
            use_cfg = cfg or (get_dataset_config_names(name, download_config=dlc)[0] if get_dataset_config_names(name, download_config=dlc) else None)
            splits = get_dataset_split_names(name, use_cfg, download_config=dlc)
            train_choice = next((s for s in PREFERRED_TRAIN_SPLITS if s in splits), None) or next((s for s in splits if "train" in s), splits[0])
            eval_choice  = next((s for s in PREFERRED_EVAL_SPLITS if s in splits), None)
            train_ds = map_to_prompt_reference(safe_load_dataset(name, use_cfg, train_choice, local_only))  # type: ignore
            eval_ds  = map_to_prompt_reference(safe_load_dataset(name, use_cfg, eval_choice,  local_only)) if eval_choice else None  # type: ignore
            return train_ds, eval_ds, train_choice, eval_choice
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect splits for dataset '{name}': {e}")

# ---------- main ----------
def maybe_login_hf(token: Optional[str]):
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=True)
        print("[INFO] HF token set.")
    except Exception as e:
        print(f"[WARN] HF login failed (continuing): {e}")

def main():
    ap = argparse.ArgumentParser(description="Local GRPO (TRL) on Windows with HF models/datasets")
    # model & data
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--eval-jsonl", default=None)
    ap.add_argument("--output-dir", required=True)

    # training
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # GRPO sampling
    ap.add_argument("--num-generations", type=int, default=4, help="completions per prompt (group size)")
    ap.add_argument("--num-iterations", type=int, default=1, help="policy updates per generation batch")

    # system / precision
    ap.add_argument("--attn-impl", default="sdpa", choices=["eager","sdpa"])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--local-only", action="store_true")

    # LoRA / quant
    ap.add_argument("--lora-off", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--use-4bit", action="store_true")

    # hub
    ap.add_argument("--hub-token", default=None)

    args = ap.parse_args()

    # env/perf
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    maybe_login_hf(args.hub_token or os.getenv("HF_TOKEN"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, trust_remote_code=True, local_files_only=args.local_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit optional
    quant_cfg = None
    if args.use_4bit:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            print(f"[WARN] bitsandbytes unavailable/incompatible: {e}")
            quant_cfg = None

    # precision + model kwargs (send attn_implementation here)
    prec = choose_precision(args)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if prec["bf16"] else torch.float16,
        trust_remote_code=True,
        local_files_only=args.local_only,
        attn_implementation=args.attn_impl,
    )
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": 0} if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    try:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    # LoRA
    if not args.lora_off:
        targets = [m.strip() for m in args.lora_target.split(",") if m.strip()]
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                              target_modules=targets, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # data
    if args.dataset:
        train_ds, eval_ds, tname, ename = auto_load_hf_splits(args.dataset, args.dataset_config, args.local_only)
        print(f"[INFO] Using dataset='{args.dataset}' config='{args.dataset_config}'")
        print(f"[INFO] Picked train split: {tname} | eval split: {ename}")
    elif args.train_jsonl:
        train_ds = map_to_prompt_reference(load_dataset("json", data_files=args.train_jsonl, split="train"))
        eval_ds  = map_to_prompt_reference(load_dataset("json", data_files=args.eval_jsonl,  split="train")) if args.eval_jsonl else None
        print(f"[INFO] Using local JSONL. Train: {args.train_jsonl} | Eval: {args.eval_jsonl}")
    else:
        raise ValueError("Provide either --dataset or --train-jsonl.")


# ↓ add this
    N = 100
    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(N, len(train_ds))))
    if eval_ds is not None:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(N, len(eval_ds))))
    # reward refs (optional)
    reward_refs = train_ds["reference"] if "reference" in train_ds.column_names else None
    def _wrapped_reward(completions: List[Any], **kwargs) -> List[float]:
        return reward_function(completions, reward_refs, **kwargs)

    # GRPO args — note: no `group_size` in GRPOConfig; use `num_generations`
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        report_to=[],
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        fp16=prec["fp16"],
        bf16=prec["bf16"],
        # GRPO-specific
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_gen_len,
        num_generations=args.num_generations,   # ← controls group size
        num_iterations=args.num_iterations,
        temperature=0.7,
        top_p=0.9,
        # do_sample=True,
    )

    trainer = GRPOTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[_wrapped_reward],
        # dataset_text_field="prompt",
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    try: tokenizer.save_pretrained(args.output_dir)
    except Exception: pass
    print("[INFO] Done. Artifacts in:", args.output_dir)

if __name__ == "__main__":
    main()
