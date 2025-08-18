# -*- coding: utf-8 -*-
"""
Financial "Thinking" Fine-Tune on Gemma 3 270M: SFT + GRPO (Windows-friendly)

Highlights
- Multi-level reward:
  (1) strict format gate,
  (2) reasoning bundle (quality, logic, context),
  (3) FinBERT teacher alignment,
  (4) confidence calibration,
  (5) directional consistency.
- Clean prompts with hard contracts.
- Argparse overrides for datasets, steps, batches, lr, lengths, generations, beta, decoding, etc.
- Works with Unsloth 4-bit; optional bitsandbytes not required explicitly.

Tested with (approx): transformers≈4.55, trl≈0.14.x, datasets≈3.x, torch 2.7, Windows+CUDA.
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # safer on Windows
import argparse
import re
import gc
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import TextStreamer


# Optional cosmetics (not required)
try:
    import seaborn as sns
    SEABORN_OK = True
except Exception:
    SEABORN_OK = False

import matplotlib.pyplot as plt

# TRL
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

# --- FinBERT Teacher ----------------------------------------------------------
from transformers import (
    AutoTokenizer as HFAutoTokenizer,
    AutoModelForSequenceClassification,
)

class FinBERTTeacher:
    def __init__(self, device: Optional[str] = None):
        self.labels = ["negative", "neutral", "positive"]
        self.tok = HFAutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev).eval()

    @torch.no_grad()
    def predict_proba(self, text: str) -> Dict[str, float]:
        if not text:
            # equal uncertainty if blank
            return {k: 1.0 / len(self.labels) for k in self.labels}
        enc = self.tok(
            text[:1024],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        return dict(zip(self.labels, probs))

# --- Tokens & Prompts ---------------------------------------------------------
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SENTIMENT_START = "<SENTIMENT>"
SENTIMENT_END = "</SENTIMENT>"
CONFIDENCE_START = "<CONFIDENCE>"
CONFIDENCE_END = "</CONFIDENCE>"

SYSTEM_FINANCIAL_SFT = (
    "You are a financial analyst. Analyze sentiment with clear, balanced reasoning.\n"
    "OUTPUT CONTRACT (exactly this structure):\n"
    f"{REASONING_START} 2–3 concise sentences with both positives and negatives; use financial terms and connectives (because/however/therefore). {REASONING_END}\n"
    f"{SENTIMENT_START} one of: positive | negative | neutral {SENTIMENT_END}\n"
    f"{CONFIDENCE_START} a decimal between 0.1 and 1.0 {CONFIDENCE_END}\n"
    "Do not add any other sections or tags.\n"
    "\n"
    "Example:\n"
    f"{REASONING_START} Revenue growth is strong but margins compressed due to higher input costs; guidance is cautious, suggesting near-term volatility. Therefore, outlook is balanced with upside from new products. {REASONING_END}\n"
    f"{SENTIMENT_START} neutral {SENTIMENT_END}\n"
    f"{CONFIDENCE_START} 0.72 {CONFIDENCE_END}"
)

SYSTEM_FINANCIAL_GRPO = (
    "Follow the exact contract. Keep the reasoning compact and balanced.\n"
    f"{REASONING_START} ... {REASONING_END}\n"
    f"{SENTIMENT_START} positive|negative|neutral {SENTIMENT_END}\n"
    f"{CONFIDENCE_START} 0.1–1.0 {CONFIDENCE_END}"
)

# --- Config -------------------------------------------------------------------
@dataclass
class FinancialConfig:
    model_name: str = "unsloth/gemma-3-270m-it"
    max_seq_length: int = 512

    # SFT
    sft_epochs: int = 3
    sft_batch_size: int = 4
    sft_grad_accum: int = 2
    sft_lr: float = 1e-4
    sft_warmup: float = 0.1
    sft_weight_decay: float = 0.01

    # GRPO
    grpo_epochs: float = 2.0
    grpo_batch_size: int = 1
    grpo_grad_accum: int = 4
    grpo_lr: float = 1e-5
    grpo_warmup: float = 0.1
    grpo_weight_decay: float = 0.01
    num_generations: int = 6
    max_completion_length: int = 256
    max_prompt_length: int = 512
    beta: float = 0.15
    temperature: float = 0.7
    top_p: float = 0.9

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64

    # Data
    data_mode: str = "mixed"  # [mixed|real|synthetic]
    max_real_examples: int = 200
    min_total_examples: int = 20

# --- Synthetic fallback examples ---------------------------------------------
SYNTHETIC = [
    {
        "text": "Tech company reports 25% revenue growth but 8% profit decline due to R&D investment",
        "reasoning": "Revenue growth is strong, reflecting demand and expansion. Profit dipped due to R&D, which is a strategic cost with long-term upside. Near-term margins compress, but growth story remains intact.",
        "sentiment": "positive",
        "confidence": 0.75,
    },
    {
        "text": "Bank announces 3% dividend increase while facing regulatory scrutiny over compliance issues",
        "reasoning": "Dividend increase signals capital strength and shareholder focus. However, regulatory scrutiny introduces material risk, including fines or constraints; this offsets the positive signal.",
        "sentiment": "negative",
        "confidence": 0.80,
    },
    {
        "text": "Manufacturing firm reports stable earnings but warns of supply chain disruptions ahead",
        "reasoning": "Current results are stable, showing operational control. The forward warning raises uncertainty, with potential cost and delivery impacts. Overall, signals are mixed.",
        "sentiment": "neutral",
        "confidence": 0.65,
    },
]

# --- Dataset utilities --------------------------------------------------------
def load_phrasebank(split_cfg: str = "sentences_50agree", max_n: int = 200) -> List[Dict[str, Any]]:
    out = []
    try:
        ds = load_dataset("financial_phrasebank", split="train", name=split_cfg, trust_remote_code=True)
        n = min(max_n, len(ds))
        for ex in ds.select(range(n)):
            text = ex["sentence"]
            label = ex["label"]  # 0 neg, 1 neu, 2 pos
            if label == 0:
                rsn = "Text implies risks or deteriorating performance; signals likely weigh on valuation."
                sent = "negative"
                conf = 0.75
            elif label == 1:
                rsn = "Information is balanced; positives and negatives offset each other, implying a wait-and-see stance."
                sent = "neutral"
                conf = 0.60
            else:
                rsn = "Text implies improving fundamentals or favorable momentum; sentiment is constructive."
                sent = "positive"
                conf = 0.75
            out.append(
                {"text": text, "reasoning": rsn, "sentiment": sent, "confidence": conf, "source": "phrasebank"}
            )
    except Exception as e:
        print(f"[WARN] Could not load Financial PhraseBank: {e}")
    return out

def build_dataset(cfg: FinancialConfig, data_mode: str) -> Dataset:
    examples: List[Dict[str, Any]] = []
    if data_mode in ("mixed", "real"):
        real = load_phrasebank(max_n=cfg.max_real_examples)
        examples.extend(real)
    if data_mode in ("mixed", "synthetic") or (not examples):
        examples.extend(SYNTHETIC)
    ds = Dataset.from_list(examples)
    if len(ds) < cfg.min_total_examples:
        print(f"[WARN] Only {len(ds)} examples; below min_total_examples={cfg.min_total_examples}. Training will still run.")
    return ds

# --- Formatting for SFT / GRPO ------------------------------------------------
def format_sft(ex, tokenizer) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_FINANCIAL_SFT},
        {"role": "user", "content": f"Analyze the sentiment of this financial news:\n{ex['text']}"},
        {
            "role": "assistant",
            "content": (
                f"{REASONING_START}{ex['reasoning']}{REASONING_END}\n"
                f"{SENTIMENT_START}{ex['sentiment']}{SENTIMENT_END}\n"
                f"{CONFIDENCE_START}{ex['confidence']}{CONFIDENCE_END}"
            ),
        },
    ]
    enc = tokenizer.apply_chat_template(messages, tokenize=True)
    token_len = len(enc["input_ids"]) if isinstance(enc, dict) else len(enc)
    text_str = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"token_len": token_len, "text": text_str}

def format_grpo(ex, tokenizer) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_FINANCIAL_GRPO},
        {"role": "user", "content": f"Analyze the sentiment of this financial news:\n{ex['text']}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(prompt, truncation=False)["input_ids"]
    return {"prompt": prompt, "full_len": len(full_ids), "gold_text": ex["text"]}

# --- Light analyzers for the reasoning bundle --------------------------------
class FinancialReasoningAnalyzer:
    def __init__(self):
        self.financial_terms = ["revenue", "profit", "margin", "guidance", "debt", "cash", "capex", "dividend"]
        self.connectives = ["because", "however", "although", "while", "despite", "therefore", "thus"]
        self.context_terms = ["market", "sector", "industry", "trend", "environment", "macro", "near-term", "long-term"]

    def quality(self, txt: str) -> float:
        t = txt.lower()
        score = 0.0
        score += min(sum(1 for w in self.financial_terms if w in t) / 3.0, 1.0) * 0.4
        score += min(sum(1 for w in self.connectives if w in t) / 2.0, 1.0) * 0.3
        # balanced: contains both positive and negative cues
        pos = any(w in t for w in ["growth", "increase", "improve", "strong", "up"])
        neg = any(w in t for w in ["decline", "decrease", "worse", "weak", "down"])
        score += (0.3 if (pos and neg) else 0.15 if (pos or neg) else 0.0)
        return max(0.0, min(1.0, score))

    def logic(self, txt: str) -> float:
        t = txt.lower()
        # crude contradiction check
        contradictory = ("growth" in t and "decline" in t) or ("profit" in t and "loss" in t)
        score = 0.5 if not contradictory else 0.2
        if "therefore" in t or "thus" in t:
            score += 0.3
        if "mixed" in t or "uncertain" in t or "cautious" in t:
            score += 0.2
        return max(0.0, min(1.0, score))

    def context(self, txt: str) -> float:
        t = txt.lower()
        c = min(sum(1 for w in self.context_terms if w in t) / 3.0, 1.0) * 0.6
        timing = 0.4 if ("short-term" in t or "long-term" in t or "near-term" in t) else 0.2 if "future" in t else 0.0
        return max(0.0, min(1.0, c + timing))

# --- Reward helpers -----------------------------------------------------------
def extract_components(s: str) -> Dict[str, str]:
    def _grab(a, b):
        m = re.search(rf"{re.escape(a)}(.*?){re.escape(b)}", s, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    return {
        "reasoning": _grab(REASONING_START, REASONING_END),
        "sentiment": _grab(SENTIMENT_START, SENTIMENT_END).lower(),
        "confidence": _grab(CONFIDENCE_START, CONFIDENCE_END),
    }

def reward_format_gate(txt: str) -> float:
    need = [(REASONING_START, REASONING_END), (SENTIMENT_START, SENTIMENT_END), (CONFIDENCE_START, CONFIDENCE_END)]
    ok = all(txt.count(s) == 1 and txt.count(e) == 1 for s, e in need)
    return 1.0 if ok else 0.0

def parse_raw_text_from_prompt(prompt_str: str) -> str:
    key = "Analyze the sentiment of this financial news:\n"
    if key in prompt_str:
        return prompt_str.split(key, 1)[1].strip()
    return ""

def reward_confidence_calibration(p_teacher: float, p_model: float) -> float:
    # Brier-like: 1 - (gap^2), clamp [0,1]
    try:
        return max(0.0, 1.0 - float(p_model - p_teacher) ** 2)
    except Exception:
        return 0.0

def reward_directional(text: str, reasoning: str, sentiment: str) -> float:
    t = reasoning.lower()
    pos = any(w in t for w in ["increase", "growth", "improve", "up", "higher"])
    neg = any(w in t for w in ["decrease", "decline", "worse", "down", "lower"])
    if pos and neg and sentiment == "neutral":
        return 1.0
    if pos and not neg and sentiment == "positive":
        return 1.0
    if neg and not pos and sentiment == "negative":
        return 1.0
    return 0.0

# Robust reward wrapper that matches TRL call orders across versions
def make_rewards(analyzer: FinancialReasoningAnalyzer, teacher: FinBERTTeacher):
    def reward_gate(prompts=None, completions=None, **kwargs):
        # Accept either (completions, **kwargs) or (prompts, completions, **kwargs)
        comp_list = completions if completions is not None else kwargs.get("completions") or prompts
        return [reward_format_gate(str(c)) for c in comp_list]

    def reward_finance(prompts=None, completions=None, **kwargs):
        # Normalize args
        pr = prompts
        comp_list = completions
        if comp_list is None:
            # Some TRL versions pass (completions, **kwargs)
            comp_list = pr
            pr = kwargs.get("prompts", [])
        if pr is None:
            pr = []
        pr = list(pr)
        comp_list = list(comp_list)

        scores = []
        for p, c in zip(pr, comp_list):
            txt = str(c)
            gate = reward_format_gate(txt)
            if gate == 0.0:
                scores.append(0.0)
                continue

            comp = extract_components(txt)
            sent = comp["sentiment"]
            try:
                conf = float(comp["confidence"])
            except Exception:
                conf = 0.0

            raw = parse_raw_text_from_prompt(str(p))
            probs = teacher.predict_proba(raw)
            p_teacher = float(probs.get(sent, 0.0))

            r_q = analyzer.quality(comp["reasoning"])
            r_l = analyzer.logic(comp["reasoning"])
            r_c = analyzer.context(comp["reasoning"])
            r_reason = 0.5 * r_q + 0.3 * r_l + 0.2 * r_c

            r_sent = p_teacher
            r_cal = reward_confidence_calibration(p_teacher, conf)
            r_dir = reward_directional(raw, comp["reasoning"], sent)

            total = (0.35 * r_sent) + (0.25 * r_reason) + (0.20 * r_cal) + (0.15 * r_dir)
            scores.append(float(gate * total))
        return scores

    return reward_gate, reward_finance

# --- Plot utility -------------------------------------------------------------
def plot_grpo_metrics(log_history: List[dict]) -> None:
    if not log_history:
        print("No GRPO log history to plot.")
        return
    if SEABORN_OK:
        plt.style.use("seaborn-v0_8")
        sns.set_palette("pastel")

    steps, losses, rewards, kls = [], [], [], []
    for log in log_history:
        if "step" in log and "loss" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
            rewards.append(log.get("reward", None))
            kls.append(log.get("kl", None))

    plt.figure(figsize=(11, 3))
    plt.subplot(1, 3, 1); plt.plot(steps, losses, marker="x"); plt.title("Policy Loss"); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2); plt.plot(steps, rewards, marker="x"); plt.title("Total Reward"); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 3); plt.plot(steps, kls, marker="x"); plt.title("KL Penalty"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# --- Main pipeline ------------------------------------------------------------
def main(args):
    cfg = FinancialConfig(
        model_name=args.base_model or "unsloth/gemma-3-270m-it",
        max_seq_length=args.max_prompt_length or 512,
        sft_epochs=args.sft_epochs,
        sft_batch_size=args.sft_batch,
        sft_grad_accum=args.sft_grad_accum,
        sft_lr=args.sft_lr,
        sft_warmup=args.sft_warmup,
        sft_weight_decay=args.sft_weight_decay,
        grpo_epochs=args.grpo_epochs,
        grpo_batch_size=args.grpo_batch,
        grpo_grad_accum=args.grpo_grad_accum,
        grpo_lr=args.grpo_lr,
        grpo_warmup=args.grpo_warmup,
        grpo_weight_decay=args.grpo_weight_decay,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        data_mode=args.data_mode,
        max_real_examples=args.max_real_examples,
        min_total_examples=args.min_total_examples,
    )

    print("🚀 Financial Thinking Pipeline — SFT + GRPO")
    print(f"Base model: {cfg.model_name}")
    print(f"Data mode: {cfg.data_mode}")

    # Load model/tokenizer via Unsloth (4-bit optional)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=args.use_4bit,
        fast_inference=False,
        max_lora_rank=cfg.lora_rank,
        local_files_only=args.local_only,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        random_state=123,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --------------------- SFT ---------------------
    print("\n[Phase 1] Supervised Fine-Tuning (SFT)")
    if args.train_jsonl:
        raw_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
        # Expect fields: text, reasoning, sentiment, confidence
        if not all(k in raw_ds.column_names for k in ["text", "reasoning", "sentiment", "confidence"]):
            raise ValueError("Custom SFT JSONL must contain fields: text, reasoning, sentiment, confidence")
        base_ds = raw_ds
    else:
        base_ds = build_dataset(cfg, data_mode=cfg.data_mode)

    sft_ds = base_ds.map(lambda ex: format_sft(ex, tokenizer), remove_columns=base_ds.column_names)
    sft_ds = sft_ds.filter(lambda ex: ex["token_len"] <= cfg.max_seq_length)
    # cap if desired (keep all by default)
    if args.sft_limit > 0:
        sft_ds = sft_ds.select(range(min(args.sft_limit, len(sft_ds))))

    sft_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "sft"),
        seed=123,
        do_train=True,
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=cfg.sft_batch_size,
        gradient_accumulation_steps=cfg.sft_grad_accum,
        learning_rate=cfg.sft_lr,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.sft_warmup,
        weight_decay=cfg.sft_weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="none",
        dataset_num_proc=1,
    )
    sft_trainer = SFTTrainer(model=model, args=sft_args, train_dataset=sft_ds, tokenizer=tokenizer)
    sft_trainer.train()

    del sft_trainer, sft_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------- GRPO ---------------------
    print("\n[Phase 2] GRPO (RL with multi-level rewards)")
    if args.eval_jsonl:
        raw_grpo = load_dataset("json", data_files=args.eval_jsonl, split="train")
        if "text" not in raw_grpo.column_names:
            raise ValueError("Custom GRPO JSONL must contain 'text' field.")
        grpo_src = raw_grpo
    else:
        grpo_src = build_dataset(cfg, data_mode=cfg.data_mode)

    grpo_ds = grpo_src.map(lambda ex: format_grpo(ex, tokenizer), remove_columns=grpo_src.column_names)
    if args.grpo_limit > 0:
        grpo_ds = grpo_ds.select(range(min(args.grpo_limit, len(grpo_ds))))

    analyzer = FinancialReasoningAnalyzer()
    teacher = FinBERTTeacher()
    r_gate, r_fin = make_rewards(analyzer, teacher)

    grpo_args = GRPOConfig(
        seed=123,
        do_train=True,
        num_train_epochs=cfg.grpo_epochs,
        per_device_train_batch_size=cfg.grpo_batch_size,
        gradient_accumulation_steps=cfg.grpo_grad_accum,
        learning_rate=cfg.grpo_lr,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.grpo_warmup,
        weight_decay=cfg.grpo_weight_decay,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="none",
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        beta=cfg.beta,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=grpo_ds,
            args=grpo_args,
            reward_funcs=[r_gate, r_fin],
        )
    except TypeError:
        grpo_trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=grpo_ds,
            args=grpo_args,
            reward_funcs=[r_gate, r_fin],
        )
    grpo_trainer.train()

    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"VRAM peak reserved: {used_mem} GB")

    plot_grpo_metrics(grpo_trainer.state.log_history)

    # --------------------- Quick sanity inference ---------------------
    print("\n[Sanity Check] Generation samples")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [
        "Energy company reports 30% production increase but faces environmental lawsuit",
        "Software firm announces major acquisition while reporting 5% decline in quarterly revenue",
        "Bank reports record profits but warns of potential regulatory changes affecting lending",
    ]
    for i, s in enumerate(samples):
        messages = [
            {"role": "system", "content": SYSTEM_FINANCIAL_GRPO},
            {"role": "user", "content": f"Analyze the sentiment of this financial news:\n{s}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        toks = tokenizer(text, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        streamer = TextStreamer(tokenizer, skip_prompt=False)
        _ = model.generate(
            **toks,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_completion_length,
            streamer=streamer,
            do_sample=True,
        )
        print("\n" + "=" * 60)

    # Save final
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"\n✔ Saved final model to {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Financial Thinking Model: SFT + GRPO on Gemma 3 270M")

    # Model & IO
    ap.add_argument("--base-model", type=str, default="unsloth/gemma-3-270m-it")
    ap.add_argument("--output-dir", type=str, default="financial_reasoning_improved-outputs")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--use-4bit", action="store_true", default=True)

    # Data control
    ap.add_argument("--data-mode", choices=["mixed", "real", "synthetic"], default="mixed",
                    help="Use Financial PhraseBank (real) and/or synthetic fallbacks.")
    ap.add_argument("--max-real-examples", type=int, default=200)
    ap.add_argument("--min-total-examples", type=int, default=20)
    ap.add_argument("--train-jsonl", type=str, default=None,
                    help="Custom SFT JSONL with fields: text, reasoning, sentiment, confidence")
    ap.add_argument("--eval-jsonl", type=str, default=None,
                    help="Custom GRPO JSONL with field: text")
    ap.add_argument("--sft-limit", type=int, default=0, help="Limit SFT examples (0 = all)")
    ap.add_argument("--grpo-limit", type=int, default=0, help="Limit GRPO examples (0 = all)")

    # SFT knobs
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--sft-batch", type=int, default=12)
    ap.add_argument("--sft-grad-accum", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--sft-warmup", type=float, default=0.1)
    ap.add_argument("--sft-weight-decay", type=float, default=0.01)

    # GRPO knobs
    ap.add_argument("--grpo-epochs", type=float, default=4.0)
    ap.add_argument("--grpo-batch", type=int, default=12)
    ap.add_argument("--grpo-grad-accum", type=int, default=4)
    ap.add_argument("--grpo-lr", type=float, default=1e-5)
    ap.add_argument("--grpo-warmup", type=float, default=0.1)
    ap.add_argument("--grpo-weight-decay", type=float, default=0.01)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    # LoRA parameters
    ap.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for parameter-efficient fine-tuning")
    ap.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling factor")

    # Logging
    ap.add_argument("--logging-steps", type=int, default=10)

    args = ap.parse_args()
    main(args)
