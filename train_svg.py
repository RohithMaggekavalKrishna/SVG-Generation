"""
train_svg.py — Fine-tune Qwen2.5-7B on train.csv for Text-to-SVG generation.

Usage (see submit_svg_sft.sh for SLURM wrapper):
    python train_svg.py --train-csv data/train.csv --output-dir /scratch/hk4488/SVG-Generation/outputs/svg_sft
"""

import unsloth  # noqa: F401 — must be imported first for Unsloth optimizations

import argparse
import os
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# ── Constants ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SVG code generator. "
    "Given a text description, output only valid SVG markup. "
    "Rules:\n"
    "- Root element: <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"256\" height=\"256\" viewBox=\"0 0 256 256\">\n"
    "- Use only these elements: svg, g, path, rect, circle, ellipse, line, polyline, polygon, "
    "defs, use, symbol, clipPath, mask, linearGradient, radialGradient, stop, text, tspan, "
    "title, desc, style, pattern, marker, filter\n"
    "- No scripts, no animation, no event handlers, no external references\n"
    "- Maximum 16000 characters, maximum 256 path elements\n"
    "- Match the visual complexity of the description — do not over-simplify or over-elaborate\n"
    "- Output only the SVG code, nothing else"
)

ALLOWED_TAGS = {
    "svg", "g", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon", "defs", "use",
    "symbol", "clippath", "mask", "lineargradient",
    "radialgradient", "stop", "text", "tspan", "title",
    "desc", "style", "pattern", "marker", "filter",
}

DISALLOWED_TAGS = {
    "script", "animate", "animatetransform", "animatemotion",
    "animatecolor", "set", "foreignobject",
}

FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
    '<rect x="0" y="0" width="256" height="256" fill="white"/>'
    '<circle cx="128" cy="128" r="80" fill="#4A90D9"/>'
    '</svg>'
)

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


# ── SVG helpers ────────────────────────────────────────────────────────────────

def is_valid_svg(svg_text: str) -> bool:
    if not svg_text or len(svg_text) > 16000:
        return False
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return False
    # Check root is svg
    if not root.tag.lower().endswith("svg"):
        return False
    # Walk all elements
    path_count = 0
    for elem in root.iter():
        local = elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower()
        if local in DISALLOWED_TAGS:
            return False
        if local not in ALLOWED_TAGS:
            return False
        # No event handlers
        for attr in elem.attrib:
            if attr.lower().startswith("on"):
                return False
        # No external refs
        for attr in ("href", "{http://www.w3.org/1999/xlink}href"):
            val = elem.attrib.get(attr, "")
            if val.startswith("http") or val.startswith("//"):
                return False
        if local == "path":
            path_count += 1
    if path_count > 256:
        return False
    return True


def clean_svg(svg_text: str) -> str:
    """Ensure SVG has required attributes and is within limits."""
    if len(svg_text) > 16000:
        svg_text = svg_text[:16000]
    # Ensure xmlns
    if 'xmlns=' not in svg_text:
        svg_text = svg_text.replace(
            "<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1
        )
    # Ensure width/height/viewBox
    for attr, val in [
        ('width="256"', 'width="256"'),
        ('height="256"', 'height="256"'),
        ('viewBox="0 0 256 256"', 'viewBox="0 0 256 256"'),
    ]:
        if attr.split('=')[0] not in svg_text:
            svg_text = svg_text.replace("<svg", f"<svg {val}", 1)
    return svg_text


# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_csv(path: str, max_samples: int = None) -> Dataset:
    df = pd.read_csv(path)

    # Detect column names flexibly
    prompt_col = next((c for c in df.columns if c.lower() in ("prompt", "description", "text", "caption")), None)
    svg_col = next((c for c in df.columns if c.lower() in ("svg", "svg_code", "output", "completion")), None)

    if prompt_col is None or svg_col is None:
        raise ValueError(
            f"Could not detect prompt/svg columns. Found: {list(df.columns)}"
        )

    print(f"Using columns: prompt='{prompt_col}', svg='{svg_col}'")

    df = df[[prompt_col, svg_col]].rename(columns={prompt_col: "prompt", svg_col: "svg"})
    df = df.dropna(subset=["prompt", "svg"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["svg"] = df["svg"].astype(str).str.strip()

    # Keep only rows where svg starts with <svg
    df = df[df["svg"].str.lower().str.startswith("<svg")]
    # Keep valid SVGs only
    df = df[df["svg"].apply(is_valid_svg)]

    print(f"Rows after filtering: {len(df)}")

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {max_samples} rows")

    return Dataset.from_pandas(df[["prompt", "svg"]], preserve_index=False)


def format_sft_example(example):
    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['svg']}<|im_end|>"
    )
    return {"text": text}


# ── Training ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/train.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--max-seq-length", type=int, default=6144)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-size", type=float, default=0.02)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.train_csv}")
    ds = load_train_csv(args.train_csv, max_samples=args.max_train_samples)

    splits = ds.train_test_split(test_size=args.eval_size, seed=args.seed)
    train_ds = splits["train"].map(format_sft_example, remove_columns=["prompt", "svg"])
    eval_ds = splits["test"].map(format_sft_example, remove_columns=["prompt", "svg"])

    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    print("Sample text (first 300 chars):")
    print(train_ds[0]["text"][:300])

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    # TRL 0.10+ requires SFTConfig (superset of TrainingArguments) for SFT-specific params
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.seed,
        # SFT-specific params (TRL 0.24+: max_length, not max_seq_length)
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("Starting training...")
    result = trainer.train()
    print(f"Training complete. Loss: {result.training_loss:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save a clean lora-only copy
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to: {lora_dir}")


if __name__ == "__main__":
    main()
