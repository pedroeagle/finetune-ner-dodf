#!/usr/bin/env python3
"""Fine-tune Qwen3-4B on the DODF/UnB-KnEDLe corpus with LoRA or DoRA.

Uses TRL SFTTrainer + PEFT LoraConfig. QLoRA (NF4) is enabled with --use-qlora.
The adapter is saved in PEFT format (adapter_config.json + adapter_model.safetensors).

Usage:
  python experiments/train.py --adapter lora --rank 8
  python experiments/train.py --adapter dora --rank 8
  python experiments/train.py --adapter lora --rank 8 --use-qlora
  python experiments/train.py --adapter lora --rank 4   # rank sweep (paper: r in {4,8,16})
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"
RESULTS_DIR = ROOT / "results"

CHATML_TMPL = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_as_chat(record: dict, tokenizer) -> str:
    # instruction_fmt = task statement + output-format description (build_dataset.py);
    # the same prompt used at evaluation time.
    instruction = record["instruction_fmt"]
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{record['input']}"},
        {"role": "assistant", "content": record["output"]},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except Exception:
        return CHATML_TMPL.format(
            user=f"{instruction}\n\n{record['input']}",
            assistant=record["output"],
        )


def build_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    return Dataset.from_list(
        [{"text": format_as_chat(r, tokenizer)} for r in records]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Base model (HuggingFace hub id or local path)")
    parser.add_argument("--adapter", choices=["lora", "dora"], default="lora",
                        help="Adaptation technique: LoRA or DoRA (paper section 4.4)")
    parser.add_argument("--rank", type=int, default=8, choices=[4, 8, 16],
                        help="Rank r — the paper sweeps {4, 8, 16}")
    parser.add_argument("--use-qlora", action="store_true",
                        help="Quantize the base model to NF4 to reduce VRAM (QLoRA)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation. Effective batch = batch-size × grad-accum × "
                             "world_size (number of GPUs); the reported runs used 8 GPUs → 4×4×8 = 128")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (paper: 2e-4, linear scheduler)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum context length in tokens (input + BIO output)")
    parser.add_argument("--output-dir", default=None,
                        help="Adapter output directory (default: results/checkpoints/…)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (data shuffle + init). Default 42 = original "
                             "run. Use other seeds to estimate variance (H2).")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="Fraction of the training set (0<f≤1) for the data curve. Default "
                             "1.0 = all 87k. Deterministic subsampling by --seed.")
    args = parser.parse_args()

    if not (0.0 < args.train_frac <= 1.0):
        parser.error("--train-frac must be in (0, 1]")

    alpha = args.rank * 2  # α/r = 2 (paper section 4.4; original LoRA convention)

    # Suffixes only when off-default, to avoid colliding with the canonical run (seed 42, frac 1.0).
    seed_suffix = "" if args.seed == 42 else f"_s{args.seed}"
    frac_suffix = "" if args.train_frac >= 1.0 else f"_f{int(round(args.train_frac * 100))}"
    qlora_suffix = "_qlora" if args.use_qlora else ""
    out_dir = args.output_dir or str(
        RESULTS_DIR / f"checkpoints/{args.adapter}_r{args.rank}_fmt{qlora_suffix}{frac_suffix}{seed_suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- NF4 quantization (QLoRA) ----
    bnb_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    # ---- base model ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    model_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if args.use_qlora:
        model_kwargs["quantization_config"] = bnb_config
    else:
        # bf16, not fp16: fp16 on ROCm MI250 silently produces NaN in the adapter weights.
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- LoRA / DoRA ----
    # target_modules=["q_proj","v_proj"] follows the original LoRA paper (Hu et al. 2022)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        use_dora=(args.adapter == "dora"),
        bias="none",
    )

    # ---- datasets ----
    train_records = load_jsonl(FT_DIR / "train.jsonl")
    dev_records = load_jsonl(FT_DIR / "dev.jsonl")

    # Data curve: deterministically subsample a fraction of the training set (RNG by --seed).
    if args.train_frac < 1.0:
        import random as _random
        n_keep = max(1, int(round(len(train_records) * args.train_frac)))
        _rng = _random.Random(args.seed)
        idx = _rng.sample(range(len(train_records)), n_keep)
        train_records = [train_records[i] for i in idx]
        print(f"Data curve: frac={args.train_frac} → {n_keep:,} training examples")

    train_ds = build_hf_dataset(train_records, tokenizer)
    dev_ds = build_hf_dataset(dev_records, tokenizer)

    print(f"Adapter    : {args.adapter.upper()} | r={args.rank} | α={alpha}")
    print(f"QLoRA      : {'yes (NF4)' if args.use_qlora else 'no'}")
    print(f"Train      : {len(train_ds):,} examples")
    print(f"Validation : {len(dev_ds):,} examples")
    print(f"Output     : {out_dir}")

    # ---- SFTTrainer ----
    sft_config = SFTConfig(
        output_dir=out_dir,
        # Truncate input + BIO output. Without this, SFTConfig uses its default of 1024
        # and cuts ~4% of the examples (the tokenizer's model_max_length only controls the warning).
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        optim="adamw_torch",
        bf16=True,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataset_text_field="text",
        seed=args.seed,
        data_seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    checkpoints = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(checkpoints[-1]) if checkpoints else None
    trainer.train(resume_from_checkpoint=resume)

    # Save the PEFT adapter (adapter_config.json records the base model for later loading).
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nAdapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
