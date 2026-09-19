#!/usr/bin/env python3
"""Evaluate generative NER: inline BIO output → seqeval → metrics (H1).

Supports distributed inference via torchrun: each process loads the full model on
its GPU and handles 1/N of the dataset; rank 0 aggregates and computes metrics.

Usage:
  # fine-tuned model (PEFT adapter detected automatically)
  torchrun --nproc_per_node=8 experiments/evaluate_ner.py --model results/checkpoints/lora_r8

  # zero-shot baseline
  torchrun --nproc_per_node=8 experiments/evaluate_ner.py --model Qwen/Qwen3-4B

  # compare two results (ΔF1 + bootstrap) — runs locally without a GPU
  python experiments/evaluate_ner.py \\
      --compare results/ner_lora_r8_fmt_test.json results/ner_Qwen3-4B_fmt_test.json

The expected output format is written into the dataset (field instruction_fmt, see
scripts/build_dataset.py) and used in the query; results carry the _fmt suffix
(e.g., ner_lora_r8_fmt_test.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import torch
import torch._dynamo
torch._dynamo.disable()

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"
RESULTS_DIR = ROOT / "results"

_UNIT_RE = re.compile(r"^(.+)\(([BI]-[\w]+|O)\)$")
CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
CHATML_PAIR = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>\n"
)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """Initialize NCCL when running via torchrun. Returns (rank, world_size, local_rank)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(hours=4),
        )
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, device: torch.device,
                             dtype: torch.dtype = torch.bfloat16):
    """Load base (+ adapter, if any) with a uniform dtype across base and fine-tuned
    (fair H1 comparison). Uses torch_dtype= (dtype= is ignored by older versions)."""
    adapter_cfg = Path(model_path) / "adapter_config.json"

    if adapter_cfg.exists():
        from peft import PeftModel

        with open(adapter_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        print(f"PEFT adapter detected. Base: {base} | dtype={dtype}")

        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)
        model = PeftModel.from_pretrained(model, model_path)
        # The adapter may be saved in fp32; unify base+adapter to the same dtype.
        model = model.to(dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device).to(dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inline GNER format parsing
# ---------------------------------------------------------------------------

def parse_inline_bio(text: str) -> tuple[list[str], list[str]]:
    """Convert 'tok1(B-etype) tok2(I-etype) tok3(O) …' into (tokens, labels)."""
    tokens, labels = [], []
    for unit in text.split():
        m = _UNIT_RE.match(unit)
        if m:
            tokens.append(m.group(1))
            labels.append(m.group(2))
    return tokens, labels


def align_to_gold(pred_labels: list[str], gold_len: int) -> list[str]:
    if len(pred_labels) >= gold_len:
        return pred_labels[:gold_len]
    return pred_labels + ["O"] * (gold_len - len(pred_labels))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_few_shot_demos(n: int, seed: int = 42) -> dict[str, list[dict]]:
    """Select up to n demos per (act, entity_type) from the training split.

    Returns dict["act/entity_type" → list[record]]. At inference, each test example
    gets demos from its own (act, entity_type) — the model sees the output format and
    the exact labels for that entity in that act. They come from the training set
    (never the test) to avoid leakage. Deterministic by seed.
    """
    recs = []
    with open(FT_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))

    by_key: dict[str, list[dict]] = defaultdict(list)
    for rec in recs:
        if rec.get("is_negative"):
            continue  # negative demos teach "there is no entity" — exclude
        key = f"{rec['act']}/{rec['entity_type']}"
        by_key[key].append(rec)

    rng = random.Random(seed)
    result: dict[str, list[dict]] = {}
    for key, pool in sorted(by_key.items()):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        result[key] = shuffled[:n]
    return result


def build_prompt(record: dict, tokenizer, demos: list[dict] | None = None) -> str:
    """Inference prompt identical to the training format (train.py:format_as_chat):
    apply_chat_template with enable_thinking=False + add_generation_prompt=True. The query
    uses instruction_fmt (statement + format). With `demos`, prepend few-shot user/assistant
    pairs (with the bare instruction; the format already appears in each demo's answer)."""
    def _content(rec: dict, fmt: bool = False) -> str:
        instruction = rec["instruction_fmt"] if fmt else rec["instruction"]
        return f"{instruction}\n\n{rec['input']}"

    messages = []
    for d in (demos or []):
        messages.append({"role": "user", "content": _content(d)})
        messages.append({"role": "assistant", "content": d["output"]})
    messages.append({"role": "user", "content": _content(record, fmt=True)})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        pairs = "".join(
            CHATML_PAIR.format(user=_content(d), assistant=d["output"])
            for d in (demos or [])
        )
        return pairs + CHATML_USER.format(content=_content(record, fmt=True))


def generate_predictions(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    rank: int = 0,
    demo_pool: dict[str, list[dict]] | None = None,
) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    # left-truncation preserves the query (end of the prompt) if it exceeds the limit.
    if demo_pool:
        # Scales with the number of shots (+1024 per demo; +160 covers the fixed format block).
        n_shot = max(len(v) for v in demo_pool.values())
        max_length = 1024 * (1 + n_shot) + 160
    else:
        # Zero-shot/FT: a cap of 1536 covers the dev long tail; padding only fills up to
        # the largest prompt in the batch, so it costs nothing on test (short inputs).
        n_shot = 0
        max_length = 1536
    tokenizer.truncation_side = "left"
    results = []

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]

        if batch_start % 200 == 0:
            print(f"  [rank {rank}] [{batch_start:>6}/{len(records)}]")

        def _demos(rec: dict) -> list[dict] | None:
            if not demo_pool:
                return None
            return demo_pool.get(f"{rec['act']}/{rec['entity_type']}") or None

        prompts = [build_prompt(rec, tokenizer, _demos(rec)) for rec in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, rec in enumerate(batch):
            generated = tokenizer.decode(
                out_ids[i][input_len:],
                skip_special_tokens=True,
            ).strip()

            if rank == 0 and batch_start == 0 and i < 3:
                raw_ids = out_ids[i][input_len:].tolist()
                prompt_decoded = tokenizer.decode(out_ids[i][:input_len], skip_special_tokens=False)
                generated_raw = tokenizer.decode(out_ids[i][input_len:], skip_special_tokens=False)
                print(f"\n--- DEBUG example {i} ---")
                print(f"  gold[:80]       : {rec['output'][:80]!r}")
                print(f"  pred[:80]       : {generated[:80]!r}")
                print(f"  pred_raw[:80]   : {generated_raw[:80]!r}")
                print(f"  token_ids[:20]  : {raw_ids[:20]}")
                print(f"  input_len       : {input_len}")
                print(f"  prompt[-100:]   : {prompt_decoded[-100:]!r}")
                print(f"  pred tokens generated: {out_ids[i].shape[0] - input_len}")

            _, gold_labels = parse_inline_bio(rec["output"])
            _, pred_labels = parse_inline_bio(generated)
            pred_labels = align_to_gold(pred_labels, len(gold_labels))

            results.append(
                {
                    "publication": rec["publication"],
                    "act": rec["act"],
                    "entity_type": rec["entity_type"],
                    "is_negative": rec["is_negative"],
                    "gold": gold_labels,
                    "pred": pred_labels,
                    # Raw generation (before the BIO parse): allows tolerant re-parsing
                    # offline without re-running inference (e.g., separating a format
                    # failure from a capability failure).
                    "gen": generated,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _strict_scores(gold: list, pred: list) -> dict:
    return {
        "f1": f1_score(gold, pred, mode="strict", scheme=IOB2),
        "precision": precision_score(gold, pred, mode="strict", scheme=IOB2),
        "recall": recall_score(gold, pred, mode="strict", scheme=IOB2),
    }


def overall_metrics(results: list[dict]) -> dict:
    gold = [r["gold"] for r in results]
    pred = [r["pred"] for r in results]
    return {
        "f1": f1_score(gold, pred),
        "precision": precision_score(gold, pred),
        "recall": recall_score(gold, pred),
        "strict": _strict_scores(gold, pred),
        "report": classification_report(gold, pred, output_dict=True),
    }


def per_act_metrics(results: list[dict]) -> dict:
    by_act: dict[str, list] = defaultdict(list)
    for r in results:
        by_act[r["act"]].append(r)

    out = {}
    for act, recs in sorted(by_act.items()):
        gold = [r["gold"] for r in recs]
        pred = [r["pred"] for r in recs]
        out[act] = {
            "f1": f1_score(gold, pred),
            "precision": precision_score(gold, pred),
            "recall": recall_score(gold, pred),
            "f1_strict": f1_score(gold, pred, mode="strict", scheme=IOB2),
            "n_records": len(recs),
        }
    return out


def per_entity_metrics(results: list[dict]) -> dict:
    by_key: dict[tuple, list] = defaultdict(list)
    for r in results:
        by_key[(r["act"], r["entity_type"])].append(r)

    out = {}
    for (act, etype), recs in sorted(by_key.items()):
        gold = [r["gold"] for r in recs]
        pred = [r["pred"] for r in recs]
        out[f"{act}/{etype}"] = {
            "act": act,
            "entity_type": etype,
            "f1": f1_score(gold, pred),
            "precision": precision_score(gold, pred),
            "recall": recall_score(gold, pred),
            "f1_strict": f1_score(gold, pred, mode="strict", scheme=IOB2),
            "n_records": len(recs),
        }
    return out


def bootstrap_f1(results: list[dict], n_iter: int = 1000, seed: int = 42) -> dict:
    # Strict IOB2 F1 — the same canonical metric as the point estimate and the comparison.
    rng = random.Random(seed)
    n = len(results)
    f1s = []
    for _ in range(n_iter):
        idx = rng.choices(range(n), k=n)
        gold = [results[i]["gold"] for i in idx]
        pred = [results[i]["pred"] for i in idx]
        f1s.append(f1_score(gold, pred, mode="strict", scheme=IOB2))
    f1s.sort()
    return {
        "mean": sum(f1s) / n_iter,
        "ci_lower": f1s[int(0.025 * n_iter)],
        "ci_upper": f1s[int(0.975 * n_iter)],
    }


# ---------------------------------------------------------------------------
# Compare two results (ΔF1 + bootstrap)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def compare(path_a: str, path_b: str) -> None:
    _log(f"Loading {Path(path_a).name} …")
    with open(path_a, encoding="utf-8") as f:
        da = json.load(f)
    _log(f"Loading {Path(path_b).name} …")
    with open(path_b, encoding="utf-8") as f:
        db = json.load(f)

    # Canonical H1 metric = strict IOB2 F1 (same as the bootstrap, via _spans).
    # Falls back to the default F1 only if the JSON is old and lacks the strict field.
    def _f1_strict(d: dict) -> float:
        return d["metrics"].get("strict", {}).get("f1", d["metrics"]["f1"])

    fa = _f1_strict(da)
    fb = _f1_strict(db)
    tag_a = da.get("model", Path(path_a).stem)
    tag_b = db.get("model", Path(path_b).stem)

    print(f"\nModel A: {tag_a}  →  strict F1 = {fa:.4f}")
    print(f"Model B: {tag_b}  →  strict F1 = {fb:.4f}")
    print(f"ΔF1 (A − B) = {fa - fb:+.4f}")

    if "predictions" not in da or "predictions" not in db:
        print("(individual predictions unavailable; bootstrap not run)")
        return

    preds_a = da["predictions"]
    preds_b = db["predictions"]
    if len(preds_a) != len(preds_b):
        print("Warning: sets of different sizes; bootstrap skipped.")
        return

    # Pre-compute TP/FP/FN per sample to avoid running seqeval 1000×.
    # Bootstrap = draw indices and sum integers → F1 in O(n_iter) instead of O(n·n_iter).
    def _spans(labels: list[str]) -> set[tuple]:
        spans, cur_type, cur_start = set(), None, None
        for i, lbl in enumerate(labels):
            if lbl.startswith("B-"):
                if cur_type is not None:
                    spans.add((cur_start, i - 1, cur_type))
                cur_type, cur_start = lbl[2:], i
            elif lbl.startswith("I-") and cur_type == lbl[2:]:
                pass
            else:
                if cur_type is not None:
                    spans.add((cur_start, i - 1, cur_type))
                cur_type, cur_start = None, None
        if cur_type is not None:
            spans.add((cur_start, len(labels) - 1, cur_type))
        return spans

    def _tpfpfn(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
        g, p = _spans(gold), _spans(pred)
        tp = len(g & p)
        return tp, len(p) - tp, len(g) - tp

    _log(f"Pre-computing TP/FP/FN (n={len(preds_a)}) …")
    counts_a = np.array([_tpfpfn(r["gold"], r["pred"]) for r in preds_a], dtype=np.int32)
    counts_b = np.array([_tpfpfn(r["gold"], r["pred"]) for r in preds_b], dtype=np.int32)

    def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    _log(f"Bootstrap (1000 iterations) …")
    rng_np = np.random.default_rng(42)
    n = len(preds_a)
    deltas = []
    for i in range(1000):
        if i % 200 == 0:
            _log(f"  bootstrap {i}/1000 …")
        idx = rng_np.integers(0, n, size=n)
        sa = counts_a[idx].sum(axis=0)
        sb = counts_b[idx].sum(axis=0)
        deltas.append(_f1_from_counts(*sa) - _f1_from_counts(*sb))
    deltas.sort()
    lo, hi = deltas[25], deltas[975]

    print(f"95% CI of ΔF1: [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("→ H1 supported: CI entirely positive.")
    else:
        print("→ H1 not supported: CI includes zero or negative values.")

    _log("Computing per-act metrics …")
    acts = sorted({r["act"] for r in preds_a})
    print(f"\n{'Act':42s} {'F1 A':>7} {'F1 B':>7} {'Δ':>7}")
    print("-" * 65)
    by_act_a: dict = defaultdict(list)
    by_act_b: dict = defaultdict(list)
    for r in preds_a:
        by_act_a[r["act"]].append(r)
    for r in preds_b:
        by_act_b[r["act"]].append(r)
    per_act = {}
    for act in acts:
        ga, pa = [r["gold"] for r in by_act_a[act]], [r["pred"] for r in by_act_a[act]]
        gb, pb = [r["gold"] for r in by_act_b[act]], [r["pred"] for r in by_act_b[act]]
        fa_act = f1_score(ga, pa, mode="strict", scheme=IOB2)
        fb_act = f1_score(gb, pb, mode="strict", scheme=IOB2)
        per_act[act] = {"f1_a": fa_act, "f1_b": fb_act, "delta": fa_act - fb_act}
        print(f"{act:42s} {fa_act:7.4f} {fb_act:7.4f} {fa_act - fb_act:+7.4f}")

    stem_a = Path(path_a).stem
    stem_b = Path(path_b).stem
    out_path = RESULTS_DIR / f"compare_{stem_a}_vs_{stem_b}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_a": tag_a,
                "model_b": tag_b,
                "f1_a": fa,
                "f1_b": fb,
                "delta_f1": fa - fb,
                "bootstrap": {
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "h1_sustained": bool(lo > 0),
                },
                "per_act": per_act,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _log(f"Result saved to: {out_path}")
    _log("Comparison done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model path or PEFT adapter")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                        help="Model precision (bf16 default; fp32 for NaN diagnosis)")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Number of few-shot demonstrations from training (0 = zero-shot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples of the split (quick probe; "
                             "dev has ~18.7k). Deterministic: same N every time.")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two result JSONs (ΔF1 + bootstrap)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.model is None:
        parser.error("--model is required for evaluation")

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)

    records: list[dict] = []
    with open(FT_DIR / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]

    demo_pool = load_few_shot_demos(args.few_shot) if args.few_shot > 0 else None

    if rank == 0:
        if demo_pool:
            shot_desc = f"{args.few_shot}-shot adaptive (per act/entity)"
        else:
            shot_desc = "zero-shot"
        print(f"Split '{args.split}': {len(records):,} examples ({world_size} GPU(s)) | {shot_desc}")
        print("Generating predictions…")

    # Each rank handles 1/world_size of the dataset, interleaved
    records_slice = records[rank::world_size]

    # The filename carries the _fmt suffix. Fine-tuning checkpoints already have it in
    # the directory name; only append it when it is not present yet.
    model_tag = Path(args.model).name
    if args.few_shot > 0:
        model_tag += f"_fs{args.few_shot}"
    if "_fmt" not in model_tag:
        model_tag += "_fmt"
    out_path = args.output or str(RESULTS_DIR / f"ner_{model_tag}_{args.split}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    predictions = generate_predictions(
        model, tokenizer, records_slice, args.max_new_tokens, args.batch_size, rank,
        demo_pool,
    )

    # Atomic write (write .partial and rename): rank 0 waits for the files via the
    # filesystem, so the presence of the final file guarantees it is complete.
    tmp_path = out_path + f".rank{rank}"
    with open(tmp_path + ".partial", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    os.replace(tmp_path + ".partial", tmp_path)

    # Synchronization WITHOUT a collective: ranks finish at very different times
    # (generations with no EOS run to the cap), and a dist.barrier() would blow past
    # the RCCL watchdog. Each rank already wrote its .rankN; rank 0 waits for them via
    # the filesystem.
    if rank != 0:
        return

    if world_size > 1:
        import time
        expected = [out_path + f".rank{r}" for r in range(world_size)]
        deadline = time.time() + 6 * 3600
        while time.time() < deadline:
            if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in expected):
                break
            time.sleep(15)
        else:
            missing = [p for p in expected
                       if not (os.path.exists(p) and os.path.getsize(p) > 0)]
            raise RuntimeError(f"Timeout waiting for ranks: {missing}")
        time.sleep(3)  # margin for the flush of the just-closed writer

    # Rank 0 aggregates in the original order and computes metrics
    if rank == 0:
        rank_preds = []
        for r in range(world_size):
            tmp = out_path + f".rank{r}"
            with open(tmp, encoding="utf-8") as f:
                rank_preds.append(json.load(f))
            os.remove(tmp)

        # Rebuild the original order: record i went to rank (i % world_size)
        all_predictions = []
        iters = [iter(rp) for rp in rank_preds]
        for i in range(len(records)):
            all_predictions.append(next(iters[i % world_size]))

        metrics = overall_metrics(all_predictions)
        per_act = per_act_metrics(all_predictions)
        per_entity = per_entity_metrics(all_predictions)
        boot = bootstrap_f1(all_predictions)

        print(f"\nF1 (default) : {metrics['f1']:.4f}")
        print(f"F1 (strict)  : {metrics['strict']['f1']:.4f}")
        print(f"Precision    : {metrics['precision']:.4f} "
              f"(strict {metrics['strict']['precision']:.4f})")
        print(f"Recall       : {metrics['recall']:.4f} "
              f"(strict {metrics['strict']['recall']:.4f})")
        print(f"95% CI (F1)  : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        print(f"\nPer act:\n{'Act':42s} {'F1':>7} {'F1str':>7} {'P':>7} {'R':>7} {'N':>6}")
        print("-" * 80)
        for act, m in per_act.items():
            print(
                f"{act:42s} {m['f1']:7.4f} {m['f1_strict']:7.4f} "
                f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_records']:6d}"
            )

        print(f"\nPer entity type:\n{'Act/Entity':42s} {'F1':>7} {'F1str':>7} "
              f"{'P':>7} {'R':>7} {'N':>6}")
        print("-" * 80)
        for key, m in per_entity.items():
            print(
                f"{key:42s} {m['f1']:7.4f} {m['f1_strict']:7.4f} "
                f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_records']:6d}"
            )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "split": args.split,
                    "metrics": metrics,
                    "per_act": per_act,
                    "per_entity": per_entity,
                    "bootstrap": boot,
                    "predictions": all_predictions,
                },
                f,
                ensure_ascii=False,
                indent=2,
                cls=_NumpyEncoder,
            )
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
