#!/usr/bin/env python3
from __future__ import annotations
"""Measure catastrophic forgetting on the general-ability benchmarks (H2).

Runs lm-evaluation-harness on the paper's benchmarks (MMLU, HellaSwag, ARC-Easy,
ARC-Challenge, WinoGrande, and the extended sets) and saves the result as JSON.
Automatically detects whether the path is a PEFT adapter.

Paper protocol (section 4.2):
  1. Evaluate base Qwen3-4B → baseline
  2. NER fine-tuning (train.py)
  3. Evaluate the fine-tuned model on the same benchmarks
  4. Report Δ (pp) per benchmark

Usage:
  # step 1: baseline (before fine-tuning)
  python experiments/evaluate_forgetting.py --model Qwen/Qwen3-4B --tag base

  # step 3: after fine-tuning
  python experiments/evaluate_forgetting.py --model results/checkpoints/lora_r8_fmt --tag lora_r8

  # step 4: compare (Δ pp per benchmark + interpretation)
  python experiments/evaluate_forgetting.py \\
      --compare results/bench_base.json results/bench_lora_r8.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# (task_name, [candidate metric keys, in preference order]).
# Sets aligned with the axes of Luo et al. (2023):
#   core  — the paper's 5 (MMLU/HellaSwag/ARC-E/ARC-C/WinoGrande), 0-shot.
#   ext   — RACE (reading), BoolQ, PIQA (reasoning), 0-shot.
#   gsm8k — generative arithmetic (exact-match), 5-shot.
#   scripted — CrowS-Pairs (bias) + MathQA; datasets with a .py loading script,
#              only in the isolated datasets<3.0 venv (see bench-scripted in submit.sh).
BENCHMARKS = {
    "mmlu":          ("mmlu",          ["acc,none"]),
    "hellaswag":     ("hellaswag",     ["acc_norm,none"]),
    "arc_easy":      ("arc_easy",      ["acc_norm,none"]),
    "arc_challenge": ("arc_challenge", ["acc_norm,none"]),
    "winogrande":    ("winogrande",    ["acc,none"]),
}

BENCHMARKS_EXT = {
    "race":        ("race",     ["acc,none"]),
    "boolq":       ("boolq",    ["acc,none"]),
    "piqa":        ("piqa",     ["acc_norm,none", "acc,none"]),
}

BENCHMARKS_GSM8K = {
    "gsm8k": ("gsm8k", ["exact_match,strict-match", "exact_match,flexible-extract"]),
}

BENCHMARKS_SCRIPTED = {
    "crows_pairs": ("crows_pairs_english", ["pct_stereotype,none"]),
    "mathqa":      ("mathqa",              ["acc_norm,none", "acc,none"]),
}

# task-set -> (registry, default num_fewshot, output file prefix)
TASK_SETS = {
    "core":     (BENCHMARKS,          0, "bench"),
    "ext":      (BENCHMARKS_EXT,      0, "bench_ext"),
    "gsm8k":    (BENCHMARKS_GSM8K,    5, "bench_gsm8k"),
    "scripted": (BENCHMARKS_SCRIPTED, 0, "bench_scripted"),
}

# task-sets that require --trust_remote_code (datasets with a loading script).
_TRUST_REMOTE = {"scripted"}

# Interpretation bands from the paper (section 5.2)
_THRESH_MODERATE    = 1.0   # pp
_THRESH_SIGNIFICANT = 3.0   # pp


def detect_peft(model_path: str) -> tuple[bool, str | None]:
    cfg = Path(model_path) / "adapter_config.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            data = json.load(f)
        return True, data["base_model_name_or_path"]
    return False, None


def build_model_args(model_path: str) -> str:
    is_peft, base = detect_peft(model_path)
    if is_peft:
        return (
            f"pretrained={base},"
            f"peft={model_path},"
            f"dtype=bfloat16"
        )
    return (
        f"pretrained={model_path},"
        f"dtype=bfloat16"
    )


def run_lm_eval(model_path: str, output_dir: str, registry: dict,
                num_fewshot: int, limit: int | None = None,
                trust_remote_code: bool = False) -> dict:
    existing = sorted(Path(output_dir).rglob("results*.json"))
    if existing:
        print(f"Using cached results: {existing[-1]}")
        with open(existing[-1], encoding="utf-8") as f:
            return json.load(f)

    tasks = ",".join(t for t, _ in registry.values())
    model_args = build_model_args(model_path)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", output_dir,
    ]
    if limit is not None:
        # Smoke test: only N examples per task (do NOT use for the paper's numbers).
        cmd += ["--limit", str(limit)]
    if trust_remote_code:
        # Required for datasets with a .py loading script (crows_pairs, mathqa)
        # under 'datasets<3.0' in the isolated evaluation venv.
        cmd += ["--trust_remote_code"]
    print("Running lm-eval:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    result_files = sorted(Path(output_dir).rglob("results*.json"))
    if not result_files:
        raise FileNotFoundError(f"lm-eval did not produce results.json in {output_dir}")
    with open(result_files[-1], encoding="utf-8") as f:
        return json.load(f)


def extract_scores(raw: dict, registry: dict) -> dict[str, float | None]:
    results = raw.get("results", {})
    scores: dict[str, float | None] = {}
    for bench, (task, metric_keys) in registry.items():
        task_results = results.get(task, {})
        val = None
        for k in metric_keys:
            if task_results.get(k) is not None:
                val = task_results[k]
                break
        scores[bench] = val
    return scores


def interpret_delta(delta_pp: float) -> str:
    a = abs(delta_pp)
    if a < _THRESH_MODERATE:
        return "negligible"
    if a < _THRESH_SIGNIFICANT:
        return "moderate"
    return "SIGNIFICANT"


def compare(path_a: str, path_b: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        da = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        db = json.load(f)

    sa = da["scores"]
    sb = db["scores"]
    tag_a = da.get("tag", Path(path_a).stem)
    tag_b = db.get("tag", Path(path_b).stem)

    print(f"\n{'Benchmark':20s} {tag_a:>14s} {tag_b:>14s} {'Δ (pp)':>9s}  Interpretation")
    print("-" * 80)
    for bench in sa:
        a = sa.get(bench)
        b = sb.get(bench)
        if a is None or b is None:
            print(f"{bench:20s} {'N/A':>14s} {'N/A':>14s} {'N/A':>9s}")
            continue
        delta = (b - a) * 100
        label = interpret_delta(delta)
        print(f"{bench:20s} {a*100:>13.2f}% {b*100:>13.2f}% {delta:>+9.2f}  {label}")

    print()
    print("Bands (paper, section 5.2):")
    print("  |Δ| < 1 pp  → negligible")
    print("  1–3 pp      → moderate")
    print("  > 3 pp      → significant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model or PEFT adapter (auto-detected)")
    parser.add_argument("--tag", default=None,
                        help="Result identifier (e.g., base, lora_r8)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--task-set", default="core", choices=list(TASK_SETS),
                        help="core (paper's 5) | ext (RACE/BoolQ/PIQA) | gsm8k | scripted (CrowS/MathQA)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Override the task-set's default num_fewshot")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only N examples per task (do not use for the paper)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two result JSONs (Δ pp per benchmark)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.model is None:
        parser.error("--model is required")

    registry, default_fewshot, prefix = TASK_SETS[args.task_set]
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else default_fewshot

    tag = args.tag or Path(args.model).name
    # Smoke test (--limit) writes to its own _smoke files, to never collide with the
    # real JSONs nor become their cache.
    smoke = "_smoke" if args.limit is not None else ""
    lm_eval_out = str(RESULTS_DIR / f"lm_eval_{args.task_set}_{tag}{smoke}")
    out_path = args.output or str(RESULTS_DIR / f"{prefix}_{tag}{smoke}.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = run_lm_eval(args.model, lm_eval_out, registry, num_fewshot, args.limit,
                      trust_remote_code=(args.task_set in _TRUST_REMOTE))
    scores = extract_scores(raw, registry)

    print(f"\nBenchmarks — {tag}:")
    for bench, score in scores.items():
        val = f"{score*100:.2f}%" if score is not None else "N/A"
        print(f"  {bench:20s}: {val}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model, "tag": tag, "task_set": args.task_set,
             "num_fewshot": num_fewshot, "scores": scores, "raw": raw},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
