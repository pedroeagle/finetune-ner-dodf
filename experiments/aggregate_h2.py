#!/usr/bin/env python3
"""Aggregate the H2 forgetting results across seeds (reproduces the tab:forget table).

For each strategy (LoRA, DoRA, SDFT) it reads the benchmarks of the three seeds
(42/43/44), computes the Δ (pp) vs the base model per benchmark (mean across seeds)
and the average drop over the 10 accuracy benchmarks with the standard deviation
across seeds. CrowS-Pairs is reported separately (bias probe, not in the average).

Pure re-analysis of the JSONs saved by evaluate_forgetting.py; runs no model.

Usage:
  python experiments/aggregate_h2.py --results /path/to/results
"""
import argparse
import json
import os
from statistics import mean, stdev

# 10 accuracy benchmarks (in table order) + CrowS-Pairs reported separately.
ACC = ["mmlu", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
       "boolq", "piqa", "mathqa", "race", "gsm8k"]
LABEL = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "arc_easy": "ARC-Easy",
         "arc_challenge": "ARC-Chal.", "winogrande": "WinoGrande", "boolq": "BoolQ",
         "piqa": "PIQA", "mathqa": "MathQA", "race": "RACE", "gsm8k": "GSM8K",
         "crows_pairs": "CrowS-Pairs"}

# Base: one file per benchmark family.
BASE = {"core": "bench_base.json", "ext": "bench_ext_base.json",
        "gsm8k": "bench_gsm8k_base.json", "scripted": "bench_scripted_base.json"}

# Strategy -> seed -> {family: file}. Explicit because the SDFT (core) file names
# carry the winning checkpoint, which differs by seed.
RUNS = {
    "LoRA": {
        42: {"core": "bench_lora_r8.json", "ext": "bench_ext_lora_r8.json",
             "gsm8k": "bench_gsm8k_lora_r8.json", "scripted": "bench_scripted_lora_r8.json"},
        43: {"core": "bench_lora_r8_s43.json", "ext": "bench_ext_lora_r8_s43.json",
             "gsm8k": "bench_gsm8k_lora_r8_s43.json", "scripted": "bench_scripted_lora_r8_s43.json"},
        44: {"core": "bench_lora_r8_s44.json", "ext": "bench_ext_lora_r8_s44.json",
             "gsm8k": "bench_gsm8k_lora_r8_s44.json", "scripted": "bench_scripted_lora_r8_s44.json"},
    },
    "DoRA": {
        42: {"core": "bench_dora_r8.json", "ext": "bench_ext_dora_r8.json",
             "gsm8k": "bench_gsm8k_dora_r8.json", "scripted": "bench_scripted_dora_r8.json"},
        43: {"core": "bench_dora_r8_s43.json", "ext": "bench_ext_dora_r8_s43.json",
             "gsm8k": "bench_gsm8k_dora_r8_s43.json", "scripted": "bench_scripted_dora_r8_s43.json"},
        44: {"core": "bench_dora_r8_s44.json", "ext": "bench_ext_dora_r8_s44.json",
             "gsm8k": "bench_gsm8k_dora_r8_s44.json", "scripted": "bench_scripted_dora_r8_s44.json"},
    },
    "SDFT": {
        42: {"core": "bench_sdft_r8_full_budget_checkpoint-8205.json", "ext": "bench_ext_sdft_r8_s42.json",
             "gsm8k": "bench_gsm8k_sdft_r8_s42.json", "scripted": "bench_scripted_sdft_r8_s42.json"},
        43: {"core": "bench_sdft_r8_full_budget_s43_checkpoint-8205.json", "ext": "bench_ext_sdft_r8_s43.json",
             "gsm8k": "bench_gsm8k_sdft_r8_s43.json", "scripted": "bench_scripted_sdft_r8_s43.json"},
        44: {"core": "bench_sdft_r8_full_budget_s44_checkpoint-5450.json", "ext": "bench_ext_sdft_r8_s44.json",
             "gsm8k": "bench_gsm8k_sdft_r8_s44.json", "scripted": "bench_scripted_sdft_r8_s44.json"},
    },
}


def load_scores(results_dir, files):
    """Merge the `scores` of the four families into a single dict {benchmark: score}."""
    scores = {}
    for fam in ("core", "ext", "gsm8k", "scripted"):
        with open(os.path.join(results_dir, files[fam]), encoding="utf-8") as f:
            scores.update(json.load(f)["scores"])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"),
                    help="Directory with the bench_*.json files (default: ../results)")
    args = ap.parse_args()

    base = load_scores(args.results, BASE)

    # Per strategy: Δ per benchmark and per seed (in pp).
    deltas = {}  # method -> bench -> [Δ per seed]
    acc_avg = {}  # method -> [avg of the 10 per seed]
    for method, seeds in RUNS.items():
        deltas[method] = {b: [] for b in ACC + ["crows_pairs"]}
        acc_avg[method] = []
        for seed, files in seeds.items():
            s = load_scores(args.results, files)
            for b in ACC + ["crows_pairs"]:
                deltas[method][b].append((s[b] - base[b]) * 100)
            acc_avg[method].append(mean((s[b] - base[b]) * 100 for b in ACC))

    methods = list(RUNS)
    print(f"{'Benchmark':13s} {'Base':>7s}" + "".join(f" {'Δ'+m[0]:>8s}" for m in methods))
    print("-" * (13 + 8 + 9 * len(methods)))
    for b in ACC:
        row = f"{LABEL[b]:13s} {base[b]*100:7.2f}"
        row += "".join(f" {mean(deltas[m][b]):+8.2f}" for m in methods)
        print(row)
    print("-" * (13 + 8 + 9 * len(methods)))
    base_avg = mean(base[b] * 100 for b in ACC)
    row = f"{'Avg. acc.':13s} {base_avg:7.2f}"
    row += "".join(f" {mean(acc_avg[m]):+5.1f}±{stdev(acc_avg[m]):.1f}" for m in methods)
    print(row)
    print("-" * (13 + 8 + 9 * len(methods)))
    b = "crows_pairs"
    row = f"{LABEL[b]:13s} {base[b]*100:7.2f}"
    row += "".join(f" {mean(deltas[m][b]):+8.2f}" for m in methods)
    print(row)


if __name__ == "__main__":
    main()
