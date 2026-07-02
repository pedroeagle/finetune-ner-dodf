#!/usr/bin/env python3
from __future__ import annotations
"""Mede esquecimento catastrófico nos 5 benchmarks MCQ (H2).

Roda lm-evaluation-harness nos benchmarks do paper (MMLU, HellaSwag,
ARC-Easy, ARC-Challenge, WinoGrande) e salva resultado em JSON.
Detecta automaticamente se o caminho é um adaptador PEFT.

Protocolo do paper (seção 4.2):
  1. Avaliar Qwen3-4B base → baseline
  2. Fine-tuning NER (train.py)
  3. Avaliar modelo fine-tuned nos mesmos benchmarks
  4. Reportar Δ (pp) por benchmark

Uso:
  # passo 1: baseline (antes do fine-tuning)
  python experiments/evaluate_forgetting.py --model Qwen/Qwen3-4B --tag base

  # passo 3: após fine-tuning
  python experiments/evaluate_forgetting.py --model results/checkpoints/lora_r8_fmt --tag lora_r8

  # passo 4: comparar (Δ pp por benchmark + interpretação)
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

# Nomes de task no lm-eval e métrica principal de cada benchmark
BENCHMARKS = {
    "mmlu":          ("mmlu",          "acc,none"),
    "hellaswag":     ("hellaswag",     "acc_norm,none"),
    "arc_easy":      ("arc_easy",      "acc_norm,none"),
    "arc_challenge": ("arc_challenge", "acc_norm,none"),
    "winogrande":    ("winogrande",    "acc,none"),
}

# Faixas de interpretação do paper (seção 5.2)
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


def run_lm_eval(model_path: str, output_dir: str) -> dict:
    existing = sorted(Path(output_dir).rglob("results*.json"))
    if existing:
        print(f"Usando resultados em cache: {existing[-1]}")
        with open(existing[-1], encoding="utf-8") as f:
            return json.load(f)

    tasks = ",".join(t for t, _ in BENCHMARKS.values())
    model_args = build_model_args(model_path)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", "0",
        "--batch_size", "auto",
        "--output_path", output_dir,
    ]
    print("Executando lm-eval:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    result_files = sorted(Path(output_dir).rglob("results*.json"))
    if not result_files:
        raise FileNotFoundError(f"lm-eval não gerou results.json em {output_dir}")
    with open(result_files[-1], encoding="utf-8") as f:
        return json.load(f)


def extract_scores(raw: dict) -> dict[str, float | None]:
    results = raw.get("results", {})
    scores: dict[str, float | None] = {}
    for bench, (task, metric_key) in BENCHMARKS.items():
        task_results = results.get(task, {})
        scores[bench] = task_results.get(metric_key)
    return scores


def interpret_delta(delta_pp: float) -> str:
    a = abs(delta_pp)
    if a < _THRESH_MODERATE:
        return "negligível"
    if a < _THRESH_SIGNIFICANT:
        return "moderado"
    return "SIGNIFICATIVO"


def compare(path_a: str, path_b: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        da = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        db = json.load(f)

    sa = da["scores"]
    sb = db["scores"]
    tag_a = da.get("tag", Path(path_a).stem)
    tag_b = db.get("tag", Path(path_b).stem)

    print(f"\n{'Benchmark':20s} {tag_a:>14s} {tag_b:>14s} {'Δ (pp)':>9s}  Interpretação")
    print("-" * 80)
    for bench in BENCHMARKS:
        a = sa.get(bench)
        b = sb.get(bench)
        if a is None or b is None:
            print(f"{bench:20s} {'N/A':>14s} {'N/A':>14s} {'N/A':>9s}")
            continue
        delta = (b - a) * 100
        label = interpret_delta(delta)
        print(f"{bench:20s} {a*100:>13.2f}% {b*100:>13.2f}% {delta:>+9.2f}  {label}")

    print()
    print("Faixas (paper, seção 5.2):")
    print("  |Δ| < 1 pp  → negligível")
    print("  1–3 pp      → moderado")
    print("  > 3 pp      → significativo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Modelo ou adaptador PEFT (detecção automática)")
    parser.add_argument("--tag", default=None,
                        help="Identificador do resultado (ex: base, lora_r8)")
    parser.add_argument("--output", default=None, help="Arquivo JSON de saída")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compara dois JSONs de resultado (Δ pp por benchmark)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.model is None:
        parser.error("--model é obrigatório")

    tag = args.tag or Path(args.model).name
    lm_eval_out = str(RESULTS_DIR / f"lm_eval_{tag}")
    out_path = args.output or str(RESULTS_DIR / f"bench_{tag}.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = run_lm_eval(args.model, lm_eval_out)
    scores = extract_scores(raw)

    print(f"\nBenchmarks — {tag}:")
    for bench, score in scores.items():
        val = f"{score*100:.2f}%" if score is not None else "N/A"
        print(f"  {bench:20s}: {val}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model, "tag": tag, "scores": scores, "raw": raw},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
