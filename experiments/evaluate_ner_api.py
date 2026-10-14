#!/usr/bin/env python3
"""Zero-shot generative NER via a hosted chat API (closed frontier LLM).

Mirror of experiments/evaluate_ner.py, but instead of running a local model it
sends each query to an OpenAI-compatible /chat/completions endpoint. The prompt,
BIO parsing and seqeval scoring are IDENTICAL to the local path, so the resulting
JSON is directly comparable with the local baselines via:

    python experiments/evaluate_ner.py --compare \\
        results/ner_<api-tag>_fmt_test.json results/ner_Qwen3-4B_fmt_test.json

PRIVACY NOTE: the corpus carries civil-servant names and registration numbers.
This script SENDS those texts to an external API. It exists to measure a one-time
zero-shot baseline on already-public DODF data; it is not a deployment path. See
the paper's Ethics Statement.

The API key is read from an environment variable (never a flag, never logged):

    export OPENAI_API_KEY="sk-..."          # do NOT paste the key into any chat
    pip install openai seqeval numpy

Usage:
    # 1) smoke test: 20 examples, check format + cost before the full run
    python experiments/evaluate_ner_api.py --model gpt-6-luna --split test --limit 20

    # 2) full test split (18,737 queries) once the smoke test looks right
    python experiments/evaluate_ner_api.py --model gpt-6-luna --split test

    # custom gateway / env var name
    python experiments/evaluate_ner_api.py --model gpt-6-luna \\
        --base-url https://my-gateway/v1 --api-key-env MY_KEY_VAR

The run is resumable: predictions stream to <output>.partial.jsonl as they finish;
re-running with the same --output skips indices already done (no re-billing).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ROOT / "datasets/dodfp-corpus/splits/finetune"
RESULTS_DIR = ROOT / "results"

_UNIT_RE = re.compile(r"^(.+)\(([BI]-[\w]+|O)\)$")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Prompt + BIO parsing  (verbatim from evaluate_ner.py — keep in sync)
# ---------------------------------------------------------------------------

def build_messages(record: dict) -> list[dict]:
    """Zero-shot user turn identical to the local path: instruction_fmt + input.
    No system prompt is added, to match the local chat-template zero-shot."""
    content = f"{record['instruction_fmt']}\n\n{record['input']}"
    return [{"role": "user", "content": content}]


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
# Metrics  (verbatim from evaluate_ner.py — keep in sync)
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
# API call  (defensive: adapts to a model that rejects a param, strips it)
# ---------------------------------------------------------------------------

_UNSUPPORTED_HINTS = {
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "seed": ("seed",),
    "max_tokens": ("max_tokens", "max_completion_tokens"),
}


def make_client(base_url: str | None, api_key_env: str):
    from openai import OpenAI  # imported here so the file loads without the SDK

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(
            f"No API key in ${api_key_env}. Set it in your shell "
            f"(export {api_key_env}=...), never on the command line or in a chat."
        )
    kwargs = {"api_key": api_key}
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_api(client, model: str, messages: list[dict], max_tokens: int,
             temperature: float, seed: int, use_max_completion: bool,
             max_retries: int = 6) -> tuple[str, dict]:
    """One completion with exponential backoff. Returns (text, usage_dict).
    If the model rejects a parameter, it is dropped and the call retried."""
    params: dict = {"model": model, "messages": messages}
    params["max_completion_tokens" if use_max_completion else "max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**params)
            text = (resp.choices[0].message.content or "").strip()
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                }
            return text, usage
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            # If a specific param is unsupported, strip it and retry immediately.
            stripped = False
            for param, hints in _UNSUPPORTED_HINTS.items():
                present = [k for k in params if k in (param,) + hints]
                if present and any(h in msg for h in hints) and (
                    "unsupported" in msg or "not supported" in msg
                    or "unknown" in msg or "does not support" in msg
                    or "must be" in msg
                ):
                    for k in present:
                        params.pop(k, None)
                    stripped = True
            if stripped:
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="API model name (e.g. gpt-6-luna)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="Dir with {split}.jsonl (data is gitignored; point here)")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL (default: SDK/OPENAI_BASE_URL)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="Env var holding the API key (never pass the key directly)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Generation cap (512 matches the local baseline)")
    ap.add_argument("--use-max-completion-tokens", action="store_true",
                    help="Send max_completion_tokens instead of max_tokens")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 for greedy/deterministic (dropped if the model refuses it)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Request seed for determinism (dropped if unsupported)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent requests")
    ap.add_argument("--limit", type=int, default=None,
                    help="First N examples only (deterministic smoke test)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Random N examples (changes comparability; re-score baselines "
                         "on the same subset). Uses --sample-seed.")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--output", default=None, help="Output JSON path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    split_path = data_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise SystemExit(f"Split not found: {split_path} (use --data-dir)")

    records: list[dict] = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.sample is not None:
        rng = random.Random(args.sample_seed)
        idxs = sorted(rng.sample(range(len(records)), min(args.sample, len(records))))
        records = [records[i] for i in idxs]
    if args.limit is not None:
        records = records[: args.limit]

    model_tag = args.model.replace("/", "-")
    if "_fmt" not in model_tag:
        model_tag += "_fmt"
    out_path = args.output or str(RESULTS_DIR / f"ner_{model_tag}_{args.split}.json")
    partial_path = out_path + ".partial.jsonl"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: load predictions already written to the partial file.
    done: dict[int, dict] = {}
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done[e["idx"]] = e
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(done)} of {len(records)} already done.")

    client = make_client(args.base_url, args.api_key_env)

    todo = [i for i in range(len(records)) if i not in done]
    print(f"Split '{args.split}': {len(records):,} queries | {len(todo):,} to send "
          f"| model={args.model} | workers={args.workers}")
    print("PRIVACY: sending act texts (civil-servant names/IDs) to the external API.")

    lock = threading.Lock()
    pf = open(partial_path, "a", encoding="utf-8")
    counters = {"done": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def work(i: int) -> None:
        rec = records[i]
        try:
            text, usage = call_api(
                client, args.model, build_messages(rec), args.max_tokens,
                args.temperature, args.seed, args.use_max_completion_tokens,
            )
        except Exception as e:  # noqa: BLE001
            with lock:
                counters["errors"] += 1
            print(f"  [idx {i}] ERROR: {type(e).__name__}: {str(e)[:120]}")
            return
        _, gold_labels = parse_inline_bio(rec["output"])
        _, pred_labels = parse_inline_bio(text)
        pred_labels = align_to_gold(pred_labels, len(gold_labels))
        entry = {
            "idx": i,
            "publication": rec["publication"],
            "act": rec["act"],
            "entity_type": rec["entity_type"],
            "is_negative": rec["is_negative"],
            "gold": gold_labels,
            "pred": pred_labels,
            "gen": text,
        }
        with lock:
            pf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            pf.flush()
            done[i] = entry
            counters["done"] += 1
            counters["prompt_tokens"] += usage.get("prompt_tokens", 0)
            counters["completion_tokens"] += usage.get("completion_tokens", 0)
            n = counters["done"]
            if n % 100 == 0 or n == len(todo):
                print(f"  [{n:>6}/{len(todo)}] sent | tok in/out "
                      f"{counters['prompt_tokens']}/{counters['completion_tokens']} "
                      f"| errors {counters['errors']}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, i) for i in todo]
            for _ in as_completed(futures):
                pass
    finally:
        pf.close()

    if counters["errors"]:
        print(f"\n{counters['errors']} queries errored and are NOT in the results. "
              f"Re-run the same command to retry only those (resume skips the done ones).")

    # Assemble in original order; require every record present before scoring.
    missing = [i for i in range(len(records)) if i not in done]
    if missing:
        print(f"\n{len(missing)} queries still missing (e.g. idx {missing[:5]}). "
              f"Re-run to finish them before the number is final. No metrics written yet.")
        return

    all_predictions = [done[i] for i in range(len(records))]
    metrics = overall_metrics(all_predictions)
    per_act = per_act_metrics(all_predictions)
    per_entity = per_entity_metrics(all_predictions)
    boot = bootstrap_f1(all_predictions)

    print(f"\nF1 (default) : {metrics['f1']:.4f}")
    print(f"F1 (strict)  : {metrics['strict']['f1']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f} (strict {metrics['strict']['precision']:.4f})")
    print(f"Recall       : {metrics['recall']:.4f} (strict {metrics['strict']['recall']:.4f})")
    print(f"95% CI (F1)  : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print(f"Tokens in/out: {counters['prompt_tokens']}/{counters['completion_tokens']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "split": args.split,
                "api": True,
                "metrics": metrics,
                "per_act": per_act,
                "per_entity": per_entity,
                "bootstrap": boot,
                "predictions": all_predictions,
            },
            f, ensure_ascii=False, indent=2, cls=_NumpyEncoder,
        )
    print(f"\nResults saved to: {out_path}")
    print(f"(partial kept at {partial_path}; delete it to force a clean re-run)")


if __name__ == "__main__":
    main()
