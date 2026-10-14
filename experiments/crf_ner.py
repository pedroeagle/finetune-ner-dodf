#!/usr/bin/env python3
"""Classical CRF baseline for DODF NER (CPU-only, no GPU, no network).

Companion to the BERTimbau encoder baseline (train_bertimbau.py +
evaluate_bertimbau.py): same CoNLL splits, same document unit (publication, act),
a single global model over all BIO labels, and the SAME slicing onto the decoder's
test.jsonl records — so the resulting JSON is directly comparable via:

    python experiments/evaluate_ner.py --compare \\
        results/ner_crf_test.json results/ner_bertimbau_base_test.json

Unlike the neural baselines, the CRF trains in minutes on CPU. This script trains on
train.conll and evaluates on --split in one pass; pass --save-model/--load-model to
persist the trained CRF.

Deps:  pip install sklearn-crfsuite seqeval numpy joblib

Usage:
    python experiments/crf_ner.py --split test \\
        --data-dir datasets/dodfp-corpus/splits/conll \\
        --ft-dir   datasets/dodfp-corpus/splits/finetune
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
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
DEFAULT_CONLL_DIR = ROOT / "datasets/dodfp-corpus/splits/conll"
DEFAULT_FT_DIR = ROOT / "datasets/dodfp-corpus/splits/finetune"
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
# Data  (read_conll mirrors train_bertimbau.py; parse/slice mirror evaluate_*)
# ---------------------------------------------------------------------------

def read_conll(path: Path) -> list[dict]:
    """CoNLL (Publication:/Act:/token label/blank line) → list of
    {'publication','act','tokens','labels'}; document unit = (publication, act)."""
    docs: list[dict] = []
    pub = act = None
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        if tokens:
            docs.append({"publication": pub, "act": act,
                         "tokens": list(tokens), "labels": list(labels)})

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("Publication:"):
            pub = line.split(":", 1)[1].strip()
        elif line.startswith("Act:"):
            act = line.split(":", 1)[1].strip()
            tokens, labels = [], []
        elif line.strip() == "":
            flush()
            tokens, labels = [], []
        else:
            word, label = line.rsplit(" ", 1)
            tokens.append(word)
            labels.append(label)
    flush()
    return docs


def parse_inline_bio(text: str) -> tuple[list[str], list[str]]:
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


def slice_to_entity(doc_labels: list[str], entity_type: str) -> list[str]:
    """Keep only B-/I- of the queried entity_type; everything else → 'O'
    (identical to evaluate_bertimbau.slice_to_entity)."""
    keep = {f"B-{entity_type}", f"I-{entity_type}"}
    return [lab if lab in keep else "O" for lab in doc_labels]


# ---------------------------------------------------------------------------
# CRF features  (classical token features: identity, shape, affixes, context)
# ---------------------------------------------------------------------------

def _shape(w: str) -> str:
    s = re.sub(r"[A-ZÀ-Ý]", "X", w)      # uppercase first
    s = re.sub(r"[a-zà-ÿ]", "x", s)      # then lowercase
    s = re.sub(r"[0-9]", "d", s)         # then digits
    return s


def word_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    feats = {
        "bias": 1.0,
        "w.lower": w.lower(),
        "w[-3:]": w[-3:],
        "w[-2:]": w[-2:],
        "w[:2]": w[:2],
        "w[:3]": w[:3],
        "w.isupper": w.isupper(),
        "w.istitle": w.istitle(),
        "w.isdigit": w.isdigit(),
        "w.hasdigit": any(c.isdigit() for c in w),
        "w.haspunct": any(not c.isalnum() for c in w),
        "w.len": len(w),
        "w.shape": _shape(w),
    }
    if i > 0:
        p = tokens[i - 1]
        feats.update({
            "-1.lower": p.lower(), "-1.istitle": p.istitle(),
            "-1.isupper": p.isupper(), "-1.isdigit": p.isdigit(),
            "-1.shape": _shape(p),
        })
    else:
        feats["BOS"] = True
    if i > 1:
        feats["-2.lower"] = tokens[i - 2].lower()
    if i < len(tokens) - 1:
        n = tokens[i + 1]
        feats.update({
            "+1.lower": n.lower(), "+1.istitle": n.istitle(),
            "+1.isupper": n.isupper(), "+1.isdigit": n.isdigit(),
            "+1.shape": _shape(n),
        })
    else:
        feats["EOS"] = True
    if i < len(tokens) - 2:
        feats["+2.lower"] = tokens[i + 2].lower()
    return feats


def doc2features(tokens: list[str]) -> list[dict]:
    return [word_features(tokens, i) for i in range(len(tokens))]


# ---------------------------------------------------------------------------
# Metrics  (identical seqeval calls to evaluate_ner.py)
# ---------------------------------------------------------------------------

def _strict(gold, pred):
    return {
        "f1": f1_score(gold, pred, mode="strict", scheme=IOB2),
        "precision": precision_score(gold, pred, mode="strict", scheme=IOB2),
        "recall": recall_score(gold, pred, mode="strict", scheme=IOB2),
    }


def overall_metrics(results):
    gold = [r["gold"] for r in results]
    pred = [r["pred"] for r in results]
    return {
        "f1": f1_score(gold, pred),
        "precision": precision_score(gold, pred),
        "recall": recall_score(gold, pred),
        "strict": _strict(gold, pred),
        "report": classification_report(gold, pred, output_dict=True),
    }


def _group_metrics(results, keyfn):
    by: dict = defaultdict(list)
    for r in results:
        by[keyfn(r)].append(r)
    out = {}
    for k, recs in sorted(by.items()):
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        out[k if isinstance(k, str) else f"{k[0]}/{k[1]}"] = {
            "f1": f1_score(g, p),
            "precision": precision_score(g, p),
            "recall": recall_score(g, p),
            "f1_strict": f1_score(g, p, mode="strict", scheme=IOB2),
            "n_records": len(recs),
        }
    return out


def bootstrap_f1(results, n_iter=1000, seed=42):
    rng = random.Random(seed)
    n = len(results)
    f1s = []
    for _ in range(n_iter):
        idx = rng.choices(range(n), k=n)
        g = [results[i]["gold"] for i in idx]
        p = [results[i]["pred"] for i in idx]
        f1s.append(f1_score(g, p, mode="strict", scheme=IOB2))
    f1s.sort()
    return {"mean": sum(f1s) / n_iter,
            "ci_lower": f1s[int(0.025 * n_iter)],
            "ci_upper": f1s[int(0.975 * n_iter)]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--data-dir", default=str(DEFAULT_CONLL_DIR),
                    help="Dir with {split}.conll")
    ap.add_argument("--ft-dir", default=str(DEFAULT_FT_DIR),
                    help="Dir with {split}.jsonl (decoder records, for slicing)")
    ap.add_argument("--c1", type=float, default=0.1, help="L1 penalty")
    ap.add_argument("--c2", type=float, default=0.1, help="L2 penalty")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--load-model", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import sklearn_crfsuite  # late import: clearer error if not installed

    conll_dir = Path(args.data_dir)
    ft_dir = Path(args.ft_dir)

    if args.load_model:
        import joblib
        print(f"Loading CRF from {args.load_model}")
        crf = joblib.load(args.load_model)
    else:
        print(f"Reading {conll_dir / (args.train_split + '.conll')} …")
        train_docs = read_conll(conll_dir / f"{args.train_split}.conll")
        X = [doc2features(d["tokens"]) for d in train_docs]
        y = [d["labels"] for d in train_docs]
        n_tok = sum(len(s) for s in y)
        print(f"Training CRF on {len(train_docs):,} docs / {n_tok:,} tokens "
              f"(c1={args.c1}, c2={args.c2}, max_iter={args.max_iter}) …")
        t0 = time.time()
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=args.c1, c2=args.c2,
            max_iterations=args.max_iter, all_possible_transitions=True,
        )
        crf.fit(X, y)
        print(f"Trained in {time.time() - t0:.1f}s | {len(crf.classes_)} labels")
        if args.save_model:
            import joblib
            joblib.dump(crf, args.save_model)
            print(f"Saved CRF to {args.save_model}")

    # Predict per-document labels on the eval split.
    print(f"Predicting on {conll_dir / (args.split + '.conll')} …")
    eval_docs = read_conll(conll_dir / f"{args.split}.conll")
    doc_preds: dict[tuple, list[str]] = {}
    X_eval = [doc2features(d["tokens"]) for d in eval_docs]
    y_pred = crf.predict(X_eval)
    for d, labs in zip(eval_docs, y_pred):
        doc_preds[(d["publication"], d["act"])] = list(labs)

    # Slice onto the SAME (pub, act, entity_type) records as the decoder/encoder.
    records = []
    with open(ft_dir / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    predictions, missing = [], 0
    for rec in records:
        key = (rec["publication"], rec["act"])
        doc_labels = doc_preds.get(key)
        if doc_labels is None:
            missing += 1
            doc_labels = ["O"] * len(rec["input"].split())
        _, gold_labels = parse_inline_bio(rec["output"])
        pred_sliced = slice_to_entity(doc_labels, rec["entity_type"])
        pred_labels = align_to_gold(pred_sliced, len(gold_labels))
        predictions.append({
            "publication": rec["publication"], "act": rec["act"],
            "entity_type": rec["entity_type"], "is_negative": rec["is_negative"],
            "gold": gold_labels, "pred": pred_labels,
        })
    if missing:
        print(f"WARNING: {missing} records with no matching conll doc (pred='O').")

    metrics = overall_metrics(predictions)
    per_act = _group_metrics(predictions, lambda r: r["act"])
    per_entity = _group_metrics(predictions, lambda r: (r["act"], r["entity_type"]))
    boot = bootstrap_f1(predictions)

    print(f"\nF1 (default) : {metrics['f1']:.4f}")
    print(f"F1 (strict)  : {metrics['strict']['f1']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f} (strict {metrics['strict']['precision']:.4f})")
    print(f"Recall       : {metrics['recall']:.4f} (strict {metrics['strict']['recall']:.4f})")
    print(f"95% CI (F1)  : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    out_path = args.output or str(RESULTS_DIR / f"ner_crf_{args.split}.json")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "crf", "split": args.split,
            "metrics": metrics, "per_act": per_act, "per_entity": per_entity,
            "bootstrap": boot, "predictions": predictions,
        }, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
