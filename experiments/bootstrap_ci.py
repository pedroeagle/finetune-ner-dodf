#!/usr/bin/env python3
"""Document-level bootstrap for the H1 NER confidence intervals.

The split and the annotation correlation are at the document level (templated
acts, several correlated questions per document), so resampling whole documents
gives honest CIs. Strict entity-level F1 is micro-averaged, so TP/pred/gold
counts are additive per document and can be pre-aggregated; resampling documents
with replacement is then exact and fast.

Pure re-analysis of saved predictions (results/ner_*_{test,dev}.json); runs no
model. Point F1 estimates are unchanged, only the CI widths.
"""
import argparse, json, sys, os
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"),
                 help="Directory with the ner_*_{test,dev}.json files (default: ../results)")
RESULTS = _ap.parse_args().results
RNG = np.random.default_rng(42)
N_BOOT = 1000


def entities_iob2(tags):
    """Extract strict IOB2 entities as (type, start, end) spans."""
    ents, cur, start = [], None, None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur is not None:
                ents.append((cur, start, i))
            cur, start = t[2:], i
        elif t.startswith("I-"):
            if cur is not None and t[2:] == cur:
                continue
            # I- without a matching open B- : invalid under strict IOB2
            if cur is not None:
                ents.append((cur, start, i))
            cur, start = None, None
        else:  # "O"
            if cur is not None:
                ents.append((cur, start, i))
            cur, start = None, None
    if cur is not None:
        ents.append((cur, start, len(tags)))
    return set(ents)


def load_counts(fname):
    """Return per-document arrays (tp, npred, ngold) and the doc order."""
    d = json.load(open(os.path.join(RESULTS, fname)))
    preds = d["predictions"]
    docs, doc_idx = [], {}
    tp_l, np_l, ng_l = [], [], []
    for ex in preds:
        pub = ex["publication"]
        if pub not in doc_idx:
            doc_idx[pub] = len(docs)
            docs.append(pub)
            tp_l.append(0); np_l.append(0); ng_l.append(0)
        g = entities_iob2(ex["gold"])
        p = entities_iob2(ex["pred"])
        j = doc_idx[pub]
        tp_l[j] += len(g & p)
        np_l[j] += len(p)
        ng_l[j] += len(g)
    return docs, np.array(tp_l), np.array(np_l), np.array(ng_l)


def f1_from(tp, npred, ngold):
    prec = tp / npred if npred else 0.0
    rec = tp / ngold if ngold else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def point_f1(tp, npred, ngold):
    return f1_from(tp.sum(), npred.sum(), ngold.sum())


def boot_single(counts, samples):
    tp, npred, ngold = counts
    return np.array([f1_from(tp[s].sum(), npred[s].sum(), ngold[s].sum())
                     for s in samples])


def ci(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run(pairs, singles, tag):
    # Load once per model file (aligned doc order assumed identical: same split)
    cache = {}
    order_ref = None
    for f in set([p[2] for p in pairs] + [p[3] for p in pairs] + [s[1] for s in singles]):
        docs, tp, npred, ngold = load_counts(f)
        if order_ref is None:
            order_ref = docs
        assert docs == order_ref, f"doc order mismatch in {f}"
        cache[f] = (tp, npred, ngold)
    ndocs = len(order_ref)
    samples = [RNG.integers(0, ndocs, size=ndocs) for _ in range(N_BOOT)]

    boots = {f: boot_single(cache[f], samples) for f in cache}
    print(f"\n===== {tag} (cluster bootstrap, {N_BOOT} resamples, {ndocs} docs) =====")
    print("-- point F1 (strict) --")
    for f in cache:
        print(f"  {f:42s} F1={point_f1(*cache[f]):.4f}")
    print("-- single-model 95% CI --")
    for name, f in singles:
        lo, hi = ci(boots[f])
        print(f"  {name:24s} F1={point_f1(*cache[f]):.3f}  CI[{lo:.3f}, {hi:.3f}]")
    print("-- paired delta F1 (A vs B) 95% CI --")
    rows = []
    for name, _, fa, fb in pairs:
        d = boots[fa] - boots[fb]
        dpt = point_f1(*cache[fa]) - point_f1(*cache[fb])
        lo, hi = ci(d)
        rows.append((name, dpt, lo, hi))
        print(f"  {name:26s} d={dpt:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]")
    return rows


TEST = "test"
# (label, _, file_A, file_B)
pairs = [
    ("LoRA vs zero-shot",        TEST, "ner_lora_r8_fmt_test.json",    "ner_Qwen3-4B_fmt_test.json"),
    ("few-shot 3 vs zero-shot",  TEST, "ner_Qwen3-4B_fs3_fmt_test.json","ner_Qwen3-4B_fmt_test.json"),
    ("few-shot 3 vs few-shot 1", TEST, "ner_Qwen3-4B_fs3_fmt_test.json","ner_Qwen3-4B_fs1_fmt_test.json"),
    ("LoRA vs few-shot 3",       TEST, "ner_lora_r8_fmt_test.json",    "ner_Qwen3-4B_fs3_fmt_test.json"),
    ("DoRA vs LoRA",             TEST, "ner_dora_r8_fmt_test.json",    "ner_lora_r8_fmt_test.json"),
    ("LoRA vs BERTimbau base",   TEST, "ner_lora_r8_fmt_test.json",    "ner_bertimbau_base_test.json"),
    ("LoRA vs BERTimbau large",  TEST, "ner_lora_r8_fmt_test.json",    "ner_bertimbau_large_test.json"),
    ("BERTimbau large vs base",  TEST, "ner_bertimbau_large_test.json","ner_bertimbau_base_test.json"),
]
singles = [("SDFT r8", "ner_sdft_r8_full_budget_checkpoint-8205_test.json")]

run(pairs, singles, "H1 TEST")

# rank sweep on dev
dev_pairs = [
    ("r8 vs r4",  "dev", "ner_lora_r8_fmt_dev.json",  "ner_lora_r4_fmt_dev.json"),
    ("r16 vs r8", "dev", "ner_lora_r16_fmt_dev.json", "ner_lora_r8_fmt_dev.json"),
]
run(dev_pairs, [], "RANK SWEEP DEV")
