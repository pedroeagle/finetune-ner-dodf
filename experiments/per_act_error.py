#!/usr/bin/env python3
"""Per-act-type error analysis (paper Section "Per-entity analysis",
complementary breakdown by document/act type instead of entity type).

Pure re-analysis of saved predictions (results/ner_*_test.json). No model run.

Reports, for each of the 11 act types:
  - strict entity-level micro F1 for LoRA and BERTimbau-large, and the gap;
  - decomposition of LoRA's handling of gold spans into exact / boundary /
    omission, to show where the omission failure concentrates.
"""
import argparse, json, os
from collections import defaultdict

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"),
                 help="Directory with the ner_*_test.json files (default: ../results)")
RESULTS = _ap.parse_args().results


def ents(tags):
    out, cur, st = set(), None, None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur is not None:
                out.add((cur, st, i))
            cur, st = t[2:], i
        elif t.startswith("I-"):
            if cur is not None and t[2:] == cur:
                continue
            if cur is not None:
                out.add((cur, st, i))
            cur, st = None, None
        else:
            if cur is not None:
                out.add((cur, st, i))
            cur, st = None, None
    if cur is not None:
        out.add((cur, st, len(tags)))
    return out


def per_act(fname):
    d = json.load(open(os.path.join(RESULTS, fname)))["predictions"]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for ex in d:
        a = ex["act"]
        g, p = ents(ex["gold"]), ents(ex["pred"])
        tp[a] += len(g & p); fp[a] += len(p - g); fn[a] += len(g - p)
    f1, sup = {}, {}
    for a in set(list(tp) + list(fn)):
        P = tp[a] / (tp[a] + fp[a]) if (tp[a] + fp[a]) else 0.0
        R = tp[a] / (tp[a] + fn[a]) if (tp[a] + fn[a]) else 0.0
        f1[a] = 2 * P * R / (P + R) if (P + R) else 0.0
        sup[a] = tp[a] + fn[a]
    return f1, sup


def omission_by_act(fname):
    d = json.load(open(os.path.join(RESULTS, fname)))["predictions"]
    dec = defaultdict(lambda: [0, 0, 0])  # exact, boundary, omit
    for ex in d:
        a = ex["act"]
        g, p = ents(ex["gold"]), ents(ex["pred"])
        for (t, s, e) in g:
            if (t, s, e) in p:
                dec[a][0] += 1
            elif any(pt == t and not (pe <= s or ps >= e) for (pt, ps, pe) in p):
                dec[a][1] += 1
            else:
                dec[a][2] += 1
    return dec


lf, ls = per_act("ner_lora_r8_fmt_test.json")
bf, _ = per_act("ner_bertimbau_large_test.json")
dec = omission_by_act("ner_lora_r8_fmt_test.json")

order = sorted(ls, key=lambda a: -ls[a])
print(f"{'act':30s} {'sup':>6s} {'LoRA':>6s} {'BERT-L':>7s} {'gap':>7s} "
      f"{'exact':>6s} {'bound':>6s} {'omit':>6s}")
for a in order:
    ex, bo, om = dec[a]
    tot = ex + bo + om or 1
    print(f"{a:30s} {ls[a]:6d} {lf[a]:6.3f} {bf[a]:7.3f} {lf[a]-bf[a]:+7.3f} "
          f"{ex/tot:6.1%} {bo/tot:6.1%} {om/tot:6.1%}")

# support-weighted summary
tot_sup = sum(ls.values())
wgap = sum((lf[a] - bf[a]) * ls[a] for a in ls) / tot_sup
allex = sum(dec[a][0] for a in dec)
allbo = sum(dec[a][1] for a in dec)
allom = sum(dec[a][2] for a in dec)
allt = allex + allbo + allom
print(f"\nsupport-weighted gap LoRA-BERT = {wgap:+.3f}")
print(f"overall spans: exact={allex/allt:.1%} boundary={allbo/allt:.1%} "
      f"omission={allom/allt:.1%}  (n={allt})")
