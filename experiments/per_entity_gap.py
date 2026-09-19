#!/usr/bin/env python3
"""Quantify the encoder-decoder gap concentration and the omission pattern
(paper Section "Per-entity analysis", strengthening finding S3).

Pure re-analysis of saved predictions (results/ner_*_test.json). No model run.

Reports, at the aggregate entity-type level (42 types, matching Table 4):
  - support-weighted mean gap F1(LoRA) - F1(BERTimbau-large) for the
    template-regular group (LoRA F1 > 0.90) vs the ten free-form entities
    (F1 < 0.60);
  - decomposition of LoRA's handling of gold spans into exact / boundary /
    omission for each group.
"""
import argparse, json, os
from collections import defaultdict

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"),
                 help="Directory with the ner_*_test.json files (default: ../results)")
RESULTS = _ap.parse_args().results
FREE_FORM = ["informacao_corrigida", "simbolo_substituto", "vigencia",
             "informacao_errada", "carreira", "fundamento_legal",
             "cargo_orgao_cessionario", "hierarquia_lotacao",
             "cargo_objeto_substituicao", "orgao_cessionario"]


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


def per_type(fname):
    d = json.load(open(os.path.join(RESULTS, fname)))["predictions"]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for ex in d:
        et = ex["entity_type"]
        g, p = ents(ex["gold"]), ents(ex["pred"])
        tp[et] += len(g & p); fp[et] += len(p - g); fn[et] += len(g - p)
    f1, sup = {}, {}
    for et in set(list(tp) + list(fn)):
        P = tp[et] / (tp[et] + fp[et]) if (tp[et] + fp[et]) else 0.0
        R = tp[et] / (tp[et] + fn[et]) if (tp[et] + fn[et]) else 0.0
        f1[et] = 2 * P * R / (P + R) if (P + R) else 0.0
        sup[et] = tp[et] + fn[et]
    return f1, sup


def omission(fname, subset):
    d = json.load(open(os.path.join(RESULTS, fname)))["predictions"]
    exact = boundary = omit = 0
    for ex in d:
        if ex["entity_type"] not in subset:
            continue
        g, p = ents(ex["gold"]), ents(ex["pred"])
        for (t, s, e) in g:
            if (t, s, e) in p:
                exact += 1
            elif any(pt == t and not (pe <= s or ps >= e) for (pt, ps, pe) in p):
                boundary += 1
            else:
                omit += 1
    tot = exact + boundary + omit
    return exact, boundary, omit, tot


lf, ls = per_type("ner_lora_r8_fmt_test.json")
bf, _ = per_type("ner_bertimbau_large_test.json")
tmpl = [e for e in lf if lf[e] > 0.90 and ls[e] > 0]


def wgap(es):
    num = sum((lf[e] - bf[e]) * ls[e] for e in es)
    den = sum(ls[e] for e in es)
    return num / den if den else 0.0


for name, g in [("template (LoRA F1>0.90)", tmpl), ("free-form (F1<0.60)", FREE_FORM)]:
    print(f"{name:26s} n={len(g):2d} support={sum(ls[e] for e in g):5d} "
          f"weighted gap LoRA-BERT = {wgap(g):+.3f}")
for name, ss in [("template", set(tmpl)), ("free-form", set(FREE_FORM))]:
    ex, bo, om, tot = omission("ner_lora_r8_fmt_test.json", ss)
    print(f"{name:9s}: gold spans={tot:5d} exact={ex/tot:.1%} "
          f"boundary={bo/tot:.1%} omission={om/tot:.1%}")
