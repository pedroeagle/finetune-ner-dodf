#!/usr/bin/env python3
"""Prepare the UnB-KnEDLe DODF corpus for the paper's experiment:

  1. Read the 11 .conll files (one per act type), in CoNLL/BIO format.
  2. Split 70/15/15 AT THE DOCUMENT LEVEL, stratified by act type.
     - Stratifying by act == splitting each act independently into 70/15/15,
       which guarantees all 11 acts appear in the 3 sets (the paper's goal).
  3. Anti-leakage: documents with identical bodies (templated corpus) are
     grouped and always assigned to the same split — never straddling train/test.
  4. Write the split as .conll (train/dev/test) preserving the original format.
  5. Generate the decoder-usable version for fine-tuning (Qwen) in the GNER style:
     one prompt per entity type of the act, so the same sentence yields several
     prompts. The output is inline BIO token-tagging ("word(B-type) word(I-type)
     word(O) ..."), labeling ALL tokens but only with the queried type (others -> O).
     Negative instances (type absent) have an output with (O) on every token.
     Each record's structure: {instruction, input, output, ...metadata}.

Deterministic (fixed seed). No external dependencies.
"""

import glob
import hashlib
import json
import os
import random
from collections import defaultdict, OrderedDict

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, dev, test

CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "lre-dodfpcorpus-main", "corpus",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "lre-dodfpcorpus-main", "splits",
)
CONLL_OUT = os.path.join(OUT_DIR, "conll")
FT_OUT = os.path.join(OUT_DIR, "finetune")


def act_name_from_file(path):
    base = os.path.basename(path)
    name = base[:-len(".conll")]
    if name.startswith("Ato_"):
        name = name[len("Ato_"):]
    return name


def parse_conll(path):
    """Return a list of documents. Each doc = dict(pub, tokens, tags)."""
    docs = []
    pub = None
    tokens, tags = [], []

    def flush():
        nonlocal pub, tokens, tags
        if tokens:
            docs.append({"pub": pub, "tokens": tokens, "tags": tags})
        pub, tokens, tags = None, [], []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.strip() == "":
                flush()
                continue
            if line.startswith("Publication:"):
                # a new document starts here
                flush()
                pub = line.split(":", 1)[1].strip()
                continue
            parts = line.split()
            if len(parts) < 2:
                # token without a tag (rare) -> treat as 'O'
                tok, tag = parts[0], "O"
            else:
                tok, tag = parts[0], parts[-1]
            tokens.append(tok)
            tags.append(tag)
    flush()
    return docs


def doc_body_hash(doc):
    # Anti-leakage by SURFACE (tokens only). Hashing tokens+tags would make docs
    # with the same text but divergent annotation fall into different splits,
    # leaking the same sentence across train/dev/test.
    body = " ".join(doc["tokens"])
    return hashlib.md5(body.encode("utf-8")).hexdigest()


def entity_types_of(tags):
    types = set()
    for t in tags:
        if t != "O" and "-" in t:
            types.add(t.split("-", 1)[1])
    return types


def extract_spans(tokens, tags, etype):
    """Spans (surface form) of type etype, via BIO chunks."""
    spans, cur = [], []
    for tok, tag in zip(tokens, tags):
        if tag == f"B-{etype}":
            if cur:
                spans.append(" ".join(cur))
            cur = [tok]
        elif tag == f"I-{etype}":
            if cur:
                cur.append(tok)
            else:  # I- without a preceding B-: start a chunk anyway
                cur = [tok]
        else:
            if cur:
                spans.append(" ".join(cur))
                cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def stratified_split(docs):
    """Split ONE act's docs into (train, dev, test) by document, keeping
    duplicates in the same split. Deterministic."""
    groups = OrderedDict()  # hash -> [docs]
    for d in docs:
        groups.setdefault(doc_body_hash(d), []).append(d)

    units = list(groups.values())
    rng = random.Random(SEED)
    rng.shuffle(units)

    n = len(docs)
    n_test = round(RATIOS[2] * n)
    n_dev = round(RATIOS[1] * n)
    n_train = n - n_dev - n_test

    train, dev, test = [], [], []
    for unit in units:
        if len(train) < n_train:
            train.extend(unit)
        elif len(dev) < n_dev:
            dev.extend(unit)
        else:
            test.extend(unit)
    return train, dev, test


def write_conll(fh, docs, act):
    for d in docs:
        fh.write(f"Publication: {d['pub']}\n")
        fh.write(f"Act: {act}\n")
        for tok, tag in zip(d["tokens"], d["tags"]):
            fh.write(f"{tok} {tag}\n")
        fh.write("\n")


def format_instructions(etype):
    """Description of the expected output format (BIO scheme + token(label) syntax).

    Written into the dataset (field instruction_fmt) so training and evaluation use
    exactly the same prompt. Shows no real labeled token — only the *shape* of the
    output (syntactic placeholders token1/token2/…), describing the format without
    acting as a labeled example. The returned text is Portuguese: it is part of the
    dataset (the actual prompt), not translatable UI text.
    """
    return (
        "Reproduza o texto inteiro, token a token (separados por espaço), na "
        "mesma ordem e sem omitir nenhum token. Logo após cada token, anexe "
        "entre parênteses, sem espaço, o seu rótulo BIO:\n"
        f"- B-{etype} no primeiro token de cada ocorrência da entidade {etype};\n"
        f"- I-{etype} nos demais tokens da mesma ocorrência;\n"
        "- O em todo token que não pertence à entidade.\n"
        "Formato esperado (esquema sintático, não exemplo de conteúdo): "
        f"token1(O) token2(B-{etype}) token3(I-{etype}) token4(O)"
    )


def render_gner_output(tokens, tags, etype):
    """Inline BIO token-tagging: 'word(B-type) word(I-type) word(O) ...'.
    Labels ALL tokens, but only with the queried type (others -> O)."""
    out = []
    for tok, tag in zip(tokens, tags):
        if tag == f"B-{etype}":
            lab = f"B-{etype}"
        elif tag == f"I-{etype}":
            lab = f"I-{etype}"
        else:
            lab = "O"
        out.append(f"{tok}({lab})")
    return " ".join(out)


def build_gner_records(doc, act, schema, split):
    """One record per entity type of the act (negative instances included).
    GNER format: instruction/input/output with inline token-tagging."""
    text = " ".join(doc["tokens"])
    recs = []
    for etype in schema:
        spans = extract_spans(doc["tokens"], doc["tags"], etype)
        output = render_gner_output(doc["tokens"], doc["tags"], etype)
        instruction = f"Extraia todas as ocorrências da entidade {etype} do texto."
        recs.append({
            # `instruction`     : bare task statement (used as the few-shot demo).
            # `instruction_fmt` : statement + output-format description (fmt).
            #                     The canonical decoder train/eval prompt.
            "instruction": instruction,
            "instruction_fmt": f"{instruction}\n\n{format_instructions(etype)}",
            "input": text,
            "output": output,
            "act": act,
            "entity_type": etype,
            "publication": doc["pub"],
            "is_negative": len(spans) == 0,
            "split": split,
        })
    return recs


def main():
    os.makedirs(CONLL_OUT, exist_ok=True)
    os.makedirs(FT_OUT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.conll")))
    assert files, f"No .conll files in {CORPUS_DIR}"

    # 1) entity schema per act (over ALL the act's docs)
    # 2) split per act
    splits = {"train": [], "dev": [], "test": []}          # (act, doc)
    schema_by_act = {}
    stats = defaultdict(lambda: {"train": 0, "dev": 0, "test": 0, "total": 0})

    for path in files:
        act = act_name_from_file(path)
        docs = parse_conll(path)
        schema = sorted(set().union(*[entity_types_of(d["tags"]) for d in docs]))
        schema_by_act[act] = schema

        tr, dv, te = stratified_split(docs)
        for name, part in (("train", tr), ("dev", dv), ("test", te)):
            for d in part:
                splits[name].append((act, d))
            stats[act][name] = len(part)
        stats[act]["total"] = len(docs)

    # deterministic ordering inside each split (by act, then publication)
    for name in splits:
        splits[name].sort(key=lambda x: (x[0], x[1]["pub"]))

    # ---- write .conll ----
    for name in ("train", "dev", "test"):
        out = os.path.join(CONLL_OUT, f"{name}.conll")
        with open(out, "w", encoding="utf-8") as fh:
            for act, d in splits[name]:
                write_conll(fh, [d], act)

    # ---- write GNER JSONL for fine-tuning ----
    ft_counts = {}
    for name in ("train", "dev", "test"):
        out = os.path.join(FT_OUT, f"{name}.jsonl")
        n_rec = n_neg = 0
        with open(out, "w", encoding="utf-8") as fh:
            for act, d in splits[name]:
                for rec in build_gner_records(d, act, schema_by_act[act], name):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rec += 1
                    n_neg += int(rec["is_negative"])
        ft_counts[name] = {"records": n_rec, "negatives": n_neg}

    # ---- schema + statistics ----
    with open(os.path.join(OUT_DIR, "entity_schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema_by_act, fh, ensure_ascii=False, indent=2)

    totals = {"train": 0, "dev": 0, "test": 0, "total": 0}
    for act in stats:
        for k in totals:
            totals[k] += stats[act][k]

    summary = {
        "seed": SEED,
        "ratios": {"train": RATIOS[0], "dev": RATIOS[1], "test": RATIOS[2]},
        "doc_counts_per_act": dict(stats),
        "doc_totals": totals,
        "finetune_record_counts": ft_counts,
    }
    with open(os.path.join(OUT_DIR, "split_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- terminal report ----
    print(f"{'Act':32s} {'train':>6} {'dev':>5} {'test':>5} {'total':>6}")
    print("-" * 60)
    for act in sorted(stats):
        s = stats[act]
        print(f"{act:32s} {s['train']:6d} {s['dev']:5d} {s['test']:5d} {s['total']:6d}")
    print("-" * 60)
    print(f"{'TOTAL':32s} {totals['train']:6d} {totals['dev']:5d} "
          f"{totals['test']:5d} {totals['total']:6d}")
    print()
    for name in ("train", "dev", "test"):
        c = ft_counts[name]
        print(f"GNER {name:5s}: {c['records']:7d} prompts "
              f"({c['negatives']} negatives, "
              f"{100*c['negatives']/c['records']:.1f}%)")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
