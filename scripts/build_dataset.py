#!/usr/bin/env python3
"""
Prepara o corpus UnB-KnEDLe DODF para o experimento do paper:

  1. Lê os 11 arquivos .conll (um por tipo de ato), no formato CoNLL/BIO.
  2. Divide 70/15/15 NO NÍVEL DE DOCUMENTO, estratificado por tipo de ato.
     - Estratificar por ato == dividir cada ato independentemente em 70/15/15,
       o que garante a presença dos 11 atos nos 3 conjuntos (objetivo do paper).
  3. Anti-vazamento: documentos com corpo idêntico (corpus templatizado) são
     agrupados e atribuídos SEMPRE ao mesmo split — nunca straddle train/test.
  4. Escreve o split em .conll (train/dev/test) preservando o formato original.
  5. Gera a versão utilizável para fine-tuning de decoder (Qwen) no padrão GNER:
     um prompt por tipo de entidade do ato, mesma sentença gera vários prompts.
     A saída é token-tagging inline BIO ("word(B-tipo) word(I-tipo) word(O) ..."),
     rotulando TODOS os tokens, mas apenas com o tipo consultado (demais -> O).
     Instâncias negativas (tipo ausente) têm output com (O) em todos os tokens.
     Estrutura de cada registro: {instruction, input, output, ...metadados}.

Determinístico (seed fixa). Sem dependências externas.
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
    """Retorna lista de documentos. Cada doc = dict(pub, tokens, tags)."""
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
                # novo documento começa aqui
                flush()
                pub = line.split(":", 1)[1].strip()
                continue
            parts = line.split()
            if len(parts) < 2:
                # token sem tag (raro) -> trata como 'O'
                tok, tag = parts[0], "O"
            else:
                tok, tag = parts[0], parts[-1]
            tokens.append(tok)
            tags.append(tag)
    flush()
    return docs


def doc_body_hash(doc):
    # Anti-vazamento por SUPERFÍCIE (apenas tokens). Hashear tokens+tags faz
    # docs com mesmo texto e anotação divergente caírem em splits diferentes,
    # vazando a mesma sentença entre train/dev/test.
    body = " ".join(doc["tokens"])
    return hashlib.md5(body.encode("utf-8")).hexdigest()


def entity_types_of(tags):
    types = set()
    for t in tags:
        if t != "O" and "-" in t:
            types.add(t.split("-", 1)[1])
    return types


def extract_spans(tokens, tags, etype):
    """Spans (superfície) do tipo etype, via chunks BIO."""
    spans, cur = [], []
    for tok, tag in zip(tokens, tags):
        if tag == f"B-{etype}":
            if cur:
                spans.append(" ".join(cur))
            cur = [tok]
        elif tag == f"I-{etype}":
            if cur:
                cur.append(tok)
            else:  # I- sem B- precedente: inicia chunk mesmo assim
                cur = [tok]
        else:
            if cur:
                spans.append(" ".join(cur))
                cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def stratified_split(docs):
    """Divide os docs de UM ato em (train, dev, test) por documento,
    mantendo duplicatas no mesmo split. Determinístico."""
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
    """Descrição do formato de saída esperado (esquema BIO + sintaxe token(rótulo)).

    Gravada no dataset (campo instruction_fmt) para que treino e avaliação usem
    exatamente o mesmo prompt. Não mostra nenhum token real rotulado — só a *forma*
    da saída (placeholders sintáticos token1/token2/…), descrevendo o formato sem
    funcionar como exemplo rotulado.
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
    """Token-tagging inline BIO: 'word(B-tipo) word(I-tipo) word(O) ...'.
    Rotula TODOS os tokens, mas só com o tipo consultado (demais -> O)."""
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
    """Um registro por tipo de entidade do ato (instâncias negativas inclusas).
    Formato GNER: instruction/input/output com token-tagging inline."""
    text = " ".join(doc["tokens"])
    recs = []
    for etype in schema:
        spans = extract_spans(doc["tokens"], doc["tags"], etype)
        output = render_gner_output(doc["tokens"], doc["tags"], etype)
        instruction = f"Extraia todas as ocorrências da entidade {etype} do texto."
        recs.append({
            # `instruction`     : enunciado puro (usado como demo no few-shot).
            # `instruction_fmt` : enunciado + descrição do formato de saída (fmt).
            #                     É o prompt canônico de treino/avaliação do decoder.
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
    assert files, f"Nenhum .conll em {CORPUS_DIR}"

    # 1) esquema de entidades por ato (sobre TODOS os docs do ato)
    # 2) split por ato
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

    # ---- escreve .conll ----
    for name in ("train", "dev", "test"):
        out = os.path.join(CONLL_OUT, f"{name}.conll")
        with open(out, "w", encoding="utf-8") as fh:
            for act, d in splits[name]:
                write_conll(fh, [d], act)

    # ---- escreve JSONL GNER para fine-tuning ----
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

    # ---- schema + estatísticas ----
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

    # ---- relatório no terminal ----
    print(f"{'Ato':32s} {'train':>6} {'dev':>5} {'test':>5} {'total':>6}")
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
              f"({c['negatives']} negativos, "
              f"{100*c['negatives']/c['records']:.1f}%)")
    print(f"\nSaída: {OUT_DIR}")


if __name__ == "__main__":
    main()
