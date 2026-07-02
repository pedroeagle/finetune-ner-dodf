#!/usr/bin/env python3
"""Avalia o BERTimbau (encoder, classificação de tokens) no NER do DODF.

O encoder prediz em nível de documento (todos os tipos de entidade de uma vez).
Para tornar a comparação com o decoder Qwen3-4B livre de viés, a predição é
FATIADA exatamente nos mesmos registros (publicação, ato, entity_type) que o
decoder avaliou — lendo o test.jsonl e mantendo, em cada registro, apenas os
rótulos do entity_type consultado. O pool resultante reproduz, token a token, o
mesmo gold e a mesma métrica seqeval do evaluate_ner.py.

Saída: JSON no mesmo schema do evaluate_ner.py, compatível com `--compare`.

Uso:
  python experiments/evaluate_bertimbau.py --model results/checkpoints/bertimbau_base
  python experiments/evaluate_bertimbau.py --model results/checkpoints/bertimbau_large

  # ΔF1 + bootstrap pareado contra o decoder (mesmo comando do evaluate_ner.py):
  python experiments/evaluate_ner.py --compare \\
      results/ner_bertimbau_base_test.json results/ner_lora_r8_fmt_test.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from evaluate_ner import (
    _NumpyEncoder,
    align_to_gold,
    bootstrap_f1,
    init_distributed,
    overall_metrics,
    parse_inline_bio,
    per_act_metrics,
    per_entity_metrics,
)
from train_bertimbau import read_conll

ROOT = Path(__file__).parent.parent
CONLL_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/conll"
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"
RESULTS_DIR = ROOT / "results"


def predict_docs(model, tokenizer, docs: list[dict], device, max_length: int,
                 batch_size: int, rank: int = 0) -> dict[tuple, list[str]]:
    """Prediz rótulos por palavra para cada documento (publicação, ato).

    Cada palavra recebe o rótulo do seu 1º sub-token. Palavras além do limite de
    max_length (truncadas) ficam como 'O' — caso raro (~60 tokens/doc em média).
    """
    model.eval()
    id2label = model.config.id2label
    preds: dict[tuple, list[str]] = {}

    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        if start % (batch_size * 20) == 0:
            print(f"  [rank {rank}] [{start:>6}/{len(docs)}]")

        enc = tokenizer(
            [d["tokens"] for d in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()

        for i, d in enumerate(batch):
            word_ids = enc.word_ids(batch_index=i)
            word_labels = ["O"] * len(d["tokens"])
            prev = None
            for pos, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    prev = wid
                    continue
                word_labels[wid] = id2label[int(pred_ids[i][pos])]
                prev = wid
            preds[(d["publication"], d["act"])] = word_labels

    return preds


def slice_to_entity(doc_labels: list[str], entity_type: str) -> list[str]:
    """Mantém apenas os rótulos B-/I- do entity_type consultado; resto vira 'O'.

    Reproduz a formulação por-entidade do decoder: fatiar a predição global por
    tipo e juntar os fatias dá exatamente o mesmo conjunto de entidades que a
    avaliação em nível de documento (cada span é contado uma vez no seu tipo)."""
    keep = {f"B-{entity_type}", f"I-{entity_type}"}
    return [lab if lab in keep else "O" for lab in doc_labels]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Checkpoint do BERTimbau treinado")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    docs = read_conll(CONLL_DIR / f"{args.split}.conll")
    tag = Path(args.model).name
    out_path = args.output or str(RESULTS_DIR / f"ner_{tag}_{args.split}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cada rank processa 1/world_size dos documentos
    docs_slice = docs[rank::world_size]
    if rank == 0:
        print(f"Split '{args.split}': {len(docs):,} documentos "
              f"({world_size} GPU(s)) | encoder token-classification")
        print("Gerando predições por documento…")
    slice_preds = predict_docs(model, tokenizer, docs_slice, device,
                               args.max_length, args.batch_size, rank)

    # Salva predições parciais por rank (chave tupla → lista [pub, act])
    tmp_path = out_path + f".rank{rank}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump([[list(k), v] for k, v in slice_preds.items()], f, ensure_ascii=False)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if rank != 0:
        return

    # Rank 0 agrega as predições de todos os ranks
    doc_preds: dict[tuple, list[str]] = {}
    for r in range(world_size):
        tmp = out_path + f".rank{r}"
        with open(tmp, encoding="utf-8") as f:
            for key, labels in json.load(f):
                doc_preds[tuple(key)] = labels
        os.remove(tmp)

    # Fatia a predição global nos MESMOS registros (pub, ato, entity_type) do decoder.
    records: list[dict] = []
    with open(FT_DIR / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    predictions = []
    missing = 0
    for rec in records:
        key = (rec["publication"], rec["act"])
        doc_labels = doc_preds.get(key)
        if doc_labels is None:
            missing += 1
            doc_labels = ["O"] * len(rec["input"].split())
        _, gold_labels = parse_inline_bio(rec["output"])
        pred_sliced = slice_to_entity(doc_labels, rec["entity_type"])
        pred_labels = align_to_gold(pred_sliced, len(gold_labels))
        predictions.append(
            {
                "publication": rec["publication"],
                "act": rec["act"],
                "entity_type": rec["entity_type"],
                "is_negative": rec["is_negative"],
                "gold": gold_labels,
                "pred": pred_labels,
            }
        )
    if missing:
        print(f"AVISO: {missing} registros sem documento conll correspondente (pred='O').")

    metrics = overall_metrics(predictions)
    per_act = per_act_metrics(predictions)
    per_entity = per_entity_metrics(predictions)
    boot = bootstrap_f1(predictions)

    print(f"\nF1 (default) : {metrics['f1']:.4f}")
    print(f"F1 (strict)  : {metrics['strict']['f1']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f} (strict {metrics['strict']['precision']:.4f})")
    print(f"Recall       : {metrics['recall']:.4f} (strict {metrics['strict']['recall']:.4f})")
    print(f"IC 95% (F1)  : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    print(f"\nPor ato:\n{'Ato':42s} {'F1':>7} {'F1str':>7} {'P':>7} {'R':>7} {'N':>6}")
    print("-" * 80)
    for act, m in per_act.items():
        print(f"{act:42s} {m['f1']:7.4f} {m['f1_strict']:7.4f} "
              f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_records']:6d}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "split": args.split,
                "metrics": metrics,
                "per_act": per_act,
                "per_entity": per_entity,
                "bootstrap": boot,
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
            cls=_NumpyEncoder,
        )
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
