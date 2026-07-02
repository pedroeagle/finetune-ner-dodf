#!/usr/bin/env python3
"""Avalia NER generativo: saída inline BIO → seqeval → métricas (H1).

Suporta inferência distribuída via torchrun: cada processo carrega o modelo
completo em seu GPU e processa 1/N do dataset; rank 0 agrega e calcula métricas.

Uso:
  # modelo fine-tuned (adaptador PEFT detectado automaticamente)
  torchrun --nproc_per_node=8 experiments/evaluate_ner.py --model results/checkpoints/lora_r8

  # baseline zero-shot
  torchrun --nproc_per_node=8 experiments/evaluate_ner.py --model Qwen/Qwen3-4B

  # comparar dois resultados (ΔF1 + bootstrap) — roda localmente sem GPU
  python experiments/evaluate_ner.py \\
      --compare results/ner_lora_r8_fmt_test.json results/ner_Qwen3-4B_fmt_test.json

O formato de saída esperado é gravado no dataset (campo instruction_fmt, ver
scripts/build_dataset.py) e usado na query; os resultados saem com sufixo _fmt
(ex.: ner_lora_r8_fmt_test.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import torch
import torch._dynamo
torch._dynamo.disable()

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"
RESULTS_DIR = ROOT / "results"

_UNIT_RE = re.compile(r"^(.+)\(([BI]-[\w]+|O)\)$")
CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
CHATML_PAIR = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>\n"
)


# ---------------------------------------------------------------------------
# Setup distribuído
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """Inicializa NCCL quando rodando via torchrun. Retorna (rank, world_size, local_rank)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(hours=4),
        )
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Carregamento de modelo
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, device: torch.device,
                             dtype: torch.dtype = torch.bfloat16):
    """Carrega base (+ adapter, se houver) garantindo dtype uniforme.

    O dtype é aplicado de forma idêntica ao base e ao fine-tuned, para que a
    comparação do H1 seja justa. Usa `torch_dtype=` (não `dtype=`, que versões
    antigas do transformers ignoram silenciosamente, caindo em fp32).
    """
    adapter_cfg = Path(model_path) / "adapter_config.json"

    if adapter_cfg.exists():
        from peft import PeftModel

        with open(adapter_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        print(f"Adaptador PEFT detectado. Base: {base} | dtype={dtype}")

        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)
        model = PeftModel.from_pretrained(model, model_path)
        # Adapter pode vir salvo em fp32; uniformiza base+adapter no mesmo dtype.
        model = model.to(dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device).to(dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Parsing do formato GNER inline
# ---------------------------------------------------------------------------

def parse_inline_bio(text: str) -> tuple[list[str], list[str]]:
    """Converte 'tok1(B-etype) tok2(I-etype) tok3(O) …' em (tokens, labels)."""
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
# Inferência
# ---------------------------------------------------------------------------

def load_few_shot_demos(n: int, seed: int = 42) -> dict[str, list[dict]]:
    """Seleciona até n demos por (ato, entity_type) do split de treino.

    Retorna dict["act/entity_type" → list[record]]. Em inferência, cada exemplo
    de teste recebe demos do seu próprio (ato, entity_type) — o modelo vê o
    formato de saída e os labels exatos para aquela entidade naquele ato.
    Vêm do treino (nunca do test) para evitar vazamento. Determinístico por seed.
    """
    recs = []
    with open(FT_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))

    by_key: dict[str, list[dict]] = defaultdict(list)
    for rec in recs:
        if rec.get("is_negative"):
            continue  # demos negativos ensinam "não há entidade" — excluir
        key = f"{rec['act']}/{rec['entity_type']}"
        by_key[key].append(rec)

    rng = random.Random(seed)
    result: dict[str, list[dict]] = {}
    for key, pool in sorted(by_key.items()):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        result[key] = shuffled[:n]
    return result


def build_prompt(record: dict, tokenizer, demos: list[dict] | None = None) -> str:
    """Prompt de inferência idêntico ao formato de treino (train.py:format_as_chat).

    Usa o MESMO apply_chat_template com enable_thinking=False, só que com
    add_generation_prompt=True. Isso garante que:
      - a convenção do bloco <think>…</think> (se o template do Qwen3 a inserir)
        seja idêntica entre treino e inferência;
      - o modelo base não entre em thinking mode e esgote os max_new_tokens.

    A query usa `instruction_fmt` (enunciado + descrição do formato de saída),
    gravado no dataset por build_dataset.py.

    Com `demos`, prepende pares user/assistant (few-shot) antes da query final,
    mostrando ao modelo o formato de saída palavra(rótulo). As demonstrações usam
    `instruction` puro (sem o bloco de formato): o formato já fica visível na
    resposta de cada demo.
    """
    def _content(rec: dict, fmt: bool = False) -> str:
        instruction = rec["instruction_fmt"] if fmt else rec["instruction"]
        return f"{instruction}\n\n{rec['input']}"

    messages = []
    for d in (demos or []):
        messages.append({"role": "user", "content": _content(d)})
        messages.append({"role": "assistant", "content": d["output"]})
    messages.append({"role": "user", "content": _content(record, fmt=True)})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        pairs = "".join(
            CHATML_PAIR.format(user=_content(d), assistant=d["output"])
            for d in (demos or [])
        )
        return pairs + CHATML_USER.format(content=_content(record, fmt=True))


def generate_predictions(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    rank: int = 0,
    demo_pool: dict[str, list[dict]] | None = None,
) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    # left-truncation preserva a query (fim do prompt) se exceder o limite.
    if demo_pool:
        # Escala com o número de shots: +1024 por demo. +160 cobre o bloco de
        # formato (~139 tokens fixos) que a query final sempre carrega.
        n_shot = max(len(v) for v in demo_pool.values())
        max_length = 1024 * (1 + n_shot) + 160
    else:
        # Ramo zero-shot/FT. Teto 1536 (não 1024): no test o input é curto e
        # padding=True enche só até o maior prompt do batch, então o teto não
        # custa nada — H1 não é afetado. No dev cobre a cauda longa e reduz
        # prompts truncados. Documentos além de 1536 tokens ainda truncam, mas
        # o corte é simétrico entre os ranks, mantendo a seleção de rank válida.
        n_shot = 0
        max_length = 1536
    tokenizer.truncation_side = "left"
    results = []

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]

        if batch_start % 200 == 0:
            print(f"  [rank {rank}] [{batch_start:>6}/{len(records)}]")

        def _demos(rec: dict) -> list[dict] | None:
            if not demo_pool:
                return None
            return demo_pool.get(f"{rec['act']}/{rec['entity_type']}") or None

        prompts = [build_prompt(rec, tokenizer, _demos(rec)) for rec in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, rec in enumerate(batch):
            generated = tokenizer.decode(
                out_ids[i][input_len:],
                skip_special_tokens=True,
            ).strip()

            if rank == 0 and batch_start == 0 and i < 3:
                raw_ids = out_ids[i][input_len:].tolist()
                prompt_decoded = tokenizer.decode(out_ids[i][:input_len], skip_special_tokens=False)
                generated_raw = tokenizer.decode(out_ids[i][input_len:], skip_special_tokens=False)
                print(f"\n--- DEBUG exemplo {i} ---")
                print(f"  gold[:80]       : {rec['output'][:80]!r}")
                print(f"  pred[:80]       : {generated[:80]!r}")
                print(f"  pred_raw[:80]   : {generated_raw[:80]!r}")
                print(f"  token_ids[:20]  : {raw_ids[:20]}")
                print(f"  input_len       : {input_len}")
                print(f"  prompt[-100:]   : {prompt_decoded[-100:]!r}")
                print(f"  pred tokens gerados: {out_ids[i].shape[0] - input_len}")

            _, gold_labels = parse_inline_bio(rec["output"])
            _, pred_labels = parse_inline_bio(generated)
            pred_labels = align_to_gold(pred_labels, len(gold_labels))

            results.append(
                {
                    "publication": rec["publication"],
                    "act": rec["act"],
                    "entity_type": rec["entity_type"],
                    "is_negative": rec["is_negative"],
                    "gold": gold_labels,
                    "pred": pred_labels,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Métricas
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
    # F1 strict IOB2 — mesma métrica canônica do ponto estimado e da comparação.
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
# Comparação de dois resultados (ΔF1 + bootstrap)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def compare(path_a: str, path_b: str) -> None:
    _log(f"Carregando {Path(path_a).name} …")
    with open(path_a, encoding="utf-8") as f:
        da = json.load(f)
    _log(f"Carregando {Path(path_b).name} …")
    with open(path_b, encoding="utf-8") as f:
        db = json.load(f)

    # Métrica canônica do H1 = F1 strict IOB2 (mesma do bootstrap, via _spans).
    # Cai para o F1 default se o campo strict não estiver presente no JSON.
    def _f1_strict(d: dict) -> float:
        return d["metrics"].get("strict", {}).get("f1", d["metrics"]["f1"])

    fa = _f1_strict(da)
    fb = _f1_strict(db)
    tag_a = da.get("model", Path(path_a).stem)
    tag_b = db.get("model", Path(path_b).stem)

    print(f"\nModelo A: {tag_a}  →  F1 strict = {fa:.4f}")
    print(f"Modelo B: {tag_b}  →  F1 strict = {fb:.4f}")
    print(f"ΔF1 (A − B) = {fa - fb:+.4f}")

    if "predictions" not in da or "predictions" not in db:
        print("(predições individuais não disponíveis; bootstrap não executado)")
        return

    preds_a = da["predictions"]
    preds_b = db["predictions"]
    if len(preds_a) != len(preds_b):
        print("Aviso: conjuntos de tamanhos diferentes; bootstrap ignorado.")
        return

    # Pré-computa TP/FP/FN por amostra para evitar rodar seqeval 1000×.
    # Bootstrap = sortear índices e somar inteiros → F1 em O(n_iter) em vez de O(n·n_iter).
    def _spans(labels: list[str]) -> set[tuple]:
        spans, cur_type, cur_start = set(), None, None
        for i, lbl in enumerate(labels):
            if lbl.startswith("B-"):
                if cur_type is not None:
                    spans.add((cur_start, i - 1, cur_type))
                cur_type, cur_start = lbl[2:], i
            elif lbl.startswith("I-") and cur_type == lbl[2:]:
                pass
            else:
                if cur_type is not None:
                    spans.add((cur_start, i - 1, cur_type))
                cur_type, cur_start = None, None
        if cur_type is not None:
            spans.add((cur_start, len(labels) - 1, cur_type))
        return spans

    def _tpfpfn(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
        g, p = _spans(gold), _spans(pred)
        tp = len(g & p)
        return tp, len(p) - tp, len(g) - tp

    _log(f"Pré-computando TP/FP/FN (n={len(preds_a)}) …")
    counts_a = np.array([_tpfpfn(r["gold"], r["pred"]) for r in preds_a], dtype=np.int32)
    counts_b = np.array([_tpfpfn(r["gold"], r["pred"]) for r in preds_b], dtype=np.int32)

    def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    _log(f"Bootstrap (1000 iterações) …")
    rng_np = np.random.default_rng(42)
    n = len(preds_a)
    deltas = []
    for i in range(1000):
        if i % 200 == 0:
            _log(f"  bootstrap {i}/1000 …")
        idx = rng_np.integers(0, n, size=n)
        sa = counts_a[idx].sum(axis=0)
        sb = counts_b[idx].sum(axis=0)
        deltas.append(_f1_from_counts(*sa) - _f1_from_counts(*sb))
    deltas.sort()
    lo, hi = deltas[25], deltas[975]

    print(f"IC 95% do ΔF1: [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("→ H1 sustentada: IC inteiramente positivo.")
    else:
        print("→ H1 não sustentada: IC inclui zero ou valores negativos.")

    _log("Calculando métricas por ato …")
    acts = sorted({r["act"] for r in preds_a})
    print(f"\n{'Ato':42s} {'F1 A':>7} {'F1 B':>7} {'Δ':>7}")
    print("-" * 65)
    by_act_a: dict = defaultdict(list)
    by_act_b: dict = defaultdict(list)
    for r in preds_a:
        by_act_a[r["act"]].append(r)
    for r in preds_b:
        by_act_b[r["act"]].append(r)
    per_act = {}
    for act in acts:
        ga, pa = [r["gold"] for r in by_act_a[act]], [r["pred"] for r in by_act_a[act]]
        gb, pb = [r["gold"] for r in by_act_b[act]], [r["pred"] for r in by_act_b[act]]
        fa_act = f1_score(ga, pa, mode="strict", scheme=IOB2)
        fb_act = f1_score(gb, pb, mode="strict", scheme=IOB2)
        per_act[act] = {"f1_a": fa_act, "f1_b": fb_act, "delta": fa_act - fb_act}
        print(f"{act:42s} {fa_act:7.4f} {fb_act:7.4f} {fa_act - fb_act:+7.4f}")

    stem_a = Path(path_a).stem
    stem_b = Path(path_b).stem
    out_path = RESULTS_DIR / f"compare_{stem_a}_vs_{stem_b}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_a": tag_a,
                "model_b": tag_b,
                "f1_a": fa,
                "f1_b": fb,
                "delta_f1": fa - fb,
                "bootstrap": {
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "h1_sustained": bool(lo > 0),
                },
                "per_act": per_act,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _log(f"Resultado salvo em: {out_path}")
    _log("Comparação concluída.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Caminho do modelo ou adaptador PEFT")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                        help="Precisão do modelo (bf16 padrão; fp32 p/ diagnóstico de NaN)")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Nº de demonstrações few-shot do treino (0 = zero-shot)")
    parser.add_argument("--output", default=None, help="Arquivo JSON de saída")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compara dois JSONs de resultado (ΔF1 + bootstrap)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.model is None:
        parser.error("--model é obrigatório para avaliação")

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)

    records: list[dict] = []
    with open(FT_DIR / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    demo_pool = load_few_shot_demos(args.few_shot) if args.few_shot > 0 else None

    if rank == 0:
        if demo_pool:
            shot_desc = f"{args.few_shot}-shot adaptativo (por ato/entidade)"
        else:
            shot_desc = "zero-shot"
        print(f"Split '{args.split}': {len(records):,} exemplos ({world_size} GPU(s)) | {shot_desc}")
        print("Gerando predições…")

    # Cada rank processa 1/world_size do dataset de forma intercalada
    records_slice = records[rank::world_size]

    # Garante sufixo _fmt no nome do arquivo de saída, se ausente.
    model_tag = Path(args.model).name
    if args.few_shot > 0:
        model_tag += f"_fs{args.few_shot}"
    if "_fmt" not in model_tag:
        model_tag += "_fmt"
    out_path = args.output or str(RESULTS_DIR / f"ner_{model_tag}_{args.split}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    predictions = generate_predictions(
        model, tokenizer, records_slice, args.max_new_tokens, args.batch_size, rank,
        demo_pool,
    )

    # Salva resultados parciais por rank
    tmp_path = out_path + f".rank{rank}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    # Barreira: aguarda todos os ranks terminarem
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    # Rank 0 agrega na ordem original e calcula métricas
    if rank == 0:
        rank_preds = []
        for r in range(world_size):
            tmp = out_path + f".rank{r}"
            with open(tmp, encoding="utf-8") as f:
                rank_preds.append(json.load(f))
            os.remove(tmp)

        # Reconstrói ordem original: registro i foi para rank (i % world_size)
        all_predictions = []
        iters = [iter(rp) for rp in rank_preds]
        for i in range(len(records)):
            all_predictions.append(next(iters[i % world_size]))

        metrics = overall_metrics(all_predictions)
        per_act = per_act_metrics(all_predictions)
        per_entity = per_entity_metrics(all_predictions)
        boot = bootstrap_f1(all_predictions)

        print(f"\nF1 (default) : {metrics['f1']:.4f}")
        print(f"F1 (strict)  : {metrics['strict']['f1']:.4f}")
        print(f"Precision    : {metrics['precision']:.4f} "
              f"(strict {metrics['strict']['precision']:.4f})")
        print(f"Recall       : {metrics['recall']:.4f} "
              f"(strict {metrics['strict']['recall']:.4f})")
        print(f"IC 95% (F1)  : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        print(f"\nPor ato:\n{'Ato':42s} {'F1':>7} {'F1str':>7} {'P':>7} {'R':>7} {'N':>6}")
        print("-" * 80)
        for act, m in per_act.items():
            print(
                f"{act:42s} {m['f1']:7.4f} {m['f1_strict']:7.4f} "
                f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_records']:6d}"
            )

        print(f"\nPor tipo de entidade:\n{'Ato/Entidade':42s} {'F1':>7} {'F1str':>7} "
              f"{'P':>7} {'R':>7} {'N':>6}")
        print("-" * 80)
        for key, m in per_entity.items():
            print(
                f"{key:42s} {m['f1']:7.4f} {m['f1_strict']:7.4f} "
                f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_records']:6d}"
            )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "split": args.split,
                    "metrics": metrics,
                    "per_act": per_act,
                    "per_entity": per_entity,
                    "bootstrap": boot,
                    "predictions": all_predictions,
                },
                f,
                ensure_ascii=False,
                indent=2,
                cls=_NumpyEncoder,
            )
        print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
