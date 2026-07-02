#!/usr/bin/env python3
"""Diagnóstico rápido (1 GPU, sem torchrun) para o NaN do eval-ner fine-tuned.

Confirma, em poucos minutos, a causa do `token_ids = [0,0,0,...]`:
  - versão do transformers (relevante p/ o kwarg dtype= vs torch_dtype=);
  - dtype real dos pesos do base e do adapter LoRA;
  - presença de NaN/Inf nos pesos do adapter;
  - se o forward/generate produz logits NaN em bf16 — e se fp32 resolve;
  - se o formato do prompt de inferência casa com o de treino.

Uso:
  python experiments/diagnose_adapter.py --model results/checkpoints/lora_r8_fmt
  python experiments/diagnose_adapter.py --model results/checkpoints/lora_r8_fmt --dtype fp32
"""
import argparse
import json
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.disable()

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"

CHATML_TMPL = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)


def first_record(split: str = "test") -> dict:
    with open(FT_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        return json.loads(f.readline())


def training_text(record: dict, tokenizer) -> str:
    """Replica train.py:format_as_chat — o formato EXATO visto no treino."""
    instruction = record["instruction_fmt"]
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{record['input']}"},
        {"role": "assistant", "content": record["output"]},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except Exception:
        return CHATML_TMPL.format(
            user=f"{instruction}\n\n{record['input']}",
            assistant=record["output"],
        )


def inference_prompt(record: dict, tokenizer) -> str:
    """Replica evaluate_ner.py:build_prompt — o formato visto na inferência."""
    instruction = record["instruction_fmt"]
    content = f"{instruction}\n\n{record['input']}"
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return "<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n".format(c=content)


def tensor_health(t: torch.Tensor) -> str:
    return (f"dtype={t.dtype} nan={torch.isnan(t).any().item()} "
            f"inf={torch.isinf(t).any().item()} "
            f"absmax={t.abs().max().item():.4g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Caminho do adapter PEFT")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--gen-tokens", type=int, default=40)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda:0")

    print("=" * 70)
    print(f"transformers : {transformers.__version__}")
    print(f"torch        : {torch.__version__}")
    print(f"dtype pedido : {dtype}")
    print("=" * 70)

    adapter_cfg = Path(args.model) / "adapter_config.json"
    if not adapter_cfg.exists():
        raise SystemExit(f"adapter_config.json não encontrado em {args.model}")

    with open(adapter_cfg, encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg["base_model_name_or_path"]
    print(f"Base: {base}")

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 1) base puro ----
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
    ).to(device).to(dtype)

    sample_base = next(model.parameters())
    print(f"\n[base] 1º param: {tensor_health(sample_base)}")

    # ---- 2) aplica adapter ----
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.model)
    model = model.to(dtype)
    model.eval()

    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    print(f"\n[adapter] {len(lora_params)} tensores LoRA")
    any_bad = False
    for n, p in lora_params[:6]:
        bad = torch.isnan(p).any().item() or torch.isinf(p).any().item()
        any_bad = any_bad or bad
        print(f"  {n.split('.')[-3:]} -> {tensor_health(p)}")
    # varredura completa de NaN/Inf no adapter
    full_bad = any(
        torch.isnan(p).any().item() or torch.isinf(p).any().item()
        for _, p in lora_params
    )
    print(f"  NaN/Inf em ALGUM peso do adapter? {full_bad}")

    # ---- 3) formato treino vs inferência ----
    rec = first_record(args.split)
    train_txt = training_text(rec, tokenizer)
    infer_txt = inference_prompt(rec, tokenizer)
    print("\n--- FORMATO DE TREINO (fim) ---")
    print(repr(train_txt[-160:]))
    print("--- FORMATO DE INFERÊNCIA (fim) ---")
    print(repr(infer_txt[-160:]))
    print(f"inferência é prefixo do treino? "
          f"{train_txt.startswith(infer_txt.rstrip(chr(10)).rstrip())}")

    # ---- 4) forward em 1 prompt ----
    inputs = tokenizer(infer_txt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    print("\n[forward] logits:", tensor_health(logits))
    last = logits[0, -1]
    print(f"[forward] argmax último token: {last.argmax().item()} "
          f"({tokenizer.decode([last.argmax().item()])!r})")

    # ---- 5) generate ----
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=args.gen_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = gen[0][inputs["input_ids"].shape[1]:].tolist()
    print(f"\n[generate] ids[:20]: {new_ids[:20]}")
    print(f"[generate] texto   : "
          f"{tokenizer.decode(new_ids, skip_special_tokens=True)[:200]!r}")

    if logits.isnan().any():
        print("\n>>> NaN nos logits CONFIRMADO neste dtype.")
        if dtype != torch.float32:
            print(">>> Rode de novo com --dtype fp32 para confirmar que é bf16/ROCm.")
    else:
        print("\n>>> Sem NaN. Se em bf16, o caminho de eval está saudável.")


if __name__ == "__main__":
    main()
