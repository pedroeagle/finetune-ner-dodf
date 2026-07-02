#!/usr/bin/env python3
"""Fine-tuning Qwen3-4B no corpus DODF/UnB-KnEDLe com LoRA ou DoRA.

Usa TRL SFTTrainer + PEFT LoraConfig. QLoRA (NF4) ativado com --use-qlora.
O adaptador é salvo no formato PEFT (adapter_config.json + adapter_model.safetensors).

Uso:
  python experiments/train.py --adapter lora --rank 8
  python experiments/train.py --adapter dora --rank 8
  python experiments/train.py --adapter lora --rank 8 --use-qlora
  python experiments/train.py --adapter lora --rank 4   # varredura de rank (paper: r in {4,8,16})
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"
RESULTS_DIR = ROOT / "results"

CHATML_TMPL = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_as_chat(record: dict, tokenizer) -> str:
    # `instruction_fmt` = enunciado + descrição do formato de saída, gravado no
    # dataset por build_dataset.py — o mesmo prompt usado na avaliação. Índice
    # direto (não .get): falha alto se o campo estiver ausente, evitando
    # treinar silenciosamente com prompt incorreto.
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


def build_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    return Dataset.from_list(
        [{"text": format_as_chat(r, tokenizer)} for r in records]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Modelo base (HuggingFace hub ou caminho local)")
    parser.add_argument("--adapter", choices=["lora", "dora"], default="lora",
                        help="Técnica de adaptação: LoRA ou DoRA (seção 4.4 do paper)")
    parser.add_argument("--rank", type=int, default=8, choices=[4, 8, 16],
                        help="Posto r — paper faz varredura em {4, 8, 16}")
    parser.add_argument("--use-qlora", action="store_true",
                        help="Quantizar modelo base em NF4 para reduzir VRAM (QLoRA)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch por GPU; batch efetivo = batch-size × grad-accum")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Acúmulo de gradiente (batch efetivo padrão = 4×4 = 16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Taxa de aprendizado (paper: 2e-4, agendador linear)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Comprimento máximo do contexto em tokens (input+output BIO)")
    parser.add_argument("--output-dir", default=None,
                        help="Diretório de saída do adaptador (padrão: results/checkpoints/…)")
    args = parser.parse_args()

    # α/r = 2, conforme paper (seção 4.4) e recomendação do artigo original do LoRA
    alpha = args.rank * 2

    qlora_suffix = "_qlora" if args.use_qlora else ""
    out_dir = args.output_dir or str(
        RESULTS_DIR / f"checkpoints/{args.adapter}_r{args.rank}_fmt{qlora_suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- quantização NF4 (QLoRA) ----
    bnb_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # ---- tokenizador ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    # ---- modelo base ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    model_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if args.use_qlora:
        model_kwargs["quantization_config"] = bnb_config
    else:
        # bf16 (não fp16): fp16 no ROCm MI250 gera NaN silencioso nos pesos do
        # adapter. Deve casar com bf16=True no SFTConfig e com o dtype do eval.
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # obrigatório com gradient checkpointing

    # ---- LoRA / DoRA ----
    # target_modules=["q_proj","v_proj"] segue artigo original do LoRA (Hu et al. 2022)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        use_dora=(args.adapter == "dora"),
        bias="none",
    )

    # ---- datasets ----
    train_records = load_jsonl(FT_DIR / "train.jsonl")
    dev_records = load_jsonl(FT_DIR / "dev.jsonl")

    train_ds = build_hf_dataset(train_records, tokenizer)
    dev_ds = build_hf_dataset(dev_records, tokenizer)

    print(f"Adaptador : {args.adapter.upper()} | r={args.rank} | α={alpha}")
    print(f"QLoRA     : {'sim (NF4)' if args.use_qlora else 'não'}")
    print(f"Treino    : {len(train_ds):,} exemplos")
    print(f"Validação : {len(dev_ds):,} exemplos")
    print(f"Saída     : {out_dir}")

    # ---- SFTTrainer ----
    sft_config = SFTConfig(
        output_dir=out_dir,
        # Trunca a sequência (input+output BIO) no valor pedido. Sem isto o
        # SFTConfig usa o default 1024, ignorando --max-length e cortando ~4%
        # dos exemplos (perde o fim do turno do assistente). model_max_length
        # do tokenizer só controla o aviso, não o truncamento efetivo.
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        optim="adamw_torch",
        bf16=True,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataset_text_field="text",
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    checkpoints = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(checkpoints[-1]) if checkpoints else None
    trainer.train(resume_from_checkpoint=resume)

    # Salva adaptador PEFT (adapter_config.json + pesos); base_model_name_or_path
    # fica registrado no adapter_config.json para carregamento automático depois.
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nAdaptador salvo em: {out_dir}")


if __name__ == "__main__":
    main()
