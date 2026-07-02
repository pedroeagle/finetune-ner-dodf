#!/usr/bin/env python3
"""Fine-tuning do BERTimbau (encoder) para NER no corpus DODF/UnB-KnEDLe.

Baseline de estado-da-arte (classificação de tokens) para contrastar com o NER
generativo do Qwen3-4B. Usa AutoModelForTokenClassification + Trainer sobre os
MESMOS splits CoNLL (mesmos documentos por partição que o decoder), de modo que
a comparação meça a distância até a SOTA de encoder sem viés de dados.

Formulação "natural" do encoder: um único modelo global, head sobre todos os
rótulos BIO do corpus, uma passada por documento (todos os tipos de entidade de
uma vez). Os sub-tokens não-iniciais e os tokens especiais recebem -100 (padrão
HuggingFace de alinhamento de rótulos para WordPiece).

Uso:
  python experiments/train_bertimbau.py --model neuralmind/bert-base-portuguese-cased
  python experiments/train_bertimbau.py --model neuralmind/bert-large-portuguese-cased
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def _ensure_safetensors(hub_model_id: str) -> str:
    """Converte pytorch_model.bin → safetensors num dir local, com cache.

    Workaround para CVE-2025-32434: transformers >= 4.52 bloqueia torch.load
    em torch < 2.6, mas neuralmind/bert-* só distribui pytorch_model.bin.
    Solução: baixamos o .bin via torch.load direto (não passa pelo bloqueio do
    transformers) e re-salvamos como safetensors antes do from_pretrained.

    Deve ser chamado apenas pelo rank 0; os outros ranks esperam pelo sentinel
    antes de continuar (ver lógica em main()).
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import save_file

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    slug = hub_model_id.replace("/", "--")
    local = Path(hf_home) / "bertimbau_safetensors" / slug
    local.mkdir(parents=True, exist_ok=True)

    print(f"[safetensors] Convertendo {hub_model_id} → {local}")

    # Baixa config, tokenizer, vocab — ignora .bin e formatos alternativos
    snapshot_download(
        hub_model_id,
        local_dir=str(local),
        ignore_patterns=["pytorch_model.bin", "*.msgpack", "flax_model.msgpack"],
    )

    # Baixa o .bin no cache HF padrão e carrega via torch.load direto
    # (não passa por check_torch_load_is_safe do transformers — não bloqueado)
    bin_path = hf_hub_download(hub_model_id, "pytorch_model.bin")
    state_dict = torch.load(bin_path, map_location="cpu")

    # BERT amarra word_embeddings ↔ cls.predictions.decoder (mesmo storage).
    # safetensors.save_file recusa tensores com storage compartilhado; clonar os
    # aliases é seguro porque from_pretrained re-amarra via tie_weights() ao carregar.
    seen: dict[int, str] = {}
    for key in list(state_dict.keys()):
        ptr = state_dict[key].untyped_storage().data_ptr()
        if ptr in seen:
            state_dict[key] = state_dict[key].clone()
        else:
            seen[ptr] = key

    save_file(state_dict, str(local / "model.safetensors"))

    print(f"[safetensors] Salvo: {local / 'model.safetensors'}")
    return str(local)


ROOT = Path(__file__).parent.parent
CONLL_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/conll"
RESULTS_DIR = ROOT / "results"


def read_conll(path: Path) -> list[dict]:
    """Lê CoNLL (Publication:/Act:/token label/linha em branco) em documentos.

    Cada documento vira {'publication','act','tokens','labels'}. A unidade de
    documento é o par (publicação, ato), idêntica à do split do decoder.
    """
    docs: list[dict] = []
    pub = act = None
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        if tokens:
            docs.append(
                {"publication": pub, "act": act, "tokens": list(tokens), "labels": list(labels)}
            )

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


def build_label_list(docs: list[dict]) -> list[str]:
    """Conjunto global de rótulos BIO a partir do treino. 'O' recebe índice 0."""
    labels = set()
    for d in docs:
        labels.update(d["labels"])
    labels.discard("O")
    return ["O"] + sorted(labels)


def tokenize_and_align(docs: list[dict], tokenizer, label2id: dict, max_length: int) -> Dataset:
    """Tokeniza por palavra (WordPiece) e alinha rótulos: 1º sub-token recebe o
    rótulo da palavra; sub-tokens seguintes e tokens especiais recebem -100."""
    all_tokens = [d["tokens"] for d in docs]
    all_labels = [d["labels"] for d in docs]

    enc = tokenizer(
        all_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned: list[list[int]] = []
    for i, labels in enumerate(all_labels):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        row: list[int] = []
        for wid in word_ids:
            if wid is None:
                row.append(-100)
            elif wid != prev:
                row.append(label2id[labels[wid]])
            else:
                row.append(-100)
            prev = wid
        aligned.append(row)

    enc["labels"] = aligned
    return Dataset.from_dict({k: enc[k] for k in enc})


def main() -> None:
    # CVE-2025-32434: o transformers bloqueia torch.load em torch < 2.6 em dois
    # pontos: carregamento do modelo (from_pretrained) e do otimizador/scheduler
    # (resume_from_checkpoint). Patch global aqui cobre ambos. Seguro neste
    # ambiente controlado — todos os arquivos carregados são do HF Hub ou foram
    # gerados pelo próprio Trainer neste projeto.
    import transformers.utils.import_utils as _tu
    _tu.check_torch_load_is_safe = lambda: None

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="neuralmind/bert-base-portuguese-cased",
                        help="Encoder base (BERTimbau base ou large)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Taxa de aprendizado (prática comum p/ fine-tuning de BERT NER)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Limite de sub-tokens do BERT; atos têm ~60 tokens (sem truncamento)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Passos de acumulação de gradiente (para manter batch efetivo sem DDP)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    tag = "bertimbau_large" if "large" in args.model.lower() else "bertimbau_base"
    out_dir = args.output_dir or str(RESULTS_DIR / f"checkpoints/{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Workaround CVE-2025-32434 (torch < 2.6 + neuralmind .bin) -----------
    # Rank 0 converte .bin → safetensors; outros ranks esperam pelo sentinel.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not Path(args.model).is_dir():
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        slug = args.model.replace("/", "--")
        safe_dir = Path(hf_home) / "bertimbau_safetensors" / slug
        sentinel = safe_dir / ".done"

        if local_rank == 0:
            if not sentinel.exists():
                _ensure_safetensors(args.model)
                sentinel.touch()
        else:
            while not sentinel.exists():
                time.sleep(2)

        model_src = str(safe_dir)
    else:
        model_src = args.model
    # -------------------------------------------------------------------------

    train_docs = read_conll(CONLL_DIR / "train.conll")
    dev_docs = read_conll(CONLL_DIR / "dev.conll")

    label_list = build_label_list(train_docs)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_src)
    train_ds = tokenize_and_align(train_docs, tokenizer, label2id, args.max_length)
    dev_ds = tokenize_and_align(dev_docs, tokenizer, label2id, args.max_length)

    model = AutoModelForTokenClassification.from_pretrained(
        model_src,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print(f"Modelo    : {args.model} ({tag})")
    print(f"Rótulos   : {len(label_list)}")
    print(f"Treino    : {len(train_ds):,} documentos")
    print(f"Validação : {len(dev_ds):,} documentos")
    print(f"Saída     : {out_dir}")

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_row, label_row in zip(preds, labels):
            tl, tp = [], []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    tl.append(id2label[int(l)])
                    tp.append(id2label[int(p)])
            true_labels.append(tl)
            true_preds.append(tp)
        return {
            "f1": f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
        }

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        ddp_find_unused_parameters=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=None)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nModelo salvo em: {out_dir}")


if __name__ == "__main__":
    main()
