#!/usr/bin/env python3
"""Build the SDFT dataset (prompt + teacher_prompt) from the DODF/NER corpus.

Each example has two columns, as the DistilTrainer expects:
  - `prompt`         → what the STUDENT sees: only the instruction + the input text.
  - `teacher_prompt` → what the TEACHER sees: the same instruction + text, with the
                       gold answer (BIO) shown in context.

With a `tokenizer`, the columns are pre-rendered to strings with enable_thinking=False
(matches the SFT in train.py and avoids <think> blocks during on-policy generation).
Without a tokenizer, returns message lists (for inspection).

The prompt strings are Portuguese because the task and corpus are Portuguese; they
are dataset content, not translatable UI text.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
FT_DIR = ROOT / "datasets/lre-dodfpcorpus-main/splits/finetune"

# Mirrors the upstream Template (main.py::load_tooluse_dataset), in Brazilian Portuguese.
TEACHER_TMPL = Template(
    "$orig_content\n\n"
    "Este é um exemplo de resposta correta para a tarefa acima:\n"
    "$output_text\n\n"
    "Agora produza a sua própria resposta, no mesmo formato."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _user_content(record: dict) -> str:
    # instruction_fmt = task statement + BIO format description (build_dataset.py).
    return f"{record['instruction_fmt']}\n\n{record['input']}"


def _render(messages: list[dict], tokenizer) -> str:
    """Apply the chat template with thinking off; matches the SFT (train.py)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Tokenizers without the enable_thinking kwarg.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def format_example(record: dict, tokenizer=None) -> dict:
    orig = _user_content(record)
    prompt_msgs = [{"role": "user", "content": orig}]
    teacher_msgs = [
        {
            "role": "user",
            "content": TEACHER_TMPL.substitute(
                orig_content=orig, output_text=record["output"]
            ),
        }
    ]
    if tokenizer is None:
        return {"prompt": prompt_msgs, "teacher_prompt": teacher_msgs}
    return {
        "prompt": _render(prompt_msgs, tokenizer),
        "teacher_prompt": _render(teacher_msgs, tokenizer),
    }


def build_sdft_dataset(
    split: str = "train",
    tokenizer=None,
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load a split and return a Dataset with prompt/teacher_prompt columns.

    tokenizer: if given, pre-renders the columns to strings (recommended).
    max_examples: subsample (for cheap pilots). None = everything.
    """
    records = load_jsonl(FT_DIR / f"{split}.jsonl")
    ds = Dataset.from_list([format_example(r, tokenizer) for r in records])
    ds = ds.shuffle(seed=seed)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


if __name__ == "__main__":
    # Smoke test: print one formatted example (message list, no model).
    ds = build_sdft_dataset(max_examples=1)
    ex = ds[0]
    print("=== prompt (student) ===")
    print(ex["prompt"][0]["content"][:600])
    print("\n=== teacher_prompt (teacher, with gold in context) ===")
    print(ex["teacher_prompt"][0]["content"][:900])
