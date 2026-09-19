# Generative NER on DODF acts — replication code

Code for the experiments specializing a decoder LLM (**Qwen3-4B**) in **named
entity recognition (NER)** over acts of the Diário Oficial do DF (the UnB-KnEDLe /
DODF-P corpus), comparing fine-tuning strategies along two axes:

- **H1 — task performance:** strict F1 (IOB2) on NER.
- **H2 — catastrophic forgetting:** drop (Δpp) on general-ability benchmarks.

Strategies evaluated: **SFT with LoRA and DoRA** (with a rank sweep), **SDFT**
(on-policy self-distillation, with a fixed teacher and with a KL anchor β), and an
**encoder baseline (BERTimbau base/large)**. The repository also includes a data
curve (LoRA on fractions of the training set), reported as a complementary experiment.

The corpus and prompts are Portuguese; source comments and documentation are English.

## Layout

```
scripts/
  build_dataset.py      Builds the splits (70/15/15 by document) and the GNER JSONL
  setup_cluster.sh      Creates the finetune-venv on the cluster (evaluation + SFT)
experiments/
  train.py              SFT LoRA/DoRA (TRL SFTTrainer + PEFT)
  train_sdft.py         On-policy SDFT (fixed or EMA teacher; KL anchor --beta)
  train_bertimbau.py    Encoder baseline (token classification)
  evaluate_ner.py       H1: distributed generation + seqeval; comparison with bootstrap
  evaluate_forgetting.py H2: lm-eval-harness on the benchmarks (core/ext/gsm8k/scripted)
  evaluate_bertimbau.py Encoder NER evaluation (same records as the decoder)
  select_checkpoint.sh  SDFT checkpoint selection by dev-F1 + H1/H2 on the winner
  submit.sh             SLURM orchestration (one step per command)
  sdft/                 SDFT dataset (data.py) + official vendored trainer (upstream/)
  requirements*.txt     Dependencies (evaluation vs SDFT, separate venvs)
  # Result re-analysis (run no model; use --results to point at the results dir):
  bootstrap_ci.py       ΔF1 CIs by document resampling (the ΔF1 table)
  aggregate_h2.py       Mean±std Δpp across seeds on the benchmarks (the H2 table)
  per_act_error.py      F1 and omission per act type
  per_entity_gap.py     Encoder–decoder gap per entity group (template/free-form)
```

The extra venvs are separate: **sdft-venv** (trl 0.24, for SDFT training — see
`experiments/requirements-sdft.txt`) and **eval-venv-ds2** (`datasets<3.0`, required
by the `bench-scripted` step that produces MathQA and CrowS-Pairs).

## Prerequisites

- Python 3.10+, a GPU (the code was run on AMD ROCm MI250; `attn_implementation="eager"`
  and bf16 avoid NaNs on that platform).
- The corpus at `datasets/lre-dodfpcorpus-main/corpus/*.conll` (not included; see the paper).
- Dependencies: `pip install -r experiments/requirements.txt` (evaluation/SFT) and, in a
  separate venv, `pip install -r experiments/requirements-sdft.txt` (SDFT, trl 0.24).
- **For SDFT**, the vendored trainer is not shipped here — fetch it first:
  `bash experiments/sdft/upstream/fetch_upstream.sh` (clones the pinned upstream commit
  and applies the β-anchor patch; see `experiments/sdft/upstream/ATTRIBUTION.md`).

## Replication pipeline

```bash
# 1. Dataset: CoNLL splits + GNER JSONL (prompt/instruction/BIO output)
python scripts/build_dataset.py

# 2. Fine-tuning (locally or via submit.sh on SLURM)
python experiments/train.py --adapter lora --rank 8      # SFT LoRA (r∈{4,8,16})
python experiments/train.py --adapter dora --rank 8      # SFT DoRA

# 3. H1: NER on the test set (PEFT adapter detected automatically)
torchrun --nproc_per_node=8 experiments/evaluate_ner.py \
    --model results/checkpoints/lora_r8_fmt --split test
python experiments/evaluate_ner.py --compare \
    results/ner_lora_r8_fmt_test.json results/ner_Qwen3-4B_fmt_test.json

# 4. H2: general benchmarks (base and fine-tuned) and the Δpp comparison
python experiments/evaluate_forgetting.py --model Qwen/Qwen3-4B --tag base
python experiments/evaluate_forgetting.py --model results/checkpoints/lora_r8_fmt --tag lora_r8
python experiments/evaluate_forgetting.py --compare \
    results/bench_base.json results/bench_lora_r8.json

# 5. SDFT (see experiments/sdft/README.md for the method and the two collapse fixes)
BETA=0.3 ./experiments/submit.sh train-sdft-anchor mi250        # KL anchor (dissertation)
./experiments/submit.sh train-sdft-full-budget mi250            # fixed teacher (paper)
BASE=$WORK/checkpoints/sdft_r8_anchor_b03_s42 ./experiments/select_checkpoint.sh
```

On the cluster, each step above has a shortcut in `submit.sh` (e.g.,
`./experiments/submit.sh train-lora mi250 12`). Run `./experiments/submit.sh` with no
arguments to list the steps.

## Reproducibility notes

- Seed 42 throughout the pipeline; seeds 43/44 estimate the variance of H2.
- **Effective batch:** SFT (`train.py`) uses `batch-size × grad-accum × world_size`.
  In the reported runs (8 GPUs) that is **128** for LoRA/DoRA (4×4×8). In the SDFT steps
  `submit.sh` derives `grad-accum` from the GPU count to fix the effective batch at
  **32** — the two regimes differ, which matters when comparing budgets.
- **Checkpoint selection (best dev F1):** for SDFT, use `select_checkpoint.sh`. For
  LoRA/DoRA, evaluate each epoch on dev by hand, e.g.,
  `torchrun ... evaluate_ner.py --model results/checkpoints/lora_r8_fmt/checkpoint-<N> --split dev`,
  and report on test the checkpoint with the highest dev F1.
- The re-analysis scripts (`bootstrap_ci.py`, `aggregate_h2.py`, `per_act_error.py`,
  `per_entity_gap.py`) read the JSONs from `results/` (which is git-ignored); pass
  `--results <dir>` if they are not under `../results`.
- Trainings that exceed the 12h walltime are **chained** with `--dep <jobid>`
  (auto-resume from the last complete checkpoint).
- The SDFT trainer in `sdft/upstream/` is vendored from the paper's official repository;
  see `sdft/upstream/ATTRIBUTION.md` (source, commit, and the only local change: the β anchor).
