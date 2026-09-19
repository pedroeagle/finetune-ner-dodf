# SDFT experiment (On-Policy Self-Distillation) on DODF/NER

Application of the method from **"Self-Distillation Enables Continual Learning"**
(arXiv:2601.19897) to DODF NER specialization, with the goal of **degrading general
abilities less** (H2) than SFT does.

## In one sentence

The teacher is the **base model seeing the gold answer in context**; the student
learns to produce it without seeing it, via **per-token forward KL** (on-policy).
Because the target comes from the base's own distribution, the model drifts and
forgets less than SFT on hard labels.

## Layout

- `upstream/` — the official trainer (`distil_trainer.py`, `distil_config.py`), NOT
  redistributed here. Run `upstream/fetch_upstream.sh` to reconstruct it; see
  `upstream/ATTRIBUTION.md` (source, commit, license, and the only local change: the β anchor).
- `data.py` — builds `prompt` (student) and `teacher_prompt` (teacher, with the gold in
  context), reusing the same SFT statement (`enable_thinking=False`).
- `../train_sdft.py` — the analogue of the upstream `main.py`, adapted (Qwen3-4B, LoRA).
- `../requirements-sdft.txt` — a **separate** venv (trl 0.24), distinct from the eval one.

## Two fixes for EMA teacher collapse

The EMA teacher (`sync_ref_model`) tracks the student; over the long horizon, if the
student starts generating sequences that never end, the teacher drifts along and the
KL loses corrective force (F1 collapses). There are two ways to reintroduce a stable
reference to the base:

1. **Fixed teacher** (`--no-ema`): the teacher becomes the frozen base + gold. The
   target cannot degenerate and carries the task signal. This is the SDFT reported in the paper.
2. **KL anchor β** (`--beta > 0`): keep the EMA teacher and **add** to the student's
   loss a term `β·KL(student‖base)`, pulling it toward the base's general behavior
   (anti-drift). This is the dissertation experiment; the reference is always the
   frozen base (see `anchor_to_base` in `train_sdft.py` and `ATTRIBUTION.md`).

## Decisions (fidelity × comparability)

| Item | Choice | Why |
|---|---|---|
| Model | Qwen3-4B | Same anchor as the project; comparable to LoRA r=8 and the H2 metrics |
| Adaptation | LoRA r=8 | Comparable to the SFT baseline and cheap (upstream supports PEFT) |
| Generation | `use_vllm=False` | Robustness on ROCm |
| Loss | forward KL (α=0) | Identical to `main.py` |

**Scale caveat:** the paper itself shows that at ~3B SDFT loses to SFT; the gain
appears from 7B up. At 4B the outcome is uncertain — a hypothesis to test.

## How to run

Cheap pilot (validates the integration before spending budget):
```bash
python experiments/train_sdft.py --max-train 2000 --max-completion-length 1024
```

Full training (via SLURM):
```bash
# fixed teacher, budget matched to LoRA (the paper's SDFT)
./experiments/submit.sh train-sdft-full-budget mi250

# KL anchor β (dissertation); BETA required, SEED optional
BETA=0.3 ./experiments/submit.sh train-sdft-anchor mi250
```

Checkpoint selection (by dev-F1) + H1/H2 evaluation of the winner:
```bash
BASE=$WORK/checkpoints/sdft_r8_anchor_b03_s42 ./experiments/select_checkpoint.sh
```
