#!/usr/bin/env python3
"""On-Policy Self-Distillation Fine-Tuning (SDFT) on the DODF/NER corpus.

Applies the method of "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)
reusing the official trainer vendored in experiments/sdft/upstream/ (DistilTrainer);
this is the analogue of the upstream main.py adapted to our corpus. See sdft/README.md.

Method (faithful to the paper):
  - Teacher conditioned on the gold answer in context (teacher_prompt); the student
    sees only the question and learns to produce it via per-token forward KL (alpha=0, GKD).
  - On-policy (generate_from_teacher=False); cosine schedule, warmup 0.1.
  - EMA teacher (sync_ref_model): a PeftModel with the SAME LoRA config as the student,
    initialized at delta zero (= base). The teacher adapter is checkpointed
    (teacher_adapter.safetensors) and restored on resume, preserving the EMA lag
    across chained jobs. --no-ema makes the teacher fixed (= base + gold).

Adaptations vs the official main.py (Qwen3-4B + LoRA, ROCm):
  - use_vllm=False (generation via transformers) for robustness on ROCm.
  - lr 1e-4 (LoRA regime; the 2e-5 in main.py are for full fine-tuning).
  - max_prompt_length 1280 (covers the largest teacher_prompt measured, 1225 tok).

Usage:
  python experiments/train_sdft.py                    # EMA, lr 1e-4, 2 epochs
  python experiments/train_sdft.py --beta 0.3         # + KL anchor to the base
  python experiments/train_sdft.py --no-ema           # fixed teacher (base+gold)
  python experiments/train_sdft.py --max-train 2000   # cheap pilot
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
UPSTREAM = Path(__file__).resolve().parent / "sdft" / "upstream"
sys.path.insert(0, str(UPSTREAM))            # import the vendored official code
sys.path.insert(0, str(Path(__file__).resolve().parent / "sdft"))

try:
    from distil_config import DistilConfig      # noqa: E402  (vendored)
    from distil_trainer import DistilTrainer     # noqa: E402  (vendored)
except ImportError as e:
    raise SystemExit(
        f"{e}\n\nThe SDFT trainer is not distributed in this repo. Fetch it first:\n"
        "  bash experiments/sdft/upstream/fetch_upstream.sh\n"
        "(see experiments/sdft/upstream/ATTRIBUTION.md)"
    )
from data import build_sdft_dataset            # noqa: E402

# Resume under torch<2.6 (ROCm): transformers 4.57 requires torch>=2.6 for the
# torch.load of optimizer.pt (CVE-2025-32434). The checkpoint is our own (trusted
# payload), so we lift the guard to let resume work.
import transformers.trainer as _hf_trainer          # noqa: E402
import transformers.utils.import_utils as _hf_iu     # noqa: E402


def _allow_local_torch_load(*_a, **_k):
    return None


_hf_trainer.check_torch_load_is_safe = _allow_local_torch_load
_hf_iu.check_torch_load_is_safe = _allow_local_torch_load


# DDP static_graph=True: with gradient checkpointing under DDP the reducer marks the
# same parameter ready twice and aborts. The SDFT graph is static across steps, so
# static_graph fixes it; since transformers 4.57 does not expose it, we inject it here.
# Inert on 1 GPU.
from accelerate.utils import DistributedDataParallelKwargs as _DDPKwargs  # noqa: E402


def _ddp_kwargs_static_graph(*a, **k):
    k.setdefault("static_graph", True)
    return _DDPKwargs(*a, **k)


_hf_trainer.DistributedDataParallelKwargs = _ddp_kwargs_static_graph


class _SaveTeacherAdapterCallback(TrainerCallback):
    """Persist the teacher (EMA) adapter alongside each student checkpoint.

    The EMA is not part of the Trainer state; without this, resume would reinitialize
    the teacher from the student adapter, losing the moving-average lag across chained
    jobs. Only rank 0 writes (the EMA is identical on all ranks)."""

    def __init__(self, teacher_model):
        self.teacher_model = teacher_model

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        # Fault-tolerant: an error here must not bring down a long chain; resume falls
        # back to the student adapter (degrades, does not break).
        try:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            sd = {k: v.detach().cpu()
                  for k, v in get_peft_model_state_dict(self.teacher_model).items()}
            save_file(sd, str(ckpt_dir / "teacher_adapter.safetensors"))
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: failed to save teacher_adapter at {state.global_step}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--rank", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (LoRA regime; SFT uses 2e-4)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="per_device_train_batch_size; batches generation (uses spare VRAM)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Accumulation; effective batch = batch-size×grad-accum (default 4×8=32)")
    parser.add_argument("--max-prompt-length", type=int, default=1280,
                        help="Prompt cap (covers the largest teacher_prompt measured, 1225 tok)")
    parser.add_argument("--max-completion-length", type=int, default=640,
                        help="On-policy generation cap (gold p99≈440 subtokens)")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Subsample the training set for pilots (None = full 87k)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: checkpoints/sdft_r{rank}_v2_fmt")
    parser.add_argument("--no-ema", action="store_true",
                        help="Turn off the EMA: teacher = frozen base (seen with the gold)")
    parser.add_argument("--mask-truncated", action="store_true",
                        help="Drop truncated completions (no EOS) from the loss. Without this, "
                             "training on-policy on the model's own never-ending generations feeds "
                             "back a loop (clipped_ratio 0->1) that collapses F1 over the long horizon.")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Weight of the KL anchor to the frozen BASE, added to the loss: "
                             "loss = KL(student||teacher) + beta*KL(student||base). 0 = plain SDFT. "
                             ">0 (the dissertation experiment) tests whether the anchor stabilizes "
                             "the EMA teacher over the long horizon. The reference is always the base "
                             "(disable_adapter), never the EMA — see trainer.anchor_to_base below.")
    parser.add_argument("--wandb", action="store_true", help="Report to Weights & Biases")
    parser.add_argument("--debug-completions", action="store_true",
                        help="Print sample completions to the log (checks <think>/truncation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (on-policy sampling + shuffle). Default 42 = original "
                             "run. Use other seeds to estimate variance (H2).")
    args = parser.parse_args()

    alpha_lora = args.rank * 2  # α/r = 2, same as train.py (SFT)
    seed_suffix = "" if args.seed == 42 else f"_s{args.seed}"
    out_dir = args.output_dir or str(RESULTS_DIR / f"checkpoints/sdft_r{args.rank}_v2_fmt{seed_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    # DDP (torchrun): pin this rank's local GPU BEFORE loading the model, otherwise
    # all ranks pile onto cuda:0. Single-GPU: LOCAL_RANK absent → set(0).
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- student (trains) and teacher (frozen base, as in main.py) ----
    model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                        attn_implementation="eager")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # ---- LoRA (identical to the project's SFT baseline) ----
    def make_lora_config() -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=alpha_lora,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

    lora_config = make_lora_config()

    # Teacher with the SAME PEFT structure as the student: the EMA sync zips parameters()
    # positionally, so the module trees must match. lora_B starts at zero → delta zero →
    # the teacher begins exactly as the base.
    if not args.no_ema:
        teacher_model = get_peft_model(teacher_model, make_lora_config())
        teacher_model.requires_grad_(False)  # only the EMA writes to it (via .data)

    # ---- dataset (pre-rendered strings, enable_thinking=False) ----
    train_ds = build_sdft_dataset("train", tokenizer=tokenizer, max_examples=args.max_train)
    print(f"SDFT | model={args.model} | LoRA r={args.rank} α={alpha_lora}")
    print(f"Train: {len(train_ds):,} examples | out={out_dir}")

    # ---- config faithful to the paper (main.py + DistilConfig defaults) ----
    config = DistilConfig(
        output_dir=out_dir,
        seed=args.seed,
        # generation
        use_vllm=False,                       # deviation: robustness on ROCm
        vllm_importance_sampling_correction=False,  # inert without vLLM
        generate_from_teacher=False,          # on-policy: sample from the student
        temperature=1.0,                      # faithful to the paper (Qwen3 forces 0.6 if unset)
        top_p=1.0,
        num_generations=1,
        num_iterations=1,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # distillation loss
        alpha=0.0,                            # forward KL
        beta=args.beta,                       # KL anchor to the base (0 = plain SDFT; see --beta)
        num_loss_tokens_to_skip=3,
        mask_truncated_completions=args.mask_truncated,
        # EMA teacher (faithful to main.py: mixup 0.01, sync every step): the teacher
        # slowly tracks the student. --no-ema keeps it fixed at the base.
        sync_ref_model=not args.no_ema,
        ref_model_mixup_alpha=0.01,
        ref_model_sync_steps=1,
        # optimization (main.py; effective batch kept at 32)
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1,
        # Non-reentrant gradient checkpointing: the reentrant mode reuses params across
        # multiple backwards and breaks DDP. Does not affect the math.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_steps=50,  # frequent save: auto-resume loses little when the job is killed
        log_completions=args.debug_completions,
        num_completions_to_print=4 if args.debug_completions else 0,
        report_to="wandb" if args.wandb else "none",
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # The KL anchor (--beta>0) pulls the student toward the frozen BASE, not the EMA. In
    # upstream the beta term uses self.ref_model (= the EMA, which collapses too); anchor_to_base
    # forces the reference via disable_adapter() on the student = base. Inert when beta=0.
    trainer.anchor_to_base = True

    # Resume from the highest-step COMPLETE checkpoint. A job killed mid-save leaves a
    # partial checkpoint (no trainer_state.json, written last); filtering on it avoids
    # resuming from a corrupt checkpoint and taking down the chain.
    def _complete(ckpt: Path) -> bool:
        return (ckpt / "trainer_state.json").exists()

    checkpoints = [c for c in sorted(Path(out_dir).glob("checkpoint-*"),
                                     key=lambda p: int(p.name.split("-")[-1]))
                   if _complete(c)]
    resume = str(checkpoints[-1]) if checkpoints else None

    if not args.no_ema:
        # Save the teacher (EMA) adapter at each checkpoint (see callback).
        trainer.add_callback(_SaveTeacherAdapterCallback(teacher_model))

        # Align the EMA starting point. Fresh run: copy the student adapter (delta zero)
        # so the EMA starts from identical state. Resume: restore the EMA from the teacher's
        # own teacher_adapter.safetensors, preserving the lag.
        if resume:
            teacher_file = Path(resume) / "teacher_adapter.safetensors"
            adapter_file = Path(resume) / "adapter_model.safetensors"
            if teacher_file.exists():
                set_peft_model_state_dict(teacher_model, load_file(str(teacher_file)))
                print(f"Teacher (EMA) restored from checkpoint: {teacher_file}")
            elif adapter_file.exists():
                set_peft_model_state_dict(teacher_model, load_file(str(adapter_file)))
                print(f"WARNING: no teacher_adapter in {resume}; teacher reinitialized from "
                      "the student adapter (old checkpoint; EMA reconverges)")
            else:
                print(f"WARNING: {adapter_file} does not exist; teacher restarts from the base "
                      "(the EMA reconverges in ~300 steps)")
        else:
            set_peft_model_state_dict(
                teacher_model, get_peft_model_state_dict(trainer.model)
            )

    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(out_dir)               # save the PEFT adapter
    tokenizer.save_pretrained(out_dir)
    print(f"\nSDFT adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
