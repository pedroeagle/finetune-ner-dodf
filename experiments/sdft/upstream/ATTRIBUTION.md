# Vendored code — SDFT (On-Policy Self-Distillation)

The SDFT trainer used by `train_sdft.py` (`distil_trainer.py` and
`distil_config.py`) comes from the paper's official repository:

- Paper: "Self-Distillation Enables Continual Learning" — arXiv:2601.19897
- Repository: https://github.com/idanshen/Self-Distillation
  (mirror: https://github.com/Continual-Intelligence/Self-Distillation)
- Pinned commit: `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e`

These two files are **not redistributed here**: the source repository ships no
license (technically "all rights reserved"). Instead, run

```bash
bash experiments/sdft/upstream/fetch_upstream.sh
```

which clones the pinned commit, copies the two files into this directory, and
applies `beta_anchor.patch`. Both are git-ignored so they are never committed.

## Method (summary, confirmed in the code)

- Teacher = frozen base model **conditioned on the gold answer in context**
  (`teacher_prompt`); student = same model seeing only the question (`prompt`).
- **On-policy** sampling (generate from the student; `generate_from_teacher=False`).
- Loss = per-token forward KL, student→teacher (`alpha=0`, GKD-style).
- `sync_ref_model=True` with a slow EMA (`ref_model_mixup_alpha`) makes the
  teacher track the student (target network).

## Local modification (`beta_anchor.patch`)

`distil_config.py` is used unchanged. `beta_anchor.patch` makes the **single**
change to `distil_trainer.py` needed for the dissertation's KL-anchor experiment
(`--beta > 0` in `train_sdft.py`):

1. Reference forward: when `self.anchor_to_base` is `True` (set by
   `train_sdft.py`), the beta anchor's reference is the **frozen base**
   (`disable_adapter()` on the student), not `self.ref_model` — which, with the
   EMA teacher on, IS the moving teacher that collapses. Default `False` keeps
   upstream behavior.
2. Loss: `per_token_loss += self.beta * per_token_kl`. Upstream computed
   `per_token_kl` only for logging (`kl_to_base_model`) and never added it to the
   loss, so the term now actually regularizes the student.

Both hunks are inert with `beta=0.0` (the default), so SDFT without the anchor
reproduces upstream exactly.

## License

The source repository includes no license file. Use here is for academic
reproduction with citation to the paper. Confirm permission with the authors
before redistributing the upstream files themselves.
