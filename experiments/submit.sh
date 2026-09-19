#!/bin/bash
# Submit each experiment step to SLURM (AMD ROCm cluster).
#
# Usage:
#   ./experiments/submit.sh <step> [gpu_type] [hours] [--dep <jobid>]
#
# Decoder artifacts carry the _fmt suffix (the expected output format is written into
# the dataset by scripts/build_dataset.py and used in both training and evaluation).
#
# gpu_type: mi210 | mi250 | mi300 | mi325 | mi350   (default: mi250)
# hours:    1-12                                     (default: per step; max 12h)
# --dep <jobid>: wait for <jobid> to finish (afterany) before starting — chains
#                continuations of trainings that exceed the walltime (auto-resume).
#
# Examples:
#   ./experiments/submit.sh bench-base  mi250 4
#   ./experiments/submit.sh train-lora  mi250 12
#   ./experiments/submit.sh train-lora  mi250 12 --dep 335216
#   BETA=0.3 ./experiments/submit.sh train-sdft-anchor mi250 12

set -e

STEP="${1:-}"
GPU_TYPE="${2:-mi250}"
HOURS="${3:-}"
# If the 3rd arg is a flag (e.g., --dep), it is not "hours": fall back to the step default.
if [[ "$HOURS" == --* ]]; then HOURS=""; fi

# --dep <jobid>: chain this job after another finishes (for any reason).
DEP_JOBID=""
for arg in "$@"; do
    if [[ "$arg" == "--dep" ]]; then
        _next=1
    elif [[ "${_next:-0}" == "1" ]]; then
        DEP_JOBID="$arg"
        _next=0
    fi
done

VALID_STEPS="bench-base train-lora train-dora train-r4 train-r16 eval-ner eval-ner-r4-dev eval-ner-r8-dev eval-ner-r16-dev eval-ner-base eval-ner-fewshot eval-ner-fewshot3 eval-ner-dora train-bert-base train-bert-large eval-bert-base eval-bert-large bench-ft bench-ft-dora train-lora-seed eval-ner-seed bench-ft-seed train-dora-seed eval-ner-dora-seed bench-ft-dora-seed bench-luo bench-scripted train-lora-frac eval-ner-frac train-sdft-full train-sdft-full-budget train-sdft-budget-seed train-sdft-anchor eval-ner-sdft-full bench-ft-sdft-full compare-ner compare-bert compare-bench compare-sweep"
if [[ -z "$STEP" || ! " $VALID_STEPS " =~ " $STEP " ]]; then
    echo "Usage: $0 <step> [gpu_type] [hours]"
    echo "Steps: $VALID_STEPS"
    exit 1
fi

# Comparison steps run locally (no GPU).
if [[ "$STEP" == "compare-ner" ]]; then
    python experiments/evaluate_ner.py \
        --compare results/ner_lora_r8_fmt_test.json results/ner_Qwen3-4B_fmt_test.json
    exit 0
fi

if [[ "$STEP" == "compare-bert" ]]; then
    python experiments/evaluate_ner.py \
        --compare results/ner_bertimbau_base_test.json results/ner_lora_r8_fmt_test.json
    exit 0
fi

if [[ "$STEP" == "compare-bench" ]]; then
    python experiments/evaluate_forgetting.py \
        --compare results/bench_base.json results/bench_lora_r8.json
    exit 0
fi

if [[ "$STEP" == "compare-sweep" ]]; then
    # Rank selection on dev: ΔF1 of r=4 and r=16 against r=8 (the reference).
    echo "### r=4 vs r=8 (dev) ###"
    python experiments/evaluate_ner.py \
        --compare results/ner_lora_r4_fmt_dev.json results/ner_lora_r8_fmt_dev.json
    echo ""
    echo "### r=16 vs r=8 (dev) ###"
    python experiments/evaluate_ner.py \
        --compare results/ner_lora_r16_fmt_dev.json results/ner_lora_r8_fmt_dev.json
    exit 0
fi

# gpu_type → SLURM partition mapping (adjust to your cluster's partitions).
case "$GPU_TYPE" in
    mi250) PARTITION="mi2508x"; GPUS_PER_NODE=8 ;;
    mi210) PARTITION="mi2104x"; GPUS_PER_NODE=4 ;;
    mi300) PARTITION="mi3008x"; GPUS_PER_NODE=8 ;;
    mi325) PARTITION="mi3258x"; GPUS_PER_NODE=8 ;;
    mi350) PARTITION="mi3508x"; GPUS_PER_NODE=8 ;;
    *)
        echo "Invalid gpu_type. Use: mi210, mi250, mi300, mi325 or mi350"
        exit 1
        ;;
esac

# Default walltime per step (long trainings are chained with --dep).
case "$STEP" in
    bench-base)    DEFAULT_HOURS=4  ;;
    train-lora)    DEFAULT_HOURS=12 ;;
    train-dora)    DEFAULT_HOURS=12 ;;
    train-r4)      DEFAULT_HOURS=8  ;;
    train-r16)     DEFAULT_HOURS=8  ;;
    eval-ner)      DEFAULT_HOURS=8  ;;
    eval-ner-r4-dev)  DEFAULT_HOURS=8  ;;
    eval-ner-r8-dev)  DEFAULT_HOURS=8  ;;
    eval-ner-r16-dev) DEFAULT_HOURS=8  ;;
    eval-ner-base)    DEFAULT_HOURS=4  ;;
    eval-ner-fewshot) DEFAULT_HOURS=8  ;;
    eval-ner-fewshot3) DEFAULT_HOURS=8  ;;
    eval-ner-dora) DEFAULT_HOURS=8  ;;
    train-bert-base)  DEFAULT_HOURS=4  ;;
    train-bert-large) DEFAULT_HOURS=6  ;;
    eval-bert-base)   DEFAULT_HOURS=2  ;;
    eval-bert-large)  DEFAULT_HOURS=2  ;;
    bench-ft)      DEFAULT_HOURS=4  ;;
    bench-ft-dora) DEFAULT_HOURS=4  ;;
    train-lora-seed)   DEFAULT_HOURS=12 ;;
    eval-ner-seed)     DEFAULT_HOURS=8  ;;
    bench-ft-seed)     DEFAULT_HOURS=4  ;;
    train-dora-seed)   DEFAULT_HOURS=12 ;;
    eval-ner-dora-seed) DEFAULT_HOURS=8  ;;
    bench-ft-dora-seed) DEFAULT_HOURS=4  ;;
    bench-luo)         DEFAULT_HOURS=6  ;;  # ext (~15min) + gsm8k generative 5-shot (~1-3h)
    bench-scripted)    DEFAULT_HOURS=4  ;;  # crows_pairs + mathqa (datasets<3.0 venv)
    train-sdft-full)        DEFAULT_HOURS=12 ;;  # SDFT EMA β=0 (control); ~8 jobs of 12h
    train-sdft-full-budget) DEFAULT_HOURS=12 ;;  # SDFT fixed teacher, budget matched to LoRA
    train-sdft-budget-seed) DEFAULT_HOURS=12 ;;  # extra seed of the budget run
    train-sdft-anchor)      DEFAULT_HOURS=12 ;;  # KL anchor β (dissertation); env BETA + SEED
    eval-ner-sdft-full) DEFAULT_HOURS=12 ;;  # dev/test have ~18.7k examples
    bench-ft-sdft-full) DEFAULT_HOURS=4  ;;
    train-lora-frac)   DEFAULT_HOURS=12 ;;
    eval-ner-frac)     DEFAULT_HOURS=8  ;;
    *)           DEFAULT_HOURS=8  ;;
esac
HOURS="${HOURS:-$DEFAULT_HOURS}"
if (( HOURS > 12 )); then
    echo "Error: the cluster caps walltime at 12 hours (requested: ${HOURS}h)"
    exit 1
fi
TIME_STR=$(printf "%02d:00:00" "$HOURS")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Launcher: torchrun (DDP) for training and NER evaluation; single-GPU python otherwise.
# BERT uses 1 GPU to avoid RCCL all_reduce (NaN on ROCm MI250 without iommu=pt).
case "$STEP" in
    train-bert-base | train-bert-large) NUM_GPUS=1; LAUNCHER="python" ;;
    train-* | eval-ner | eval-ner-sdft-full | eval-ner-r4-dev | eval-ner-r8-dev | eval-ner-r16-dev | eval-ner-base | eval-ner-fewshot | eval-ner-fewshot3 | eval-ner-dora | eval-ner-seed | eval-ner-dora-seed | eval-ner-frac | eval-bert-base | eval-bert-large) NUM_GPUS=${GPUS_PER_NODE}; LAUNCHER="torchrun --nproc_per_node=${GPUS_PER_NODE}" ;;
    *) NUM_GPUS=1; LAUNCHER="python" ;;
esac

# Python command for each step.
case "$STEP" in
    bench-base)
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model Qwen/Qwen3-4B --tag base"
        JOB_NAME="bench-base"
        ;;
    train-lora)
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 8 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-lora-r8"
        ;;
    train-dora)
        PY_CMD="${LAUNCHER} experiments/train.py --adapter dora --rank 8 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-dora-r8"
        ;;
    train-r4)
        # Rank sweep: r=4 (same hyperparameters as train-lora r=8).
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 4 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-lora-r4"
        ;;
    train-r16)
        # Rank sweep: r=16 (same hyperparameters as train-lora r=8).
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 16 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-lora-r16"
        ;;
    eval-ner)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt --split test --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner"
        ;;
    eval-ner-r4-dev)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r4_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r4-dev"
        ;;
    eval-ner-r8-dev)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r8-dev"
        ;;
    eval-ner-r16-dev)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r16_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r16-dev"
        ;;
    eval-ner-base)
        # Zero-shot baseline (NCCL 4h timeout prevents a crash on a long run).
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 8 --max-new-tokens 1024"
        JOB_NAME="eval-ner-base"
        ;;
    eval-ner-fewshot)
        # 1-shot adaptive per (act, entity_type): the exact demo for each entity type.
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 8 --max-new-tokens 1024 --few-shot 1"
        JOB_NAME="eval-ner-fewshot"
        ;;
    eval-ner-fewshot3)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 4 --max-new-tokens 1024 --few-shot 3"
        JOB_NAME="eval-ner-fewshot3"
        ;;
    eval-ner-dora)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/dora_r8_fmt --split test --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-dora"
        ;;
    train-sdft-full)
        # SDFT EMA β=0 (dissertation CONTROL): 87k, 3 epochs, EMA teacher with no anchor.
        # Collapses over the long horizon — the baseline the β anchor rescues.
        # grad-accum derived from the GPU count for an effective batch of 32 (mi250/8→1, mi210/4→2).
        # ~8 jobs of 12h chained (--dep afterany); auto-resume. Checkpoints under WORK.
        GA=$(( 8 / GPUS_PER_NODE ))
        PY_CMD="${LAUNCHER} experiments/train_sdft.py --rank 8 --epochs 3 --grad-accum ${GA} --output-dir ${WORK}/checkpoints/sdft_r8_full_fmt"
        JOB_NAME="train-sdft-r8-full"
        ;;
    train-sdft-full-budget)
        # The paper's SDFT: FIXED teacher (--no-ema) + truncation mask, budget matched to
        # LoRA (3 epochs, lr 2e-4). With a fixed teacher the target is stationary
        # ("reproduce the gold and stop"), so there is no non-termination collapse.
        # grad-accum derived from the GPU count for an effective batch of 32. ~8 jobs of 12h chained.
        GA=$(( 8 / GPUS_PER_NODE ))
        PY_CMD="${LAUNCHER} experiments/train_sdft.py --rank 8 --epochs 3 --lr 2e-4 --grad-accum ${GA} --no-ema --mask-truncated --output-dir ${WORK}/checkpoints/sdft_r8_full_budget"
        JOB_NAME="train-sdft-r8-full-budget"
        ;;
    train-sdft-budget-seed)
        # Extra seed of the SDFT budget run (same config; only the seed changes). Env: SEED.
        : "${SEED:?set SEED=43 (or 44)}"
        GA=$(( 8 / GPUS_PER_NODE ))
        PY_CMD="${LAUNCHER} experiments/train_sdft.py --rank 8 --epochs 3 --lr 2e-4 --grad-accum ${GA} --no-ema --mask-truncated --seed ${SEED} --output-dir ${WORK}/checkpoints/sdft_r8_full_budget_s${SEED}"
        JOB_NAME="train-sdft-budget-s${SEED}"
        ;;
    train-sdft-anchor)
        # Dissertation experiment: EMA teacher + KL anchor to the base (β>0). Same config
        # as the train-sdft-full control; the only new variable is --beta. The β sweep
        # maps the H1×H2 trade-off. Env: BETA (required) + SEED (default 42).
        # Directory per β+seed; ~8 jobs of 12h chained (--dep afterany), auto-resume.
        : "${BETA:?set BETA=0.3 (or 0.1/0.5) — e.g.: BETA=0.3 ./experiments/submit.sh train-sdft-anchor mi250 12}"
        SEED="${SEED:-42}"
        BTAG=$(echo "$BETA" | tr -d '.')
        GA=$(( 8 / GPUS_PER_NODE ))
        DST="${WORK}/checkpoints/sdft_r8_anchor_b${BTAG}_s${SEED}"
        PY_CMD="${LAUNCHER} experiments/train_sdft.py --rank 8 --epochs 3 --grad-accum ${GA} --beta ${BETA} --seed ${SEED} --output-dir ${DST}"
        JOB_NAME="train-sdft-anchor-b${BTAG}-s${SEED}"
        ;;
    eval-ner-sdft-full)
        # SDFT H1. Checkpoint selection is done ON DEV (SPLIT=dev), then test on the
        # winner. Env: SDFT_MODEL (dir or checkpoint-N), SPLIT (default test), LIMIT.
        SDFT_MODEL="${SDFT_MODEL:-${WORK}/checkpoints/sdft_r8_full_budget}"
        SPLIT="${SPLIT:-test}"
        # Tag includes the parent dir to avoid collisions across runs (checkpoint-N → <parent>_checkpoint-N).
        _b=$(basename "$SDFT_MODEL")
        _parent=$(basename "$(dirname "$SDFT_MODEL")")
        if [[ "$_b" == checkpoint-* ]]; then OUT_TAG="${_parent}_${_b}"; else OUT_TAG="$_b"; fi
        LIMIT_ARG=""; LIMIT_TAG=""
        if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARG=" --limit ${LIMIT}"; LIMIT_TAG="_lim${LIMIT}"; fi
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model ${SDFT_MODEL} --split ${SPLIT} --batch-size 16 --max-new-tokens 1024${LIMIT_ARG} --output results/ner_${OUT_TAG}${LIMIT_TAG}_${SPLIT}.json"
        JOB_NAME="eval-ner-sdft-full"
        ;;
    bench-ft-sdft-full)
        # SDFT H2 (forgetting vs base). Env: SDFT_MODEL (dir or checkpoint-N).
        SDFT_MODEL="${SDFT_MODEL:-${WORK}/checkpoints/sdft_r8_full_budget}"
        _b=$(basename "$SDFT_MODEL")
        _parent=$(basename "$(dirname "$SDFT_MODEL")")
        if [[ "$_b" == checkpoint-* ]]; then TAG="${_parent}_${_b}"; else TAG="$_b"; fi
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model ${SDFT_MODEL} --tag ${TAG}"
        JOB_NAME="bench-ft-sdft-full"
        ;;
    train-bert-base)
        # HIP_VISIBLE_DEVICES=0: 1 physical GPU (no DataParallel/DDP/RCCL). Default BERT batch/lr.
        PY_CMD="HIP_VISIBLE_DEVICES=0 ${LAUNCHER} experiments/train_bertimbau.py --model neuralmind/bert-base-portuguese-cased --batch-size 32 --lr 2e-5"
        JOB_NAME="train-bert-base"
        ;;
    train-bert-large)
        # Batch=16 due to VRAM limits (large ~1.3GB/sample in fp16); lr=2e-5 default.
        PY_CMD="HIP_VISIBLE_DEVICES=0 ${LAUNCHER} experiments/train_bertimbau.py --model neuralmind/bert-large-portuguese-cased --batch-size 16 --lr 2e-5"
        JOB_NAME="train-bert-large"
        ;;
    eval-bert-base)
        PY_CMD="${LAUNCHER} experiments/evaluate_bertimbau.py --model results/checkpoints/bertimbau_base --split test"
        JOB_NAME="eval-bert-base"
        ;;
    eval-bert-large)
        PY_CMD="${LAUNCHER} experiments/evaluate_bertimbau.py --model results/checkpoints/bertimbau_large --split test"
        JOB_NAME="eval-bert-large"
        ;;
    bench-ft)
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model results/checkpoints/lora_r8_fmt --tag lora_r8"
        JOB_NAME="bench-ft-lora-r8"
        ;;
    bench-ft-dora)
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model results/checkpoints/dora_r8_fmt --tag dora_r8"
        JOB_NAME="bench-ft-dora-r8"
        ;;
    train-lora-seed)
        # Extra LoRA r8 seed to estimate the variance of H2. Env: SEED (43, 44).
        # Output → lora_r8_fmt_s${SEED} (automatic suffix in train.py).
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 8 --batch-size 4 --grad-accum 4 --max-length 2048 --seed ${SEED}"
        JOB_NAME="train-lora-s${SEED}"
        ;;
    eval-ner-seed)
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt_s${SEED} --split test --batch-size 16 --max-new-tokens 1024 --output results/ner_lora_r8_fmt_s${SEED}_test.json"
        JOB_NAME="eval-ner-s${SEED}"
        ;;
    bench-ft-seed)
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model results/checkpoints/lora_r8_fmt_s${SEED} --tag lora_r8_s${SEED}"
        JOB_NAME="bench-ft-s${SEED}"
        ;;
    train-dora-seed)
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/train.py --adapter dora --rank 8 --batch-size 4 --grad-accum 4 --max-length 2048 --seed ${SEED}"
        JOB_NAME="train-dora-s${SEED}"
        ;;
    eval-ner-dora-seed)
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/dora_r8_fmt_s${SEED} --split test --batch-size 16 --max-new-tokens 1024 --output results/ner_dora_r8_fmt_s${SEED}_test.json"
        JOB_NAME="eval-ner-dora-s${SEED}"
        ;;
    bench-ft-dora-seed)
        : "${SEED:?set SEED=43 (or 44)}"
        PY_CMD="${LAUNCHER} experiments/evaluate_forgetting.py --model results/checkpoints/dora_r8_fmt_s${SEED} --tag dora_r8_s${SEED}"
        JOB_NAME="bench-ft-dora-s${SEED}"
        ;;
    bench-luo)
        # Forgetting (H2) on the Luo et al. (2023) axes: ext (0-shot) + gsm8k
        # (generative, 5-shot). Env: MODEL and TAG. Outputs bench_ext_/bench_gsm8k_ (leaves core untouched).
        : "${MODEL:?set MODEL=<checkpoint path or HF id, e.g., Qwen/Qwen3-4B>}"
        : "${TAG:?set TAG=<e.g., base, lora_r8, sdft_s44>}"
        _LIM="${LIMIT:+--limit ${LIMIT}}"
        PY_CMD="python experiments/evaluate_forgetting.py --model ${MODEL} --tag ${TAG} --task-set ext ${_LIM} && python experiments/evaluate_forgetting.py --model ${MODEL} --tag ${TAG} --task-set gsm8k ${_LIM}"
        JOB_NAME="bench-luo-${TAG}"
        ;;
    bench-scripted)
        # CrowS-Pairs (bias) + MathQA: datasets with a .py loading script, only in the
        # isolated datasets<3.0 venv (use EVAL_VENV=eval-venv-ds2). Env: MODEL and TAG.
        : "${MODEL:?set MODEL=<checkpoint path or HF id>}"
        : "${TAG:?set TAG=<e.g., base, lora_r8, sdft_r8_s43>}"
        _LIM="${LIMIT:+--limit ${LIMIT}}"
        PY_CMD="python experiments/evaluate_forgetting.py --model ${MODEL} --tag ${TAG} --task-set scripted ${_LIM}"
        JOB_NAME="bench-scripted-${TAG}"
        ;;
    train-lora-frac)
        # Data curve: LoRA r8 on a FRACTION of the training set. Env: FRAC (0.25, 0.50).
        # Output auto-suffixed lora_r8_fmt_f${pct}.
        : "${FRAC:?set FRAC=0.25 (or 0.50)}"
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 8 --batch-size 4 --grad-accum 4 --max-length 2048 --train-frac ${FRAC}"
        JOB_NAME="train-lora-frac"
        ;;
    eval-ner-frac)
        # Fraction H1. Env: PCT (25 or 50), matching the suffix produced by train.py.
        : "${PCT:?set PCT=25 (or 50)}"
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt_f${PCT} --split test --batch-size 16 --max-new-tokens 1024 --output results/ner_lora_r8_fmt_f${PCT}_test.json"
        JOB_NAME="eval-ner-frac${PCT}"
        ;;
esac

# SDFT trains in its own venv (trl 0.24); evaluation and other steps use the
# finetune-venv. See experiments/requirements-sdft.txt.
case "$STEP" in
    train-sdft-full | train-sdft-full-budget | train-sdft-budget-seed | train-sdft-anchor) VENV_NAME="sdft-venv" ;;
    *) VENV_NAME="finetune-venv" ;;
esac

# Evaluation venv override. E.g., EVAL_VENV=eval-venv-ds2 (a clone with datasets<3.0)
# to run bench-scripted (crows_pairs/mathqa use a loading script).
if [[ -n "${EVAL_VENV:-}" ]]; then VENV_NAME="$EVAL_VENV"; fi

TEMP_SCRIPT="/tmp/slurm_${JOB_NAME}_$$.sh"

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$((NUM_GPUS * 8))
#SBATCH --time=${TIME_STR}
#SBATCH --output=${JOB_NAME}-%j.out
#SBATCH --error=${JOB_NAME}-%j.err
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME="${WORK}/hf_cache"
export NCCL_DEBUG=WARN
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8
export HSA_ENABLE_SDMA=0

cd "${PROJECT_DIR}" || exit 1
VENV_DIR="\${WORK}/${VENV_NAME}"
if [[ ! -f "\${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: venv not found at \${VENV_DIR}"
    echo "Run on the cluster: cd ~/finetune && bash scripts/setup_cluster.sh"
    exit 1
fi
echo "=============================="
echo "Step    : ${STEP}"
echo "Job ID  : \$SLURM_JOB_ID"
echo "Node    : \$(hostname)"
echo "Start   : \$(date)"
echo "=============================="


source "\${VENV_DIR}/bin/activate"

${PY_CMD}

echo ""
echo "End: \$(date)"
EOF

SBATCH_OPTS=""
if [[ -n "$DEP_JOBID" ]]; then
    SBATCH_OPTS="--dependency=afterany:${DEP_JOBID}"
fi
JOB_ID=$(sbatch $SBATCH_OPTS "$TEMP_SCRIPT" | grep -oP "Submitted batch job \K[0-9]+")
rm -f "$TEMP_SCRIPT"

echo "=============================="
echo "Step     : $STEP"
echo "Partition: $PARTITION  |  Walltime: $TIME_STR"
echo "Job ID   : $JOB_ID"
[[ -n "$DEP_JOBID" ]] && echo "Depends on: $DEP_JOBID (afterany)"
echo ""
echo "Monitor:"
echo "  squeue -j $JOB_ID"
echo "  tail -f ${JOB_NAME}-${JOB_ID}.out"
echo "=============================="
