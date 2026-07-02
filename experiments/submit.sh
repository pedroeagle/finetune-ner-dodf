#!/bin/bash
# Submete cada etapa do experimento ao SLURM (cluster AMD ROCm).
#
# Uso:
#   ./experiments/submit.sh <etapa> [gpu_type] [horas] [--dep <jobid>]
#
# O formato de saída esperado é gravado no dataset (campo instruction_fmt, ver
# scripts/build_dataset.py) e usado no treino e na avaliação; os artefatos do
# decoder carregam o sufixo _fmt (ex.: checkpoints/lora_r8_fmt, ner_lora_r8_fmt_test.json).
#
# Etapas disponíveis:
#   bench-base    — benchmarks MCQ no modelo base (H2, passo 1)
#   train-lora    — fine-tuning com LoRA  r=8
#   train-dora    — fine-tuning com DoRA  r=8
#   train-sweep   — varredura de rank (r=4,8,16) com LoRA, sequencial num job
#   train-r4      — fine-tuning LoRA r=4  (varredura, não toca no r=8 existente)
#   train-r16     — fine-tuning LoRA r=16 (varredura, não toca no r=8 existente)
#   eval-ner      — avaliação NER no modelo LoRA fine-tuned (H1)
#   eval-ner-r4-dev   — avaliação NER r=4  no split dev (seleção de rank)
#   eval-ner-r8-dev   — avaliação NER r=8  no split dev (seleção de rank)
#   eval-ner-r16-dev  — avaliação NER r=16 no split dev (seleção de rank)
#   compare-sweep — ΔF1 r=4 e r=16 vs r=8 no dev (roda localmente, sem GPU)
#   eval-ner-base — avaliação NER no modelo base zero-shot (H1, baseline)
#   eval-ner-fewshot  — avaliação NER baseline 1-shot adaptativo por (ato, entity_type)
#   eval-ner-fewshot3 — avaliação NER baseline 3-shot adaptativo por (ato, entity_type)
#   eval-ner-dora — avaliação NER no modelo DoRA fine-tuned (H1)
#   train-bert-base   — fine-tuning do BERTimbau base (encoder, SOTA NER)
#   train-bert-large  — fine-tuning do BERTimbau large (encoder, SOTA NER)
#   eval-bert-base    — avaliação NER do BERTimbau base (mesmos registros do decoder)
#   eval-bert-large   — avaliação NER do BERTimbau large (mesmos registros do decoder)
#   bench-ft      — benchmarks MCQ no modelo LoRA fine-tuned (H2, passo 3)
#   bench-ft-dora — benchmarks MCQ no modelo DoRA fine-tuned (H2, passo 3)
#   compare-ner   — ΔF1 + bootstrap decoder FT vs zero-shot (roda localmente, sem GPU)
#   compare-bert  — ΔF1 + bootstrap BERTimbau vs decoder FT (roda localmente, sem GPU)
#   compare-bench — Δ pp por benchmark (roda localmente, sem GPU)
#
# gpu_type: mi250 | mi300 | mi325 | mi350  (padrão: mi250)
# horas:    1-12                            (padrão: depende da etapa; máximo do cluster: 12h)
#
# --dep <jobid>  aguarda o job <jobid> terminar (qualquer motivo, inclusive
#                walltime) antes de iniciar — útil para encadear continuações.
#
# Exemplo completo:
#   ./experiments/submit.sh bench-base       mi250 4
#   ./experiments/submit.sh train-lora       mi250 12
#   ./experiments/submit.sh train-lora       mi250 12 --dep <JOB_ID>
#   ./experiments/submit.sh eval-ner         mi250 8
#   ./experiments/submit.sh eval-ner-base    mi250 8
#   ./experiments/submit.sh eval-ner-fewshot mi250 8
#   ./experiments/submit.sh bench-ft         mi250 4

set -e

STEP="${1:-}"
GPU_TYPE="${2:-mi250}"
HOURS="${3:-}"

# Suporte a --dep <jobid>: encadeia este job após outro terminar (qualquer motivo)
DEP_JOBID=""
for arg in "$@"; do
    if [[ "$arg" == "--dep" ]]; then
        _next=1
    elif [[ "${_next:-0}" == "1" ]]; then
        DEP_JOBID="$arg"
        _next=0
    fi
done

# Validação do step
VALID_STEPS="bench-base train-lora train-dora train-sweep train-r4 train-r16 eval-ner eval-ner-r4-dev eval-ner-r8-dev eval-ner-r16-dev eval-ner-base eval-ner-fewshot eval-ner-fewshot3 eval-ner-dora train-bert-base train-bert-large eval-bert-base eval-bert-large bench-ft bench-ft-dora compare-ner compare-bert compare-bench compare-sweep diag"
if [[ -z "$STEP" || ! " $VALID_STEPS " =~ " $STEP " ]]; then
    echo "Uso: $0 <etapa> [gpu_type] [horas]"
    echo "Etapas: $VALID_STEPS"
    exit 1
fi

# Etapas de comparação não precisam de GPU
if [[ "$STEP" == "compare-ner" ]]; then
    # H1: decoder LoRA fmt vs zero-shot fmt
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
    # Seleção de rank no dev: ΔF1 de r=4 e r=16 contra o r=8 (referência)
    echo "### r=4 vs r=8 (dev) ###"
    python experiments/evaluate_ner.py \
        --compare results/ner_lora_r4_fmt_dev.json results/ner_lora_r8_fmt_dev.json
    echo ""
    echo "### r=16 vs r=8 (dev) ###"
    python experiments/evaluate_ner.py \
        --compare results/ner_lora_r16_fmt_dev.json results/ner_lora_r8_fmt_dev.json
    exit 0
fi

# Mapeamento gpu_type → partição SLURM
case "$GPU_TYPE" in
    mi250) PARTITION="mi2508x"; GPUS_PER_NODE=8 ;;
    mi210) PARTITION="mi2104x"; GPUS_PER_NODE=4 ;;  # 4 GPUs/nó
    mi300) PARTITION="mi3008x"; GPUS_PER_NODE=8 ;;
    mi325) PARTITION="mi3258x"; GPUS_PER_NODE=8 ;;
    mi350) PARTITION="mi3508x"; GPUS_PER_NODE=8 ;;
    *)
        echo "gpu_type inválido. Use: mi210, mi250, mi300, mi325 ou mi350"
        exit 1
        ;;
esac

# Walltime padrão por etapa
case "$STEP" in
    bench-base)    DEFAULT_HOURS=4  ;;
    train-lora)    DEFAULT_HOURS=12 ;;
    train-dora)    DEFAULT_HOURS=12 ;;
    train-sweep)   DEFAULT_HOURS=12 ;;
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
    diag)          DEFAULT_HOURS=1  ;;
    *)           DEFAULT_HOURS=8  ;;
esac
HOURS="${HOURS:-$DEFAULT_HOURS}"
if (( HOURS > 12 )); then
    echo "Erro: o cluster limita o walltime a 12 horas (solicitado: ${HOURS}h)"
    exit 1
fi
TIME_STR=$(printf "%02d:00:00" "$HOURS")

# Diretório raiz do projeto (pai de experiments/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Launcher: torchrun para etapas de treino, python simples para avaliação
# BERT usa GPU única para evitar RCCL all_reduce (causa NaN no ROCm MI250 sem iommu=pt)
case "$STEP" in
    train-bert-base | train-bert-large)                                                                                                NUM_GPUS=1; LAUNCHER="python" ;;
    train-* | eval-ner | eval-ner-r4-dev | eval-ner-r8-dev | eval-ner-r16-dev | eval-ner-base | eval-ner-fewshot | eval-ner-fewshot3 | eval-ner-dora | eval-bert-base | eval-bert-large) NUM_GPUS=${GPUS_PER_NODE}; LAUNCHER="torchrun --nproc_per_node=${GPUS_PER_NODE}" ;;
    *)                                                                                                                                 NUM_GPUS=1; LAUNCHER="python" ;;
esac

# Comando Python para cada etapa
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
    train-sweep)
        # Treina LoRA com r=4,8,16 sequencialmente dentro do mesmo job
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 4 && \
                ${LAUNCHER} experiments/train.py --adapter lora --rank 8 && \
                ${LAUNCHER} experiments/train.py --adapter lora --rank 16"
        JOB_NAME="train-lora-sweep"
        ;;
    train-r4)
        # Varredura de rank: r=4 (mesmos hiperparâmetros do train-lora r=8)
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 4 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-lora-r4"
        ;;
    train-r16)
        # Varredura de rank: r=16 (mesmos hiperparâmetros do train-lora r=8)
        PY_CMD="${LAUNCHER} experiments/train.py --adapter lora --rank 16 --batch-size 4 --grad-accum 4 --max-length 2048"
        JOB_NAME="train-lora-r16"
        ;;
    eval-ner)
        # Avalia o modelo LoRA fine-tuned → ner_lora_r8_fmt_test.json
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt --split test --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner"
        ;;
    eval-ner-r4-dev)
        # Seleção de rank no dev (mesmos parâmetros de eval do eval-ner)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r4_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r4-dev"
        ;;
    eval-ner-r8-dev)
        # Seleção de rank no dev (mesmos parâmetros de eval do eval-ner)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r8_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r8-dev"
        ;;
    eval-ner-r16-dev)
        # Seleção de rank no dev (mesmos parâmetros de eval do eval-ner)
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/lora_r16_fmt --split dev --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-r16-dev"
        ;;
    eval-ner-base)
        # Avalia baseline zero-shot em 8 GPUs (~1.3h); NCCL 4h timeout previne crash
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 8 --max-new-tokens 1024"
        JOB_NAME="eval-ner-base"
        ;;
    eval-ner-fewshot)
        # 1-shot adaptativo por (ato, entity_type): demo exato p/ cada tipo de entidade
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 8 --max-new-tokens 1024 --few-shot 1"
        JOB_NAME="eval-ner-fewshot"
        ;;
    eval-ner-fewshot3)
        # 3-shot adaptativo por (ato, entity_type): batch menor p/ memória
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model Qwen/Qwen3-4B --split test --batch-size 4 --max-new-tokens 1024 --few-shot 3"
        JOB_NAME="eval-ner-fewshot3"
        ;;
    eval-ner-dora)
        # Avalia o modelo DoRA fine-tuned → ner_dora_r8_fmt_test.json
        PY_CMD="${LAUNCHER} experiments/evaluate_ner.py --model results/checkpoints/dora_r8_fmt --split test --batch-size 16 --max-new-tokens 1024"
        JOB_NAME="eval-ner-dora"
        ;;
    train-bert-base)
        # HIP_VISIBLE_DEVICES=0: força 1 GPU física — sem DataParallel/DDP/RCCL
        # fp16+AMP no script: GradScaler descarta passos com overflow automaticamente.
        # Batch=32 e lr=2e-5: valores padrão da literatura de BERT NER.
        PY_CMD="HIP_VISIBLE_DEVICES=0 ${LAUNCHER} experiments/train_bertimbau.py --model neuralmind/bert-base-portuguese-cased --batch-size 32 --lr 2e-5"
        JOB_NAME="train-bert-base"
        ;;
    train-bert-large)
        # HIP_VISIBLE_DEVICES=0: força 1 GPU física — sem DataParallel/DDP/RCCL
        # Batch=16 por limitação de VRAM (large ~1.3GB/sample em fp16); lr=2e-5 padrão.
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
    diag)
        # Diagnóstico de NaN do adapter (1 GPU, rápido)
        PY_CMD="${LAUNCHER} experiments/diagnose_adapter.py --model results/checkpoints/lora_r8_fmt"
        JOB_NAME="diag-adapter"
        ;;
esac

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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
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
VENV_DIR="\${WORK}/finetune-venv"
if [[ ! -f "\${VENV_DIR}/bin/activate" ]]; then
    echo "ERRO: venv nao encontrado em \${VENV_DIR}"
    echo "Execute no cluster: cd ~/finetune && bash setup_cluster.sh"
    exit 1
fi
echo "=============================="
echo "Etapa   : ${STEP}"
echo "Job ID  : \$SLURM_JOB_ID"
echo "Node    : \$(hostname)"
echo "Início  : \$(date)"
echo "=============================="


source "\${VENV_DIR}/bin/activate"

${PY_CMD}

echo ""
echo "Fim: \$(date)"
EOF

SBATCH_OPTS=""
if [[ -n "$DEP_JOBID" ]]; then
    SBATCH_OPTS="--dependency=afterany:${DEP_JOBID}"
fi
JOB_ID=$(sbatch $SBATCH_OPTS "$TEMP_SCRIPT" | grep -oP "Submitted batch job \K[0-9]+")
rm -f "$TEMP_SCRIPT"

echo "=============================="
echo "Etapa   : $STEP"
echo "Partição: $PARTITION  |  Walltime: $TIME_STR"
echo "Job ID  : $JOB_ID"
[[ -n "$DEP_JOBID" ]] && echo "Depende de: $DEP_JOBID (afterany)"
echo ""
echo "Monitorar:"
echo "  squeue -j $JOB_ID"
echo "  tail -f ${JOB_NAME}-${JOB_ID}.out"
echo "=============================="
