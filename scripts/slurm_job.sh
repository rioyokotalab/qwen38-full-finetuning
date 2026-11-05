#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?missing PROJECT_ROOT}"
: "${GPUS_PER_NODE:?missing GPUS_PER_NODE}"
source "$PROJECT_ROOT/config/site_setup.sh"
set -a
source "$PROJECT_ROOT/model.lock"
set +a

export HF_HOME TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export EXPECTED_NODES="$SLURM_NNODES"
export EXPECTED_WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

test -x "$PROJECT_ROOT/.venv/bin/torchrun"
test -x "$PROJECT_ROOT/scripts/distributed_entry.sh"
test -f "$PROJECT_ROOT/train_full_sft.py"
test -f "$PROJECT_ROOT/nccl_probe.py"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

mode=${1:?usage: slurm_job.sh smoke|train}
if [[ "$mode" == smoke ]]; then
  echo "NATIVE phase=topology job=$SLURM_JOB_ID nodes=$SLURM_NNODES master=$MASTER_ADDR"
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 "$PROJECT_ROOT/scripts/distributed_entry.sh" nccl
  MASTER_PORT=$((MASTER_PORT + 1)); export MASTER_PORT
  export TRAIN_DATA="$PROJECT_ROOT/data/smoke.jsonl"
  export OUTPUT_DIR="$PROJECT_ROOT/outputs/smoke-$SLURM_JOB_ID"
  export MAX_LENGTH=128 MAX_STEPS=1 GRADIENT_ACCUMULATION_STEPS=1
  unset SAVE_CHECKPOINT || true
  echo "NATIVE phase=full-parameter-finetune model=$MODEL_ID revision=$MODEL_REVISION"
elif [[ "$mode" != train ]]; then
  echo "unknown mode: $mode" >&2
  exit 64
fi

: "${TRAIN_DATA:?missing TRAIN_DATA}"
: "${OUTPUT_DIR:?missing OUTPUT_DIR}"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 "$PROJECT_ROOT/scripts/distributed_entry.sh" train
