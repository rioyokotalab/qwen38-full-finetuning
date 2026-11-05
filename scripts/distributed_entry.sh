#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?missing PROJECT_ROOT}"
: "${MASTER_ADDR:?missing MASTER_ADDR}"
: "${MASTER_PORT:?missing MASTER_PORT}"
: "${SLURM_NNODES:?missing SLURM_NNODES}"
: "${SLURM_NODEID:?missing SLURM_NODEID}"
: "${GPUS_PER_NODE:?missing GPUS_PER_NODE}"

launcher=(
  "$PROJECT_ROOT/.venv/bin/torchrun"
  --nnodes="$SLURM_NNODES"
  --nproc-per-node="$GPUS_PER_NODE"
  --node-rank="$SLURM_NODEID"
  --master-addr="$MASTER_ADDR"
  --master-port="$MASTER_PORT"
)

case "${1:-}" in
  nccl)
    exec "${launcher[@]}" "$PROJECT_ROOT/nccl_probe.py"
    ;;
  train)
    : "${MODEL_ID:?missing MODEL_ID}"
    : "${MODEL_REVISION:?missing MODEL_REVISION}"
    : "${TRAIN_DATA:?missing TRAIN_DATA}"
    : "${OUTPUT_DIR:?missing OUTPUT_DIR}"
    args=(
      --model "$MODEL_ID" --revision "$MODEL_REVISION"
      --data "$TRAIN_DATA" --output "$OUTPUT_DIR"
      --max-length "${MAX_LENGTH:-4096}"
      --max-steps "${MAX_STEPS:-1000}"
      --learning-rate "${LEARNING_RATE:-1e-6}"
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
    )
    [[ -n "${SAVE_CHECKPOINT:-}" ]] && args+=(--save)
    exec "${launcher[@]}" "$PROJECT_ROOT/train_full_sft.py" "${args[@]}"
    ;;
  *)
    echo "usage: $0 nccl|train" >&2
    exit 64
    ;;
esac
