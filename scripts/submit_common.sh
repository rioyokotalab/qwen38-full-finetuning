#!/usr/bin/env bash
set -euo pipefail

mode=${1:?missing mode}
config=${2:-config/cluster.env}
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 66; }
set -a
source "$config"
set +a

for name in PROJECT_ROOT SLURM_PARTITION NODES GPUS_PER_NODE GPU_REQUEST_STYLE CPUS_PER_TASK TIME_LIMIT HF_HOME LOG_DIR; do
  [[ -n "${!name:-}" ]] || { echo "missing $name in $config" >&2; exit 64; }
done
[[ "$PROJECT_ROOT" == /* && "$HF_HOME" == /* && "$LOG_DIR" == /* ]] || {
  echo "PROJECT_ROOT, HF_HOME, and LOG_DIR must be absolute shared paths" >&2; exit 64;
}
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/outputs"

args=(
  --parsable
  --job-name="qwen38-27b-$mode"
  --partition="$SLURM_PARTITION"
  --nodes="$NODES"
  --ntasks-per-node=1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="${MEMORY_REQUEST:-0}"
  --time="$TIME_LIMIT"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
)
[[ -n "${SLURM_ACCOUNT:-}" ]] && args+=(--account="$SLURM_ACCOUNT")
case "$GPU_REQUEST_STYLE" in
  per-node) args+=(--gpus-per-node="$GPUS_PER_NODE") ;;
  total) args+=(--gpus="$((NODES * GPUS_PER_NODE))") ;;
  gres) args+=(--gres="gpu:$GPUS_PER_NODE") ;;
  *) echo "GPU_REQUEST_STYLE must be per-node, total, or gres" >&2; exit 64 ;;
esac
if [[ -n "${SBATCH_EXTRA:-}" ]]; then
  read -r -a extra <<<"$SBATCH_EXTRA"
  args+=("${extra[@]}")
fi

export PROJECT_ROOT NODES GPUS_PER_NODE HF_HOME
job=$(sbatch "${args[@]}" --export=ALL "$PROJECT_ROOT/scripts/slurm_job.sh" "$mode")
echo "SUBMITTED job=$job mode=$mode log=$LOG_DIR/qwen38-27b-$mode-$job.out"
