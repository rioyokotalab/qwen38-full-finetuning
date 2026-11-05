#!/usr/bin/env bash
set -euo pipefail
: "${TRAIN_DATA:?Set TRAIN_DATA to an absolute shared JSONL path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to an absolute shared output path}"
[[ "$TRAIN_DATA" == /* && "$OUTPUT_DIR" == /* ]] || {
  echo "TRAIN_DATA and OUTPUT_DIR must be absolute shared paths" >&2; exit 64;
}
export MAX_LENGTH=${MAX_LENGTH:-4096}
export MAX_STEPS=${MAX_STEPS:-1000}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/submit_common.sh" train "${1:-config/cluster.env}"
