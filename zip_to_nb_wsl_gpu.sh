#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <converted_keras.zip> <work_dir> [calibration_dir] [cuda_cache_dir]" >&2
  exit 1
fi

ZIP_PATH="$1"
WORK_DIR="$2"
CALIB_DIR="${3:-}"
CACHE_DIR="${4:-/home/youadmin/work/gpu_shared_cache}"

mkdir -p "$CACHE_DIR"

export ZIP_TO_NB_IMAGE="acuity-toolkit:6.18.8-gpu"
export ZIP_TO_NB_DOCKER_ARGS="--gpus all -e CUDA_CACHE_PATH=$CACHE_DIR -e TF_FORCE_GPU_ALLOW_GROWTH=true"
export ZIP_TO_NB_QUANTIZE_DEVICE="GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/zip_to_nb_wsl.sh" "$ZIP_PATH" "$WORK_DIR" "$CALIB_DIR"
