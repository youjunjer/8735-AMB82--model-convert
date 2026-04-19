#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/home/youadmin/.venvs/amb82-model-convert"

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip >/dev/null
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" >/dev/null

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" "$ROOT_DIR/model_convert_webui.py"
