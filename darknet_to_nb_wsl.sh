#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model.cfg> <model.weights> <work_dir> [calibration_dir]" >&2
  exit 1
fi

CFG_PATH="$1"
WEIGHTS_PATH="$2"
WORK_DIR="$3"
CALIB_DIR="${4:-}"

IMAGE="${ZIP_TO_NB_IMAGE:-ghcr.io/ameba-aiot/acuity-toolkit:6.18.8}"
DOCKER_ARGS="${ZIP_TO_NB_DOCKER_ARGS:-}"
QUANTIZE_DEVICE="${ZIP_TO_NB_QUANTIZE_DEVICE:-CPU}"
PEGASUS="/usr/local/acuity_command_line_tools/pegasus.py"

mkdir -p "$WORK_DIR"
rm -rf "$WORK_DIR"/*
mkdir -p "$WORK_DIR/input_model" "$WORK_DIR/out" "$WORK_DIR/calib_ascii"

cp "$CFG_PATH" "$WORK_DIR/input_model/model.cfg"
cp "$WEIGHTS_PATH" "$WORK_DIR/input_model/model.weights"

python3 - <<'PY' "$WORK_DIR/input_model/model.cfg" "$WORK_DIR/input_model/model_meta.json"
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
width = 416
height = 416
channels = 3
for raw in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        continue
    key, value = [part.strip() for part in line.split("=", 1)]
    if key == "width":
        width = int(value)
    elif key == "height":
        height = int(value)
    elif key == "channels":
        channels = int(value)

meta_path.write_text(
    json.dumps({"width": width, "height": height, "channels": channels}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"DARKNET_META width={width} height={height} channels={channels}")
PY

if [[ -n "$CALIB_DIR" ]]; then
  python3 - <<'PY' "$CALIB_DIR" "$WORK_DIR/calib_ascii"
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
files = sorted(
    p for p in src.iterdir()
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
)
for i, p in enumerate(files[:10], 1):
    suffix = p.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    shutil.copy2(p, dst / f"img_{i:02d}{suffix}")
PY
else
  python3 - <<'PY' "$WORK_DIR/calib_ascii" "$WORK_DIR/input_model/model_meta.json"
import json
import struct
import sys
import zlib
from pathlib import Path

dst = Path(sys.argv[1])
meta = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
width = int(meta["width"])
height = int(meta["height"])

def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"".join([
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, level=9)),
        chunk(b"IEND", b""),
    ])
    path.write_bytes(png)

for i in range(1, 11):
    write_png(dst / f"img_{i:02d}.png", width, height, (127, 127, 127))
PY
fi

python3 - <<'PY' "$WORK_DIR/calib_ascii" "$WORK_DIR/dataset_real.txt"
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
with out.open("w", encoding="utf-8") as f:
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            f.write(f"calib_ascii/{p.name}\n")
PY

docker run --rm ${DOCKER_ARGS} -v "$WORK_DIR:/workspace" "$IMAGE" /bin/bash -lc "
set -e
python3 $PEGASUS import darknet \
  --model /workspace/input_model/model.cfg \
  --weights /workspace/input_model/model.weights \
  --output-model /workspace/out/darknet_model \
  --output-data /workspace/out/darknet_model.data

python3 - <<'PY'
import json
from pathlib import Path

model = json.loads(Path('/workspace/out/darknet_model').read_text(encoding='utf-8'))
layers = model.get('Layers', {})
input_layer_key = None
input_layer_name = None
input_shape = None

for key, value in layers.items():
    if value.get('op') == 'input':
        input_layer_key = key
        input_layer_name = value.get('name') or key
        input_shape = value.get('parameters', {}).get('shape')
        break

if not input_layer_key or not input_shape or len(input_shape) != 4:
    raise SystemExit('Unable to detect darknet input layer or shape.')

n, c, h, w = input_shape
if not n or n <= 0:
    n = 1

content = f'''%YAML 1.2
---
input_meta:
  databases:
  - path: dataset_real.txt
    type: TEXT
    ports:
    - lid: {input_layer_key}
      category: image
      dtype: float32
      sparse: false
      tensor_name: {input_layer_name}
      layout: nchw
      shape:
      - {n}
      - {c}
      - {h}
      - {w}
      fitting: scale
      preprocess:
        reverse_channel: false
        mean:
        - 0
        - 0
        - 0
        scale: 0.00392156862745098
        preproc_node_params:
          add_preproc_node: false
          preproc_type: IMAGE_RGB
          preproc_perm:
          - 0
          - 1
          - 2
          - 3
      redirect_to_output: false
'''
Path('/workspace/out/inputmeta_darknet.yml').write_text(content, encoding='utf-8')
print('DARKNET_INPUTMETA', input_layer_key, input_layer_name, [n, c, h, w])
PY

python3 $PEGASUS quantize \
  --model /workspace/out/darknet_model \
  --model-data /workspace/out/darknet_model.data \
  --with-input-meta /workspace/out/inputmeta_darknet.yml \
  --iterations 10 \
  --device $QUANTIZE_DEVICE \
  --output-dir /workspace/out/quantize_out \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --algorithm normal

python3 $PEGASUS export ovxlib \
  --model /workspace/out/darknet_model \
  --model-data /workspace/out/darknet_model.data \
  --model-quantize /workspace/out/darknet_model.quantize \
  --with-input-meta /workspace/out/inputmeta_darknet.yml \
  --dtype quantized \
  --output-path /workspace/out/build \
  --optimize VIP8000NANONI_PID0XAD \
  --pack-nbg-unify \
  --viv-sdk /opt/acuity/Vivante_IDE/VivanteIDE5.8.1.1/cmdtools

cp /workspace/out_nbg_unify/network_binary.nb /workspace/out_nbg_unify/yolov4_tiny.nb
chmod -R a+rwX /workspace/out /workspace/out_nbg_unify
"

if [[ ! -f "$WORK_DIR/out_nbg_unify/network_binary.nb" ]]; then
  echo "network_binary.nb was not generated." >&2
  exit 3
fi

echo "NB_OK $WORK_DIR/out_nbg_unify/yolov4_tiny.nb"
