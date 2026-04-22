#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <converted_keras.zip> <work_dir> [calibration_dir]" >&2
  exit 1
fi

ZIP_PATH="$1"
WORK_DIR="$2"
CALIB_DIR="${3:-}"

IMAGE="${ZIP_TO_NB_IMAGE:-ghcr.io/ameba-aiot/acuity-toolkit:6.18.8}"
DOCKER_ARGS="${ZIP_TO_NB_DOCKER_ARGS:-}"
QUANTIZE_DEVICE="${ZIP_TO_NB_QUANTIZE_DEVICE:-CPU}"
PEGASUS="/usr/local/acuity_command_line_tools/pegasus.py"

mkdir -p "$WORK_DIR"
rm -rf "$WORK_DIR"/*
mkdir -p "$WORK_DIR/input_zip" "$WORK_DIR/unzipped" "$WORK_DIR/out" "$WORK_DIR/calib_ascii"

cp "$ZIP_PATH" "$WORK_DIR/input_zip/converted_keras.zip"
python3 - <<'PY' "$WORK_DIR/input_zip/converted_keras.zip" "$WORK_DIR/unzipped"
import sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as zf:
    zf.extractall(dst)
PY

if [[ ! -f "$WORK_DIR/unzipped/keras_model.h5" ]]; then
  echo "keras_model.h5 not found in zip." >&2
  exit 2
fi

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
    shutil.copy2(p, dst / f"img_{i:02d}{p.suffix.lower() if p.suffix.lower() != '.jpeg' else '.jpg'}")
PY
else
  python3 - <<'PY' "$WORK_DIR/calib_ascii"
import struct
import sys
import zlib
from pathlib import Path


def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    raw = b"".join(
        b"\x00" + bytes(rgb) * width
        for _ in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, level=9)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


dst = Path(sys.argv[1])
for i in range(1, 11):
    write_png(dst / f"img_{i:02d}.png", 224, 224, (127, 127, 127))
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
python3 -m pip install --no-cache-dir tf2onnx==1.16.1 >/tmp/pip-tf2onnx.log 2>&1
python3 - <<'PY'
import tensorflow as tf
import tf2onnx
from pathlib import Path

model = tf.keras.models.load_model('/workspace/unzipped/keras_model.h5', compile=False)
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name='input'),)
out = Path('/workspace/out')
out.mkdir(exist_ok=True)
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=str(out / 'keras_model.onnx'),
)
print('ONNX_OK', [i.name for i in model_proto.graph.input], [o.name for o in model_proto.graph.output])
PY

python3 $PEGASUS import onnx --model /workspace/out/keras_model.onnx --output-model /workspace/out/keras_model

python3 - <<'PY'
from pathlib import Path

content = '''%YAML 1.2
---
input_meta:
  databases:
  - path: dataset_real.txt
    type: TEXT
    ports:
    - lid: input_105
      category: image
      dtype: float32
      sparse: false
      tensor_name:
      layout: nhwc
      shape:
      - 1
      - 224
      - 224
      - 3
      fitting: scale
      preprocess:
        reverse_channel: false
        mean:
        - 127.5
        - 127.5
        - 127.5
        scale: 0.007874015748031496
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
Path('/workspace/out/inputmeta_teachable.yml').write_text(content, encoding='utf-8')
PY

python3 $PEGASUS quantize \
  --model /workspace/out/keras_model \
  --model-data /workspace/out/keras_model.data \
  --with-input-meta /workspace/out/inputmeta_teachable.yml \
  --iterations 10 \
  --device $QUANTIZE_DEVICE \
  --output-dir /workspace/out/quantize_out \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --algorithm normal

python3 $PEGASUS export ovxlib \
  --model /workspace/out/keras_model \
  --model-data /workspace/out/keras_model.data \
  --model-quantize /workspace/out/keras_model.quantize \
  --with-input-meta /workspace/out/inputmeta_teachable.yml \
  --dtype quantized \
  --output-path /workspace/out/build \
  --optimize VIP8000NANONI_PID0XAD \
  --pack-nbg-unify \
  --viv-sdk /opt/acuity/Vivante_IDE/VivanteIDE5.8.1.1/cmdtools
"

if [[ ! -f "$WORK_DIR/out_nbg_unify/network_binary.nb" ]]; then
  echo "network_binary.nb was not generated." >&2
  exit 3
fi

cp "$WORK_DIR/out_nbg_unify/network_binary.nb" "$WORK_DIR/out_nbg_unify/imgclassification.nb"

echo "NB_OK $WORK_DIR/out_nbg_unify/imgclassification.nb"
