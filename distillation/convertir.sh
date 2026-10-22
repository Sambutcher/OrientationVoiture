#!/usr/bin/env bash
# Étape 4 : élève ONNX (NCHW) → TensorFlow.js (NHWC, float16) dans models/orientation/.
# Usage : distillation/convertir.sh <eleve.onnx>
set -euo pipefail
RACINE="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$(realpath "$1")"
TRAVAIL="$(mktemp -d)"
trap 'rm -rf "$TRAVAIL"' EXIT
cd "$TRAVAIL"

onnx2tf -i "$ONNX" -o saved_model -ois image:1,3,128,128 -osd

# tensorflowjs importe tensorflow_decision_forests sans s'en servir, et celui-ci casse protobuf :
# module vide à la place si l'import échoue (même astuce que Street Biathlon).
if ! python3 -c 'import tensorflowjs' 2>/dev/null; then
  python3 -m pip uninstall -y tensorflow-decision-forests >/dev/null 2>&1 || true
  SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$SITE/tensorflow_decision_forests"
  touch "$SITE/tensorflow_decision_forests/__init__.py"
fi
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model \
  --quantize_float16='*' saved_model tfjs

rm -rf "$RACINE/models/orientation"
cp -r tfjs "$RACINE/models/orientation"
ls -la "$RACINE/models/orientation"
