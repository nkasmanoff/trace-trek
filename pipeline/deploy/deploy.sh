#!/usr/bin/env bash
# Convert a merged HF checkpoint to a quantized GGUF and swap it into
# llama-server. Built for the Qwen3.6-35B-A3B fine-tune: train.py writes the
# merged bf16 checkpoint to outputs/merged/, this converts + quantizes it for
# cheap local inference (CPU / Mac / small GPU). 35B-total MoE at Q4_K_M is
# ~20-22GB resident; only ~3B params activate per token, so decode stays fast.
#
# Usage:
#   ./deploy.sh <merged-checkpoint-dir> <version>     convert + deploy
#   ./deploy.sh --serve-only <version>                (re)serve an existing GGUF
#
# Examples:
#   ./deploy.sh ../outputs/merged v2
#   ./deploy.sh --serve-only v1        # rollback to models/<name>-v1.gguf
#
# Env overrides:
#   MODEL_NAME  artifact basename            (default: qwen3.6-a3b-code)
#   QUANT       llama-quantize type          (default: Q4_K_M; e.g. Q5_K_M, Q6_K)
#   CTX         server context length        (default: 131072)
#   IMATRIX     path to an importance matrix (optional; better low-bit quality)
#   TEMPLATE    chat-template .jinja file    (optional; default: GGUF-embedded)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # trace-trek/
LLAMA="$ROOT/speedy-llama"
MODELS="$ROOT/models"
MODEL_NAME="${MODEL_NAME:-qwen3.6-a3b-code}"
QUANT="${QUANT:-Q4_K_M}"
CTX="${CTX:-131072}"
IMATRIX="${IMATRIX:-}"
TEMPLATE="${TEMPLATE:-}"
PORT=8080

serve() {
  local gguf="$1"
  echo "== stopping current llama-server =="
  pkill -f "llama-server -m" 2>/dev/null || true
  sleep 2
  echo "== starting llama-server with $(basename "$gguf") =="
  # Qwen3.6 thinking-mode sampling (general tasks): temp=1.0, top_p=0.95,
  # top_k=20, min_p=0.0, presence_penalty=1.5. --jinja uses the chat template
  # embedded in the GGUF unless TEMPLATE overrides it.
  local tmpl_args=(--jinja)
  [[ -n "$TEMPLATE" ]] && tmpl_args+=(--chat-template-file "$TEMPLATE")
  nohup "$LLAMA/build/bin/llama-server" \
    -m "$gguf" \
    -c "$CTX" "${tmpl_args[@]}" \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 \
    --host 127.0.0.1 --port "$PORT" \
    > "$MODELS/llama-server.log" 2>&1 &
  echo -n "waiting for health"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo " — ok"
      return 0
    fi
    echo -n "."
    sleep 2
  done
  echo " — FAILED (see $MODELS/llama-server.log)"
  return 1
}

if [[ "${1:-}" == "--serve-only" ]]; then
  VERSION="${2:?usage: deploy.sh --serve-only <version>}"
  GGUF="$MODELS/$MODEL_NAME-$VERSION.gguf"
  [[ -f "$GGUF" ]] || { echo "not found: $GGUF"; exit 1; }
  serve "$GGUF"
  exit 0
fi

CKPT="${1:?usage: deploy.sh <merged-checkpoint-dir> <version>}"
VERSION="${2:?usage: deploy.sh <merged-checkpoint-dir> <version>}"
F16="$MODELS/$MODEL_NAME-$VERSION-f16.gguf"
GGUF="$MODELS/$MODEL_NAME-$VERSION.gguf"

[[ -d "$CKPT" ]] || { echo "checkpoint dir not found: $CKPT"; exit 1; }

# Requires a current llama.cpp build: Qwen3.6's hybrid Gated DeltaNet + MoE arch
# (and its embedded chat template) are only handled by recent convert/runtime.
echo "== converting $CKPT -> f16 GGUF (needs ~70GB free disk) =="
python3 "$LLAMA/convert_hf_to_gguf.py" "$CKPT" \
  --outfile "$F16" --outtype f16

echo "== quantizing -> $QUANT =="
# An imatrix (importance matrix) noticeably improves low-bit quality; pass one
# via IMATRIX=... (build it with llama-imatrix on a calibration set).
if [[ -n "$IMATRIX" ]]; then
  "$LLAMA/build/bin/llama-quantize" --imatrix "$IMATRIX" "$F16" "$GGUF" "$QUANT"
else
  "$LLAMA/build/bin/llama-quantize" "$F16" "$GGUF" "$QUANT"
fi

echo "== removing f16 intermediate =="
rm -f "$F16"

serve "$GGUF"

echo
echo "Deployed $GGUF"
echo "Now run the eval gate:   make -C \"$ROOT/agent-improver\" eval"
echo "Rollback if it regresses: ./deploy.sh --serve-only <previous-version>"
