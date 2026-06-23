#!/usr/bin/env bash
# Convert a merged HF checkpoint to Q4_K_M GGUF and swap it into llama-server.
#
# Usage:
#   ./deploy.sh <merged-checkpoint-dir> <version>     convert + deploy
#   ./deploy.sh --serve-only <version>                (re)serve an existing GGUF
#
# Examples:
#   ./deploy.sh ./checkpoints/v2-merged v2
#   ./deploy.sh --serve-only v1        # rollback to models/north-mini-code-v1.gguf
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # trace-trek/
LLAMA="$ROOT/speedy-llama"
MODELS="$ROOT/models"
TEMPLATE="$MODELS/north-mini-code.jinja"
PORT=8080

serve() {
  local gguf="$1"
  echo "== stopping current llama-server =="
  pkill -f "llama-server -m" 2>/dev/null || true
  sleep 2
  echo "== starting llama-server with $(basename "$gguf") =="
  nohup "$LLAMA/build/bin/llama-server" \
    -m "$gguf" \
    -c 131072 --jinja \
    --chat-template-file "$TEMPLATE" \
    --temp 1.0 --top-p 0.95 \
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
  GGUF="$MODELS/north-mini-code-$VERSION.gguf"
  [[ -f "$GGUF" ]] || { echo "not found: $GGUF"; exit 1; }
  serve "$GGUF"
  exit 0
fi

CKPT="${1:?usage: deploy.sh <merged-checkpoint-dir> <version>}"
VERSION="${2:?usage: deploy.sh <merged-checkpoint-dir> <version>}"
F16="$MODELS/north-mini-code-$VERSION-f16.gguf"
GGUF="$MODELS/north-mini-code-$VERSION.gguf"

[[ -d "$CKPT" ]] || { echo "checkpoint dir not found: $CKPT"; exit 1; }

echo "== converting $CKPT -> f16 GGUF (needs ~60GB free disk) =="
python3 "$LLAMA/convert_hf_to_gguf.py" "$CKPT" \
  --outfile "$F16" --outtype f16

echo "== quantizing -> Q4_K_M =="
"$LLAMA/build/bin/llama-quantize" "$F16" "$GGUF" Q4_K_M

echo "== removing f16 intermediate =="
rm -f "$F16"

serve "$GGUF"

echo
echo "Deployed $GGUF"
echo "Now run the eval gate:   make -C \"$ROOT/agent-improver\" eval"
echo "Rollback if it regresses: ./deploy.sh --serve-only <previous-version>"
