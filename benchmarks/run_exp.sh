#!/usr/bin/env bash
set -euo pipefail
# Usage: run_exp.sh <label> "<extra_docker_env>" "<extra_sglang_flags>"
LABEL="$1"
EXTRA_ENV="${2:-}"
EXTRA_FLAGS="${3:-}"
CNAME="inkling2-mini-exp"

docker rm -f "$CNAME" >/dev/null 2>&1 || true

echo "============================================================"
echo "EXP: $LABEL"
echo "  EXTRA_ENV:   $EXTRA_ENV"
echo "  EXTRA_FLAGS: $EXTRA_FLAGS"
echo "============================================================"

# shellcheck disable=SC2086
docker run -d \
  --name "$CNAME" \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e MAX_JOBS=4 \
  -e HF_HUB_DISABLE_XET=1 \
  -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 \
  -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \
  -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 \
  -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \
  $EXTRA_ENV \
  lmsysorg/sglang:inkling-2-mini \
  sglang serve \
  --model-path thinkingmachines/Inkling-2-Mini \
  --tp 1 \
  --ep 1 \
  --attention-backend fa4 \
  --dtype bfloat16 \
  --page-size 128 \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 8 \
  --mem-fraction-static 0.45 \
  --cuda-graph-max-bs-decode 8 \
  --grammar-backend xgrammar \
  --enable-multimodal \
  --reasoning-parser inkling \
  --tool-call-parser inkling2 \
  --host 0.0.0.0 \
  --port 8000 \
  $EXTRA_FLAGS >/dev/null

READY=0
for i in $(seq 1 200); do
  if ! docker ps --format "{{.Names}}" | grep -qx "$CNAME"; then
    echo "[$LABEL] CONTAINER EXITED EARLY:"
    docker logs --tail 35 "$CNAME" 2>&1 || true
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    READY=1
    echo "[$LABEL] Ready after $((i*3))s"
    break
  fi
  sleep 3
done

if [ "$READY" -ne 1 ]; then
  echo "[$LABEL] TIMED OUT after 600s"
  docker logs --tail 25 "$CNAME" 2>&1 || true
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  exit 1
fi

python3 "$HOME/recipes/inkling-2-mini-sglang-dgx-spark/benchmarks/bench.py" \
  --label "$LABEL" \
  --temperature 0.0 \
  --out-dir "$HOME/recipes/inkling-2-mini-sglang-dgx-spark/benchmarks/results"

docker rm -f "$CNAME" >/dev/null 2>&1 || true
