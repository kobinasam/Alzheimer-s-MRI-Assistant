#!/usr/bin/env bash
# Serve the fine-tuned model with vLLM (OpenAI-compatible API).
# Run inside your vLLM env.
#
#   bash serve.sh            # merged BF16 model
#   bash serve.sh fp8        # FP8 model from quantize_fp8.py
#   bash serve.sh lora       # base model + adapter, no merge needed
#   TP=2 bash serve.sh       # tensor parallel across 2 GPUs
set -euo pipefail
CFG=${CFG:-config.yaml}
MODE=${1:-merged}
TP=${TP:-1}

y() { python3 -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(eval('c'+sys.argv[1]))" "$1"; }

NAME=$(y "['serve']['served_model_name']")
PORT=$(y "['serve']['port']")
MAXLEN=$(y "['serve']['max_model_len']")
UTIL=$(y "['serve']['gpu_memory_utilization']")

COMMON=(--port "$PORT" --max-model-len "$MAXLEN"
        --gpu-memory-utilization "$UTIL" --tensor-parallel-size "$TP"
        --limit-mm-per-prompt '{"image": 1}')

case "$MODE" in
  merged) exec vllm serve "$(y "['merge']['output_dir']")" --dtype bfloat16 --served-model-name "$NAME" "${COMMON[@]}" ;;
  fp8)    exec vllm serve "$(y "['quantize']['output_dir']")" --served-model-name "$NAME" "${COMMON[@]}" ;;
  lora)
    ADAPTER="$(y "['train']['output_dir']")/final"
    RANK=$(y "['train']['lora_r']")
    # In LoRA mode the request "model" field must be the adapter name ($NAME).
    exec vllm serve "$(y "['model_name']")" --dtype bfloat16 --served-model-name base "${COMMON[@]}" \
      --enable-lora --max-lora-rank "$RANK" --lora-modules "$NAME=$ADAPTER" ;;
  *) echo "mode must be merged | fp8 | lora"; exit 1 ;;
esac
