#!/usr/bin/env bash
# Creates the training env. Your existing vLLM env stays as the serving env.
# Usage: bash setup_envs.sh
set -euo pipefail

TRAIN_ENV=${TRAIN_ENV:-$HOME/alz-train}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv "$TRAIN_ENV" --python 3.12
# shellcheck disable=SC1091
source "$TRAIN_ENV/bin/activate"

# torch matched to the driver, then the training stack.
uv pip install torch torchvision --torch-backend=auto
uv pip install "transformers>=4.57" peft accelerate bitsandbytes \
               pillow pyyaml tqdm openai

# Optional: FP8 quantization of the merged model for serving.
# llm-compressor pins its own transformers range, so it gets its own env.
if [[ "${WITH_QUANT:-0}" == "1" ]]; then
  uv venv "$HOME/alz-quant" --python 3.12
  "$HOME/alz-quant/bin/python" -m ensurepip >/dev/null 2>&1 || true
  VIRTUAL_ENV="$HOME/alz-quant" uv pip install torch torchvision --torch-backend=auto
  VIRTUAL_ENV="$HOME/alz-quant" uv pip install llmcompressor pillow pyyaml
fi

python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "peft", peft.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.0f} GiB")
PY
echo "Training env ready: source $TRAIN_ENV/bin/activate"
