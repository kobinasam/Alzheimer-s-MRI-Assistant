# Alzheimer's VLM fine-tuning for vLLM

LoRA instruction tuning of a vision-language model on LLaVA-style JSON with JPG slices, then serving and evaluation with vLLM. vLLM does inference only, so training runs in its own env with transformers + PEFT.

Default model: `Qwen/Qwen3-VL-8B-Instruct`. Change `model_name` in `config.yaml` to switch.

## Picking the model ID

Red Hat publishes quantized copies of open models under huggingface.co/RedHatAI (for example `RedHatAI/Qwen3-VL-32B-Instruct-NVFP4`). Those are for serving. To fine-tune, open the RedHatAI repo, look at "Model tree" or the "Quantized version of ..." line on its card, and put that base repo (for example `Qwen/Qwen3-VL-32B-Instruct`) in `model_name`. After training, `quantize_fp8.py` produces the same kind of FP8 checkpoint Red Hat ships.

Rough GPU needs for LoRA in BF16 with gradient checkpointing, batch 4 at 448 px: 8B fits on one 80 GB card, and with `qlora: true` it fits in 24 GB. A 32B model needs QLoRA on 80 GB or several GPUs.

## Steps

```bash
# 0. Training env (your current vLLM env remains the serving env)
bash setup_envs.sh                     # WITH_QUANT=1 bash setup_envs.sh to add the FP8 env
source ~/alz-train/bin/activate

# 1. Edit config.yaml: llava_json, image_root, out_dir paths

# 2. Convert and split by ADNI subject ID (002_S_0295 style)
python prepare_data.py --config config.yaml --check-images

# 3. Look at one batch: loss should cover only the assistant answers
python check_batch.py --config config.yaml

# 4. Smoke test, then the real run
python train_lora.py --config config.yaml --max-train-samples 64
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_lora.py --config config.yaml

# 5. Merge the adapter into BF16 weights
python merge_lora.py --config config.yaml

# 6. (optional) FP8 for faster serving on L40S/H100
~/alz-quant/bin/python quantize_fp8.py --config config.yaml

# 7. Serve in your vLLM env and evaluate on the held-out subjects
bash serve.sh                          # or: bash serve.sh fp8 | bash serve.sh lora
python evaluate.py --config config.yaml
```

For a baseline, run `bash serve.sh lora`, then score both `--model base` and `--model alz-vlm` against the same server.

## Notes

- The split happens at the subject level. Slices from one participant land in a single split, which keeps the test numbers free of leakage. Records with no ADNI ID in their `id` or `image` path get their own group, and `prepare_data.py` reports how many there were.
- Images are resized to `image_size` (448 by default) in training and evaluation, so both stages see the same input. For Qwen3-VL (16 px patches, 2x2 merge) that comes to 196 image tokens per slice.
- Loss is computed only on assistant turns. `check_batch.py` asserts that no image tokens carry loss.
- Samples longer than `max_length` are dropped instead of truncated, because truncation can cut image tokens and break the forward pass.
- LoRA targets only the language model. The training script asserts that no adapter ends up in the vision tower. To also train the vision-language projector, set `modules_to_save: [merger]`.
- `evaluate.py` sends the first user turn of each test record and extracts CN, MCI, or AD from both the reference and the reply using the regexes in `config.yaml`. It reports accuracy, macro-F1, a confusion count, exact match, and token F1. Predictions are saved per model under `eval.out_dir`.
