"""Optional: FP8-dynamic quantization of the merged model with llm-compressor
(the same recipe Red Hat uses for its RedHatAI/*-FP8-dynamic checkpoints).
Vision tower and lm_head stay in BF16. No calibration data needed.

Run in the alz-quant env: ~/alz-quant/bin/python quantize_fp8.py --config config.yaml
FP8 compute needs Ada or Hopper (L40S, H100). On A100, vLLM runs it weight-only.
"""
import argparse

import torch
import yaml
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    src, out = cfg["merge"]["output_dir"], cfg["quantize"]["output_dir"]

    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(src)
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC",
        ignore=["re:.*lm_head", "re:.*visual.*", "re:.*vision.*"])
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)
    print(f"FP8 model saved to {out}")


if __name__ == "__main__":
    main()
