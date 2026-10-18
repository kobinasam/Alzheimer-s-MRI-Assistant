"""Merge the LoRA adapter into the BF16 base model and save a standalone checkpoint for vLLM.

Usage: python merge_lora.py --config config.yaml [--adapter path/to/adapter]
"""
import argparse
import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--adapter", default=None, help="defaults to <train.output_dir>/final")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    adapter = args.adapter or os.path.join(cfg["train"]["output_dir"], "final")
    out = cfg["merge"]["output_dir"]

    # Always merge into full-precision weights, even if training used QLoRA.
    base = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

    AutoProcessor.from_pretrained(cfg["model_name"]).save_pretrained(out)
    try:
        GenerationConfig.from_pretrained(cfg["model_name"]).save_pretrained(out)
    except OSError:
        pass
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
