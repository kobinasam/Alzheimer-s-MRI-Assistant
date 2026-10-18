"""Sanity-check one training batch before spending GPU hours.

Prints the decoded text that carries loss (should be only the assistant answers),
the number of image tokens, and the sequence length.

Usage: python check_batch.py --config config.yaml [--n 2]
"""
import argparse
import os

import torch
import yaml
from transformers import AutoProcessor

from train_lora import Collator, JsonlDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    proc = AutoProcessor.from_pretrained(cfg["model_name"])
    ds = JsonlDataset(os.path.join(cfg["data"]["out_dir"], "train.jsonl"))
    batch = Collator(proc, cfg["image_size"])([ds[i] for i in range(min(args.n, len(ds)))])

    tok = proc.tokenizer
    img_tok = getattr(proc, "image_token", None)
    img_id = tok.convert_tokens_to_ids(img_tok) if img_tok else None
    for i in range(batch["input_ids"].shape[0]):
        ids, lab, mask = batch["input_ids"][i], batch["labels"][i], batch["attention_mask"][i]
        print(f"\n=== sample {i}: {int(mask.sum())} tokens, "
              f"{int((ids == img_id).sum()) if img_id is not None else '?'} image tokens, "
              f"{int((lab != -100).sum())} loss tokens")
        print("--- full input (image tokens collapsed) ---")
        text = tok.decode(ids[mask.bool()], skip_special_tokens=False)
        if img_tok:
            text = text.replace(img_tok, "")
        print(text)
        print("--- tokens that carry loss ---")
        spans, cur = [], []
        for t, l in zip(ids.tolist(), lab.tolist()):
            if l != -100:
                cur.append(t)
            elif cur:
                spans.append(cur)
                cur = []
        if cur:
            spans.append(cur)
        for s in spans:
            print(repr(tok.decode(s)))
        assert not torch.any((lab != -100) & (ids == img_id)) if img_id is not None else True, \
            "image tokens are in the loss"
    print("\nkeys:", list(batch.keys()))


if __name__ == "__main__":
    main()
