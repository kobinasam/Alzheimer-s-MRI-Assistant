"""LoRA instruction tuning of a vision-language model on the prepared JSONL.

Loss is computed on assistant tokens only; system, user, image and padding
tokens are masked to -100.

Single GPU:  python train_lora.py --config config.yaml
Multi GPU:   torchrun --nproc_per_node=4 train_lora.py --config config.yaml
"""
import argparse
import json
import os

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_image(path, size):
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


class JsonlDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Builds model inputs plus labels that cover only assistant turns.

    Assistant spans are found by rendering the chat template up to each
    assistant turn (with and without that turn) and tokenizing with the same
    image, so expanded image tokens line up exactly with the full sequence.
    """

    def __init__(self, processor, image_size):
        self.proc = processor
        self.size = image_size
        self.proc.tokenizer.padding_side = "right"

    def _len(self, msgs, img, gen_prompt):
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt)
        return self.proc(text=[text], images=[[img]], return_tensors="pt")["input_ids"].shape[1]

    def __call__(self, batch):
        texts, images = [], []
        for ex in batch:
            img = load_image(ex["image"], self.size)
            images.append([img])
            texts.append(self.proc.apply_chat_template(ex["messages"], tokenize=False,
                                                       add_generation_prompt=False))
        enc = self.proc(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(enc["input_ids"], -100)
        for i, ex in enumerate(batch):
            msgs, img = ex["messages"], images[i][0]
            for k, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                start = self._len(msgs[:k], img, gen_prompt=True)
                end = self._len(msgs[:k + 1], img, gen_prompt=False)
                labels[i, start:end] = enc["input_ids"][i, start:end]
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def drop_long(ds, processor, image_size, max_len, name):
    """Remove samples whose full sequence exceeds max_len (truncation would cut image tokens)."""
    if not ds.rows:
        return ds
    tok = processor.tokenizer
    # Every image is resized to the same size, so it expands to the same number
    # of tokens. Measure that once, then length-check the rest on text alone.
    first = processor.apply_chat_template(ds.rows[0]["messages"], tokenize=False)
    full = processor(text=[first], images=[[load_image(ds.rows[0]["image"], image_size)]],
                     return_tensors="pt")["input_ids"].shape[1]
    extra = full - len(tok(first, add_special_tokens=False)["input_ids"])
    keep = []
    for r in ds.rows:
        text = processor.apply_chat_template(r["messages"], tokenize=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) + extra <= max_len:
            keep.append(r)
    if len(keep) < len(ds.rows) and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[{name}] dropped {len(ds.rows) - len(keep)} samples longer than {max_len} tokens")
    ds.rows = keep
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-train-samples", type=int, default=None, help="quick smoke test")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    t, size = cfg["train"], cfg["image_size"]
    data_dir = cfg["data"]["out_dir"]

    processor = AutoProcessor.from_pretrained(cfg["model_name"])

    load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if t["qlora"]:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        load_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
    model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], **load_kwargs)
    model.config.use_cache = False

    if t["qlora"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t["gradient_checkpointing"])
    elif t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
        target_modules=t["target_regex"], modules_to_save=t["modules_to_save"] or None,
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.print_trainable_parameters()
        n_vis = sum(1 for n, _ in model.named_parameters() if "lora_" in n and ("visual" in n or "vision" in n))
        assert n_vis == 0, "LoRA landed in the vision tower; check target_regex"

    train_ds = JsonlDataset(os.path.join(data_dir, "train.jsonl"))
    val_ds = JsonlDataset(os.path.join(data_dir, "val.jsonl"))
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[:args.max_train_samples]
        val_ds.rows = val_ds.rows[:max(8, args.max_train_samples // 10)]
    elif t.get("max_eval_samples") and len(val_ds.rows) > t["max_eval_samples"]:
        import random
        val_ds.rows = random.Random(0).sample(val_ds.rows, t["max_eval_samples"])
    train_ds = drop_long(train_ds, processor, size, t["max_length"], "train")
    val_ds = drop_long(val_ds, processor, size, t["max_length"], "val")

    targs = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=t["num_workers"],
        ddp_find_unused_parameters=False,
        report_to=t["report_to"],
        gradient_checkpointing=False,  # already enabled on the model above
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(processor, size), processing_class=processor)
    trainer.train()

    final = os.path.join(t["output_dir"], "final")
    trainer.save_model(final)          # adapter weights + adapter_config.json
    processor.save_pretrained(final)
    if trainer.is_world_process_zero():
        print(f"best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"adapter saved to {final}")


if __name__ == "__main__":
    main()
