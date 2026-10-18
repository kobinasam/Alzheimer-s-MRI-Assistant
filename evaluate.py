"""Score the served model on a held-out split through vLLM's OpenAI-compatible API.

For each record it sends the first user turn (with the image) and compares the
reply to the first assistant turn. Reports diagnostic-label accuracy and
macro-F1 (labels pulled from text with config regexes), plus exact match and
token-level F1 for free-text answers.

Usage:
  python evaluate.py --config config.yaml                 # fine-tuned model
  python evaluate.py --config config.yaml --model base    # base model in LoRA mode, as a baseline
"""
import argparse
import base64
import collections
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


def image_url(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def text_of(msg):
    return " ".join(c["text"] for c in msg["content"] if c["type"] == "text")


def build_request(rec, size):
    msgs, ref = [], None
    for m in rec["messages"]:
        if m["role"] == "assistant":
            ref = text_of(m)
            break
        if m["role"] == "system":
            msgs.append({"role": "system", "content": text_of(m)})
        else:
            content = []
            if any(c["type"] == "image" for c in m["content"]):
                content.append({"type": "image_url", "image_url": {"url": image_url(rec["image"], size)}})
            content.append({"type": "text", "text": text_of(m)})
            msgs.append({"role": "user", "content": content})
    return msgs, ref


def extract_label(text, patterns):
    for label, pats in patterns.items():
        if any(re.search(p, text, flags=re.I) for p in pats):
            return label
    return None


def norm_tokens(s):
    return re.findall(r"\w+", s.lower())


def token_f1(pred, ref):
    p, r = norm_tokens(pred), norm_tokens(ref)
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="served model name, defaults to serve.served_model_name")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    e = cfg["eval"]
    model = args.model or cfg["serve"]["served_model_name"]
    client = OpenAI(base_url=e["base_url"], api_key="EMPTY")

    path = Path(cfg["data"]["out_dir"]) / f"{e['split']}.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()][: args.limit]

    def run(rec):
        msgs, ref = build_request(rec, cfg["image_size"])
        r = client.chat.completions.create(model=model, messages=msgs,
                                           max_tokens=e["max_tokens"], temperature=0.0)
        return {"id": rec["id"], "subject": rec["subject"], "reference": ref,
                "prediction": r.choices[0].message.content or ""}

    with ThreadPoolExecutor(e["concurrency"]) as pool:
        results = list(tqdm(pool.map(run, rows), total=len(rows), desc=f"eval {model}"))

    pats = e["label_patterns"]
    labels = list(pats)
    conf = collections.Counter()
    n_lab = correct = 0
    em = f1 = 0.0
    for r in results:
        r["ref_label"] = extract_label(r["reference"], pats)
        r["pred_label"] = extract_label(r["prediction"], pats)
        em += norm_tokens(r["prediction"]) == norm_tokens(r["reference"])
        f1 += token_f1(r["prediction"], r["reference"])
        if r["ref_label"]:
            n_lab += 1
            correct += r["ref_label"] == r["pred_label"]
            conf[(r["ref_label"], r["pred_label"])] += 1

    per_class = {}
    for c in labels:
        tp = conf[(c, c)]
        fp = sum(v for (t, p), v in conf.items() if p == c and t != c)
        fn = sum(v for (t, p), v in conf.items() if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {"precision": prec, "recall": rec,
                        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}

    n = len(results)
    metrics = {
        "model": model, "split": e["split"], "n": n,
        "exact_match": em / n if n else 0.0,
        "token_f1": f1 / n if n else 0.0,
        "label_n": n_lab,
        "label_accuracy": correct / n_lab if n_lab else None,
        "label_macro_f1": (sum(v["f1"] for v in per_class.values() if v["support"]) /
                           max(1, sum(1 for v in per_class.values() if v["support"]))),
        "per_class": per_class,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(conf.items(), key=str)},
    }

    out = Path(e["out_dir"]) / model
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion")}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
