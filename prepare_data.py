"""Convert LLaVA-style instruction JSON into chat-format JSONL with a subject-level split.

Input record (LLaVA / LLaVA-Med format):
  {"id": "...", "image": "rel/path.jpg",
   "conversations": [{"from": "human", "value": "<image>\\nQuestion"},
                     {"from": "gpt", "value": "Answer"}, ...]}

Output line:
  {"id": ..., "subject": ..., "image": "/abs/path.jpg",
   "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ...}]},
                {"role": "assistant", "content": [{"type": "text", "text": ...}]}, ...]}

Usage: python prepare_data.py --config config.yaml
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

import yaml
from PIL import Image

ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def convert(rec, image_root, system_prompt):
    img_path = Path(image_root) / rec["image"]
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    image_placed = False
    for turn in rec["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            raise ValueError(f"unknown speaker {turn['from']!r}")
        text = turn["value"].replace("<image>", "").strip()
        content = []
        if role == "user" and not image_placed:
            content.append({"type": "image"})
            image_placed = True
        content.append({"type": "text", "text": text})
        msgs.append({"role": role, "content": content})
    roles = [m["role"] for m in msgs if m["role"] != "system"]
    if not roles or roles[0] != "user" or "assistant" not in roles:
        raise ValueError("conversation must start with user and contain an assistant turn")
    if any(a == b for a, b in zip(roles, roles[1:])):
        raise ValueError("roles must alternate user/assistant")
    return img_path, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check-images", action="store_true",
                    help="open every image to catch corrupt files (slower)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]

    records = json.load(open(d["llava_json"]))
    print(f"loaded {len(records)} records from {d['llava_json']}")

    subj_re = re.compile(d["subject_regex"]) if d.get("subject_regex") else None

    # Optional: map the regex match (e.g. ADNI Image Data ID "I381953") to a subject
    # ID using an ADNI metadata CSV, so all scans of one participant stay together.
    id_to_subject = {}
    if d.get("subject_csv"):
        import csv
        with open(d["subject_csv"], newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = row[d["csv_key_col"]].strip()
                if key and not key.startswith("I") and key.isdigit():
                    key = "I" + key          # some ADNI exports drop the leading I
                id_to_subject[key] = row[d["csv_subject_col"]].strip()
        print(f"loaded {len(id_to_subject)} image->subject mappings from {d['subject_csv']}")
    rows, skipped = [], collections.Counter()
    for i, rec in enumerate(records):
        try:
            img_path, msgs = convert(rec, d["image_root"], d.get("system_prompt"))
        except (KeyError, ValueError) as e:
            skipped[f"bad record: {e}"] += 1
            continue
        if not img_path.is_file():
            skipped["missing image"] += 1
            continue
        if args.check_images:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                skipped["corrupt image"] += 1
                continue
        rid = str(rec.get("id", i))
        subject = rid
        if subj_re:
            m = subj_re.search(rid) or subj_re.search(rec["image"])
            subject = m.group(1) if m else None
            if subject is None:
                skipped["no regex match (kept, own group)"] += 1
                subject = f"_nomatch_{rid}"
            elif id_to_subject:
                if subject in id_to_subject:
                    subject = id_to_subject[subject]
                else:
                    skipped["image id not in subject CSV (grouped by image only)"] += 1
        rows.append({"id": rid, "subject": subject, "image": str(img_path.resolve()), "messages": msgs})

    for k, v in skipped.items():
        print(f"  {k}: {v}")

    # Split whole subjects, never individual slices.
    subjects = sorted({r["subject"] for r in rows})
    random.Random(d["seed"]).shuffle(subjects)
    n_val = round(len(subjects) * d["val_fraction"])
    n_test = round(len(subjects) * d["test_fraction"])
    split_of = {}
    for j, s in enumerate(subjects):
        split_of[s] = "test" if j < n_test else "val" if j < n_test + n_val else "train"

    out = Path(d["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    counts = collections.Counter()
    for r in rows:
        s = split_of[r["subject"]]
        files[s].write(json.dumps(r) + "\n")
        counts[s] += 1
    for f in files.values():
        f.close()

    per = collections.Counter(r["subject"] for r in rows)
    print(f"records per group: min {min(per.values())}, max {max(per.values())}, "
          f"mean {len(rows) / len(per):.1f}")
    print(f"{len(subjects)} groups -> train/val/test records: "
          f"{counts['train']}/{counts['val']}/{counts['test']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
