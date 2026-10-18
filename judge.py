"""LLM-as-a-judge validation of base vs fine-tuned answers.

Reads the predictions written by evaluate.py for two models and asks a judge LLM
(text only, it never sees the image) to:
  1. score each answer against the reference (correctness 1-5, completeness 1-5,
     unsupported-claim flag 0/1), and
  2. compare the two answers blind, head to head, twice with A/B order swapped.
     An answer wins only if it wins in both orders; otherwise it is a tie. This
     cancels the judge's position bias.

The judge runs on your own vLLM server by default, so ADNI-derived text never
leaves the machine.

Usage:
  python judge.py --config config.yaml                       # base vs alz-vlm, judge = "base"
  python judge.py --config config.yaml --limit 300           # quick run
  python judge.py --config config.yaml --judge-model <name> --judge-url http://host:port/v1
"""
import argparse
import collections
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm

SYSTEM = ("You are an expert neuroradiologist grading answers written by AI assistants about a "
          "single brain MRI slice from an Alzheimer's disease study. You cannot see the image. "
          "Grade only against the expert reference answer and the question. Be strict and consistent.")

POINTWISE = """Question asked about the MRI:
{question}

Expert reference answer:
{reference}

Candidate answer:
{candidate}

Grade the candidate:
- correctness (1-5): agreement with the reference's main clinical conclusions (diagnosis or ranking of differentials, key imaging findings, expected trajectory). 5 = same conclusions, 3 = partly right, 1 = wrong or contradicts the reference.
- completeness (1-5): how many of the reference's key points the candidate covers.
- unsupported (0 or 1): 1 if the candidate states specific patient facts (test scores, age, genotype, measurements, visit details) that appear in neither the question nor the reference.

Return only JSON: {{"correctness": <int>, "completeness": <int>, "unsupported": <0 or 1>, "reason": "<one sentence>"}}"""

PAIRWISE = """Question asked about the MRI:
{question}

Expert reference answer:
{reference}

Answer A:
{a}

Answer B:
{b}

Which answer agrees better with the reference's clinical conclusions and makes fewer unsupported claims? Ignore length and style. Answer "tie" if they are equally good.

Return only JSON: {{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def first_question(rec):
    u = next(m for m in rec["messages"] if m["role"] == "user")
    return " ".join(c["text"] for c in u["content"] if c["type"] == "text")


def labels_named(text, patterns):
    return {k for k, ps in patterns.items() if any(re.search(p, text, flags=re.I) for p in ps)}


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-a", default="base", help="first model's predictions folder name")
    ap.add_argument("--model-b", default=None, help="second model, defaults to serve.served_model_name")
    ap.add_argument("--judge-model", default="base", help="served model used as the judge")
    ap.add_argument("--judge-url", default=None, help="defaults to eval.base_url")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    e = cfg["eval"]
    model_b = args.model_b or cfg["serve"]["served_model_name"]
    client = OpenAI(base_url=args.judge_url or e["base_url"], api_key="EMPTY")

    def load(name):
        path = Path(e["out_dir"]) / name / "predictions.jsonl"
        return {r["id"]: r for r in map(json.loads, open(path))}

    preds = {args.model_a: load(args.model_a), model_b: load(model_b)}
    test = {}
    for line in open(Path(cfg["data"]["out_dir"]) / f"{e['split']}.jsonl"):
        r = json.loads(line)
        test[r["id"]] = r
    ids = [i for i in preds[args.model_a] if i in preds[model_b] and i in test][: args.limit]
    pats = e["label_patterns"]
    print(f"judging {len(ids)} questions: {args.model_a} vs {model_b}, judge = {args.judge_model}")

    def ask(prompt):
        for _ in range(3):
            r = client.chat.completions.create(
                model=args.judge_model, temperature=0.0, max_tokens=200,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            out = parse_json(r.choices[0].message.content)
            if out is not None:
                return out
        return None

    def judge_one(i):
        q, ref = first_question(test[i]), preds[args.model_a][i]["reference"]
        a, b = preds[args.model_a][i]["prediction"], preds[model_b][i]["prediction"]
        res = {"id": i, "question": q,
               "label_in_question": bool(preds[args.model_a][i].get("ref_label"))
               and preds[args.model_a][i]["ref_label"] in labels_named(q, pats)}
        for name, cand in ((args.model_a, a), (model_b, b)):
            s = ask(POINTWISE.format(question=q, reference=ref, candidate=cand))
            res[name] = s if s and all(k in s for k in ("correctness", "completeness", "unsupported")) else None
        p1 = ask(PAIRWISE.format(question=q, reference=ref, a=a, b=b))   # A = model_a
        p2 = ask(PAIRWISE.format(question=q, reference=ref, a=b, b=a))   # A = model_b
        w1 = {"A": args.model_a, "B": model_b}.get((p1 or {}).get("winner"), "tie")
        w2 = {"A": model_b, "B": args.model_a}.get((p2 or {}).get("winner"), "tie")
        res["pairwise"] = w1 if w1 == w2 else "tie"
        res["pairwise_raw"] = [p1, p2]
        return res

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(tqdm(pool.map(judge_one, ids), total=len(ids), desc="judging"))

    def summarize(rows):
        out = {"n": len(rows)}
        for name in (args.model_a, model_b):
            ok = [r[name] for r in rows if r[name]]
            for k in ("correctness", "completeness", "unsupported"):
                m, ci = mean_ci([float(s[k]) for s in ok])
                out[f"{name}.{k}"] = {"mean": m, "ci95": ci, "n": len(ok)}
        c = collections.Counter(r["pairwise"] for r in rows)
        n = len(rows)
        for k in (model_b, args.model_a, "tie"):
            p = c[k] / n if n else 0.0
            out[f"pairwise.{k}_wins" if k != "tie" else "pairwise.ties"] = {
                "rate": p, "ci95": 1.96 * math.sqrt(p * (1 - p) / n) if n else None, "count": c[k]}
        return out

    summary = {
        "judge_model": args.judge_model, "model_a": args.model_a, "model_b": model_b,
        "all": summarize(results),
        "label_not_in_question": summarize([r for r in results if not r["label_in_question"]]),
    }

    out = Path(e["out_dir"]) / f"judge_{args.model_a}_vs_{model_b}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "judgements.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)

    for block in ("all", "label_not_in_question"):
        s = summary[block]
        print(f"\n== {block} (n={s['n']}) ==")
        print(f"{'':14s}{args.model_a:>18s}{model_b:>18s}")
        for k, label in (("correctness", "correctness/5"), ("completeness", "completeness/5"),
                         ("unsupported", "unsupported %")):
            vals = []
            for name in (args.model_a, model_b):
                m, ci = s[f"{name}.{k}"]["mean"], s[f"{name}.{k}"]["ci95"]
                scale = 100 if k == "unsupported" else 1
                vals.append("n/a" if m is None else f"{m * scale:.2f} ± {ci * scale:.2f}")
            print(f"{label:14s}{vals[0]:>18s}{vals[1]:>18s}")
        w = s[f"pairwise.{model_b}_wins"]["rate"]
        l = s[f"pairwise.{args.model_a}_wins"]["rate"]
        t = s["pairwise.ties"]["rate"]
        print(f"head-to-head: {model_b} wins {w:.1%}, {args.model_a} wins {l:.1%}, ties {t:.1%}")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
