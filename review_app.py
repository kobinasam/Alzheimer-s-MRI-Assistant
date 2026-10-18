"""Radiologist review app: validate model answers and the LLM judge.

For each sampled test case the reviewer sees the MRI slice, the question, the
reference answer, and the two model answers as "Answer A" / "Answer B" in a
per-case random order (model names hidden). They score each answer, flag
unsupported claims, and pick the better one. The LLM judge's verdict is hidden
by default so it can't anchor the reviewer.

The Agreement tab compares radiologists with the LLM judge (percent agreement and
Cohen's kappa on the head-to-head, correlation on correctness) and between
radiologists when two have reviewed the same cases.

Needs only files already on disk (no GPU, no vLLM):
  eval/<model_a>/predictions.jsonl, eval/<model_b>/predictions.jsonl,
  eval/judge_<a>_vs_<b>/judgements.jsonl, data/test.jsonl

Usage:
  python review_app.py --config config.yaml                 # 60 cases, questions that don't name the diagnosis first
  python review_app.py --config config.yaml --n 100 --port 7861
Open through the same SSH tunnel as app.py (use the matching port).
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import random
import re
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

CSS = """
.app-header {background: linear-gradient(120deg, #312e81 0%, #4338ca 55%, #0e7490 100%);
  color: #fff; border-radius: 16px; padding: 20px 24px; margin-bottom: 6px;}
.app-header h1 {margin: 0 0 4px 0; font-size: 24px; font-weight: 700; color: #fff;}
.app-header p {margin: 0; opacity: .9; font-size: 14px; color: #fff;}
.box {border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 14px; background: #fff;}
.box h4 {margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: .4px; color: #4338ca;}
.box p {margin: 0; white-space: pre-wrap; line-height: 1.5;}
.progress {font-size: 13px; color: #475569;}
.bar {height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-top: 4px;}
.bar > div {height: 100%; background: #4f46e5;}
.judge {font-size: 13px; color: #334155; background: #f8fafc; border: 1px dashed #cbd5e1;
  border-radius: 10px; padding: 10px 12px;}
footer {display: none !important;}
"""

SCORE_CHOICES = ["1", "2", "3", "4", "5"]


def first_question(rec):
    u = next(m for m in rec["messages"] if m["role"] == "user")
    return " ".join(c["text"] for c in u["content"] if c["type"] == "text")


def box(title, text):
    return f'<div class="box"><h4>{html.escape(title)}</h4><p>{html.escape(text or "")}</p></div>'


def cohen_kappa(pairs):
    """pairs: list of (rater1_label, rater2_label)."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({x for p in pairs for x in p})
    po = sum(a == b for a, b in pairs) / n
    c1 = collections.Counter(a for a, _ in pairs)
    c2 = collections.Counter(b for _, b in pairs)
    pe = sum(c1[l] * c2[l] for l in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-a", default="base")
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--n", type=int, default=60, help="cases to review")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    e = cfg["eval"]
    ma, mb = args.model_a, args.model_b or cfg["serve"]["served_model_name"]
    eval_dir = Path(e["out_dir"])
    review_dir = eval_dir / "radiologist_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    def load(path):
        return {r["id"]: r for r in map(json.loads, open(path))}

    pa, pb = load(eval_dir / ma / "predictions.jsonl"), load(eval_dir / mb / "predictions.jsonl")
    judge_path = eval_dir / f"judge_{ma}_vs_{mb}" / "judgements.jsonl"
    judge = load(judge_path) if judge_path.exists() else {}
    test = {}
    for line in open(Path(cfg["data"]["out_dir"]) / f"{e['split']}.jsonl"):
        r = json.loads(line)
        test[r["id"]] = r

    # Sample cases: questions that don't name the diagnosis first, one question per scan.
    ids = [i for i in pa if i in pb and i in test]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    ids.sort(key=lambda i: judge.get(i, {}).get("label_in_question", False))
    seen_scans, cases = set(), []
    for i in ids:
        if test[i]["image"] in seen_scans:
            continue
        seen_scans.add(test[i]["image"])
        cases.append(i)
        if len(cases) >= args.n:
            break

    def blind(i):
        """Per-case A/B order, stable across sessions and reviewers."""
        flip = int(hashlib.sha1(f"{args.seed}:{i}".encode()).hexdigest(), 16) % 2
        return (mb, ma) if flip else (ma, mb)

    def review_file(name):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "anonymous"
        return review_dir / f"reviews_{safe}.jsonl"

    def load_reviews(name):
        out = {}
        f = review_file(name)
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                out[r["id"]] = r          # last write wins
        return out

    # ---------------------------------------------------------------- rendering
    def render(idx, name):
        idx = max(0, min(idx, len(cases) - 1))
        i = cases[idx]
        rec = test[i]
        A, B = blind(i)
        preds = {ma: pa[i]["prediction"], mb: pb[i]["prediction"]}
        prior = load_reviews(name).get(i, {}) if name.strip() else {}
        done = len(load_reviews(name)) if name.strip() else 0
        pct = 100 * done / len(cases)
        progress = (f'<div class="progress">Case {idx + 1} of {len(cases)} &middot; '
                    f'{done} reviewed by {html.escape(name.strip() or "nobody yet")}'
                    f'<div class="bar"><div style="width:{pct:.0f}%"></div></div></div>')
        j = judge.get(i)
        if j:
            def jfmt(m):
                s = j.get(m) or {}
                return (f"correctness {s.get('correctness', '?')}/5, completeness {s.get('completeness', '?')}/5, "
                        f"unsupported {'yes' if s.get('unsupported') else 'no'}")
            winner = {A: "A", B: "B"}.get(j["pairwise"], "tie")
            judge_html = (f'<div class="judge"><b>LLM judge</b><br>Answer A: {jfmt(A)}<br>'
                          f'Answer B: {jfmt(B)}<br>Head-to-head winner: {winner}</div>')
        else:
            judge_html = '<div class="judge">No LLM judge result for this case.</div>'
        img = Image.open(rec["image"]).convert("RGB")
        get = lambda k, d=None: prior.get(k, d)
        return (idx, progress, img, box("Question", first_question(rec)),
                box("Reference answer", pa[i]["reference"]),
                box("Answer A", preds[A]), box("Answer B", preds[B]),
                get("a_correct"), get("a_complete"), get("a_unsupported", False),
                get("b_correct"), get("b_complete"), get("b_unsupported", False),
                get("preference"), get("comment", ""), judge_html)

    def save(idx, name, ac, acomp, aun, bc, bcomp, bun, pref, comment):
        if not name.strip():
            raise gr.Error("Enter your name or reviewer ID first.")
        missing = [lbl for lbl, v in (("A correctness", ac), ("A completeness", acomp),
                                      ("B correctness", bc), ("B completeness", bcomp),
                                      ("preference", pref)) if not v]
        if missing:
            raise gr.Error("Please fill in: " + ", ".join(missing))
        i = cases[idx]
        A, B = blind(i)
        rec = {"id": i, "reviewer": name.strip(), "time": dt.datetime.now().isoformat(timespec="seconds"),
               "image": test[i]["image"], "answer_a_model": A, "answer_b_model": B,
               "a_correct": ac, "a_complete": acomp, "a_unsupported": bool(aun),
               "b_correct": bc, "b_complete": bcomp, "b_unsupported": bool(bun),
               "preference": pref,
               "preferred_model": {"A": A, "B": B}.get(pref, "tie"),
               "comment": comment.strip()}
        review_dir.mkdir(parents=True, exist_ok=True)
        with open(review_file(name), "a") as f:
            f.write(json.dumps(rec) + "\n")
        gr.Info(f"Saved case {idx + 1}.")
        return render(idx + 1, name)

    def next_unreviewed(idx, name):
        done = load_reviews(name) if name.strip() else {}
        for k in list(range(idx + 1, len(cases))) + list(range(0, idx + 1)):
            if cases[k] not in done:
                return render(k, name)
        gr.Info("All cases reviewed.")
        return render(idx, name)

    # ---------------------------------------------------------------- agreement
    def agreement():
        files = sorted(review_dir.glob("reviews_*.jsonl"))
        by_rev = {}
        for f in files:
            d = {}
            for line in open(f):
                r = json.loads(line)
                d[r["id"]] = r
            if d:
                by_rev[next(iter(d.values()))["reviewer"]] = d
        if not by_rev:
            return "<p>No reviews yet.</p>", None
        rows_html, export_rows = [], []
        to_model = lambda r, side: r[f"answer_{side}_model"]
        for rev, d in by_rev.items():
            recs = list(d.values())
            n = len(recs)
            pref = collections.Counter(r["preferred_model"] for r in recs)
            corr = {m: [] for m in (ma, mb)}
            uns = {m: [] for m in (ma, mb)}
            for r in recs:
                for side in ("a", "b"):
                    corr[to_model(r, side)].append(int(r[f"{side}_correct"]))
                    uns[to_model(r, side)].append(1 if r[f"{side}_unsupported"] else 0)
                export_rows.append(r)
            pairs = [(r["preferred_model"], judge[r["id"]]["pairwise"]) for r in recs if r["id"] in judge]
            agree = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            kappa = cohen_kappa(pairs)
            xs, ys = [], []
            for r in recs:
                j = judge.get(r["id"])
                if not j:
                    continue
                for side in ("a", "b"):
                    m = to_model(r, side)
                    if j.get(m):
                        xs.append(int(r[f"{side}_correct"]))
                        ys.append(float(j[m]["correctness"]))
            rho = pearson(xs, ys)
            mean = lambda v: f"{sum(v) / len(v):.2f}" if v else "n/a"
            pct = lambda v: f"{100 * sum(v) / len(v):.0f}%" if v else "n/a"
            fmt = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"
            rows_html.append(
                f"<tr><td>{html.escape(rev)}</td><td>{n}</td>"
                f"<td>{100 * pref[mb] / n:.0f}% / {100 * pref[ma] / n:.0f}% / {100 * pref['tie'] / n:.0f}%</td>"
                f"<td>{mean(corr[mb])} vs {mean(corr[ma])}</td>"
                f"<td>{pct(uns[mb])} vs {pct(uns[ma])}</td>"
                f"<td>{fmt(agree * 100 if agree is not None else None, 0)}%</td><td>{fmt(kappa)}</td>"
                f"<td>{fmt(rho)}</td></tr>")
        inter = ""
        revs = list(by_rev)
        if len(revs) >= 2:
            parts = []
            for x in range(len(revs)):
                for y in range(x + 1, len(revs)):
                    common = set(by_rev[revs[x]]) & set(by_rev[revs[y]])
                    pairs = [(by_rev[revs[x]][i]["preferred_model"], by_rev[revs[y]][i]["preferred_model"])
                             for i in common]
                    k = cohen_kappa(pairs)
                    parts.append(f"{html.escape(revs[x])} vs {html.escape(revs[y])}: {len(common)} shared cases, "
                                 f"kappa {'n/a' if k is None else f'{k:.2f}'}")
            inter = "<h4>Between radiologists (head-to-head preference)</h4><p>" + "<br>".join(parts) + "</p>"
        table = (
            "<table style='width:100%;border-collapse:collapse' border='1' cellpadding='6'>"
            f"<tr><th>Reviewer</th><th>Cases</th><th>Prefers {html.escape(mb)} / {html.escape(ma)} / tie</th>"
            f"<th>Correctness /5 ({html.escape(mb)} vs {html.escape(ma)})</th>"
            f"<th>Unsupported claims ({html.escape(mb)} vs {html.escape(ma)})</th>"
            "<th>Agrees with LLM judge</th><th>Cohen's kappa vs judge</th>"
            "<th>Correctness correlation vs judge</th></tr>" + "".join(rows_html) + "</table>")
        note = ("<p style='font-size:13px;color:#64748b'>Kappa: below 0.2 slight, 0.2 to 0.4 fair, "
                "0.4 to 0.6 moderate, 0.6 to 0.8 substantial agreement. "
                "Correlation is Pearson r between radiologist and judge correctness scores.</p>")
        out = review_dir / "radiologist_reviews_export.csv"
        keys = ["id", "reviewer", "time", "image", "answer_a_model", "answer_b_model", "a_correct",
                "a_complete", "a_unsupported", "b_correct", "b_complete", "b_unsupported",
                "preference", "preferred_model", "comment"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys + ["judge_pairwise"], extrasaction="ignore")
            w.writeheader()
            for r in export_rows:
                w.writerow({**r, "judge_pairwise": judge.get(r["id"], {}).get("pairwise", "")})
        return table + note + inter, str(out)

    # ---------------------------------------------------------------- layout
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate", radius_size="lg")
    major = int(gr.__version__.split(".")[0])
    blocks_kw = {} if major >= 6 else {"theme": theme, "css": CSS}
    launch_kw = {"theme": theme, "css": CSS} if major >= 6 else {}

    with gr.Blocks(title="Radiologist Review", **blocks_kw) as demo:
        gr.HTML(f"""<div class="app-header"><h1>Radiologist review</h1>
<p>{len(cases)} test cases. Two AI answers per case, shown as A and B in random order. Score each
against the scan and the reference, then pick the better one. Research use only.</p></div>""")
        idx = gr.State(0)
        with gr.Tabs():
            with gr.Tab("Review"):
                with gr.Row():
                    name = gr.Textbox(label="Reviewer name or ID", placeholder="e.g. rad01", scale=3)
                    progress = gr.HTML()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        image = gr.Image(type="pil", label="MRI slice", height=380, interactive=False)
                        question = gr.HTML()
                        reference = gr.HTML()
                        with gr.Accordion("Show LLM judge verdict (after you score)", open=False):
                            judge_box = gr.HTML()
                    with gr.Column(scale=6, min_width=420):
                        with gr.Row(equal_height=False):
                            with gr.Column():
                                ans_a = gr.HTML()
                                a_correct = gr.Radio(SCORE_CHOICES, label="A: correctness (1 wrong, 5 fully right)")
                                a_complete = gr.Radio(SCORE_CHOICES, label="A: completeness")
                                a_unsup = gr.Checkbox(label="A states facts not in question or reference")
                            with gr.Column():
                                ans_b = gr.HTML()
                                b_correct = gr.Radio(SCORE_CHOICES, label="B: correctness (1 wrong, 5 fully right)")
                                b_complete = gr.Radio(SCORE_CHOICES, label="B: completeness")
                                b_unsup = gr.Checkbox(label="B states facts not in question or reference")
                        pref = gr.Radio(["A", "B", "Tie"], label="Which answer is clinically better?")
                        comment = gr.Textbox(label="Comments (optional)", lines=2)
                        with gr.Row():
                            prev_btn = gr.Button("Previous")
                            skip_btn = gr.Button("Next unreviewed")
                            save_btn = gr.Button("Save and next", variant="primary")
            with gr.Tab("Agreement"):
                gr.Markdown("Compares each radiologist with the LLM judge, and radiologists with each other.")
                calc = gr.Button("Compute agreement", variant="primary")
                agree_html = gr.HTML()
                agree_file = gr.File(label="All reviews (CSV)")

        view = [idx, progress, image, question, reference, ans_a, ans_b,
                a_correct, a_complete, a_unsup, b_correct, b_complete, b_unsup, pref, comment, judge_box]
        demo.load(lambda n: render(0, n or ""), [name], view)
        name.submit(lambda i, n: next_unreviewed(-1, n), [idx, name], view)
        name.blur(lambda i, n: render(i, n), [idx, name], view)
        save_btn.click(save, [idx, name, a_correct, a_complete, a_unsup, b_correct, b_complete,
                              b_unsup, pref, comment], view)
        prev_btn.click(lambda i, n: render(i - 1, n), [idx, name], view)
        skip_btn.click(next_unreviewed, [idx, name], view)
        calc.click(agreement, None, [agree_html, agree_file])

    image_roots = {str(Path(test[i]["image"]).parent) for i in cases}
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port,
                        allowed_paths=[str(review_dir.resolve()), *image_roots], **launch_kw)


if __name__ == "__main__":
    main()
