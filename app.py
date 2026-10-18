"""Gradio demo for the fine-tuned Alzheimer's VLM.

Upload a brain MRI slice, ask a question, and read the answer as it streams in.
Non-MRI uploads are rejected by a two-stage check (pixel heuristics, then the
model itself is asked whether the image is a brain MRI). Every inference in the
session is logged and can be downloaded as one .zip (JSONL, CSV, HTML report
and the images).

Backends:
  vllm   talks to a running `bash serve.sh` server (fast; use after training)
  local  loads the base model + a LoRA checkpoint with transformers (no server;
         good for testing intermediate checkpoints)

Examples:
  python app.py --config config.yaml
  python app.py --config config.yaml --backend local \
      --adapter /home/ubuntu/alz-vlm/checkpoints/lora/checkpoint-1000

Open from your laptop through an SSH tunnel (never share=True with ADNI data):
  ssh -L 7860:localhost:7860 ubuntu@<brev-host>   then browse http://localhost:7860
"""
import argparse
import base64
import contextlib
import csv
import datetime as dt
import hashlib
import html
import io
import json
import os
import tempfile
import threading
import time
import zipfile

import gradio as gr
import numpy as np
import yaml
from PIL import Image

GATE_PROMPT = ("Look at this image. Is it a brain MRI scan (any sequence such as T1, T2 or FLAIR, "
               "any plane: axial, sagittal or coronal)? Answer with exactly one word: yes or no.")

EXAMPLE_PROMPTS = [
    "What are the key imaging findings in this brain MRI?",
    "Is there evidence of hippocampal or medial temporal lobe atrophy?",
    "What are the top differentials for cognitive impairment based on this scan?",
    "Given a 74-year-old female with MMSE 24 and CDR-SB 2.5, how would you interpret this scan?",
    "What follow-up imaging or biomarkers would you recommend?",
]

REJECT_MSG = ("Please upload a brain MRI slice. This demo only works on brain MRI images "
              "(T1, T2 or FLAIR; axial, sagittal or coronal).")


# ----------------------------------------------------------------------------- image checks

def prep_image(img, size):
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def heuristic_check(img):
    """Fast pre-filter for obvious non-MRI uploads. Returns (ok, reason)."""
    w, h = img.size
    if min(w, h) < 32:
        return False, "The image is too small to be an MRI slice."
    if max(w, h) / min(w, h) > 3:
        return False, "The image is very elongated; MRI slices are close to square."
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    chroma = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
              + np.abs(a[..., 0] - a[..., 2])).mean() / 3
    if chroma > 10:
        return False, "This looks like a colour image. MRI slices are grayscale."
    if a.std() < 4:
        return False, "The image is almost blank."
    return True, ""


def white_background(img):
    """Display only: turn the black area around the head white. Fills the dark region
    connected to the image border, so dark structures inside the brain stay dark."""
    from PIL import ImageDraw
    orig = img.convert("L")
    g = orig.copy()
    w, h = g.size
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for xy in seeds:
        if g.getpixel(xy) < 40:
            ImageDraw.floodfill(g, xy, 255, thresh=30)
    bg = (np.asarray(g) == 255) & (np.asarray(orig) < 70)
    out = np.asarray(img.convert("RGB")).copy()
    out[bg] = 255
    return Image.fromarray(out)


def viewer_html(img, path):
    shown = white_background(img)
    shown.thumbnail((640, 640))
    buf = io.BytesIO()
    shown.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    name = html.escape(os.path.basename(path)) if path else ""
    return (f'<div class="preview"><img src="{uri}" alt="Uploaded MRI slice">'
            f'<div class="preview-name">{name}</div></div>')


def image_hash(img):
    return hashlib.sha1(img.convert("RGB").tobytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- chat helpers

def as_text(content):
    """Gradio 6 returns chat content as a list of parts; older versions as a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def build_messages(history, image, system_prompt):
    """history: list of {"role", "content"}. The image goes on the first user turn."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "text": system_prompt})
    first_user = True
    for m in history:
        text = as_text(m["content"])
        msgs.append({"role": m["role"], "text": text,
                     "image": image if (m["role"] == "user" and first_user) else None})
        if m["role"] == "user":
            first_user = False
    return msgs


# ----------------------------------------------------------------------------- backends

class VLLMBackend:
    def __init__(self, base_url, size):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.size = size
        self.models = [m.id for m in self.client.models.list().data]

    def _url(self, img):
        buf = io.BytesIO()
        prep_image(img, self.size).save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _convert(self, msgs):
        out = []
        for m in msgs:
            if m["role"] == "system":
                out.append({"role": "system", "content": m["text"]})
            elif m.get("image") is not None:
                out.append({"role": m["role"], "content": [
                    {"type": "image_url", "image_url": {"url": self._url(m["image"])}},
                    {"type": "text", "text": m["text"]}]})
            else:
                out.append({"role": m["role"], "content": m["text"]})
        return out

    def stream(self, msgs, model, max_tokens, temperature):
        r = self.client.chat.completions.create(model=model, messages=self._convert(msgs),
                                                max_tokens=max_tokens, temperature=temperature,
                                                stream=True)
        text = ""
        for chunk in r:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                yield text
        if not text:
            yield ""


class LocalBackend:
    def __init__(self, model_name, adapter, size):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.size = torch, size
        self.proc = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        self.has_adapter = bool(adapter)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.lock = threading.Lock()   # one generation at a time on the GPU
        self.models = ["fine-tuned", "base"] if adapter else ["base"]

    def stream(self, msgs, model, max_tokens, temperature):
        from transformers import TextIteratorStreamer
        chat, images = [], []
        for m in msgs:
            content = []
            if m.get("image") is not None:
                content.append({"type": "image"})
                images.append(prep_image(m["image"], self.size))
            content.append({"type": "text", "text": m["text"]})
            chat.append({"role": m["role"], "content": content})
        text = self.proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[images] if images else None,
                           return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.proc.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True, timeout=600)
        gen = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, streamer=streamer)
        if temperature > 0:
            gen["temperature"] = temperature
        err = []

        def run():
            try:
                ctx = (self.model.disable_adapter() if self.has_adapter and model == "base"
                       else contextlib.nullcontext())
                with ctx, self.torch.inference_mode():
                    self.model.generate(**inputs, **gen)
            except Exception as e:   # surface errors instead of hanging the stream
                err.append(e)
                streamer.end()

        with self.lock:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            out = ""
            for piece in streamer:
                out += piece
                yield out
            t.join()
        if err:
            raise err[0]
        if not out:
            yield ""


def complete(backend, msgs, model, max_tokens, temperature):
    text = ""
    for text in backend.stream(msgs, model, max_tokens, temperature):
        pass
    return text


# ----------------------------------------------------------------------------- session log

class Session:
    """Everything inferred in one browser session, exportable as a zip."""

    def __init__(self, root):
        self.dir = tempfile.mkdtemp(prefix="alzvlm_session_", dir=root)
        os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
        self.records = []

    def save_image(self, img):
        name = f"images/{image_hash(img)}.jpg"
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            img.convert("RGB").save(path, quality=95)
        return name

    def add(self, **rec):
        rec = {"id": len(self.records) + 1,
               "time": dt.datetime.now().isoformat(timespec="seconds"), **rec, "feedback": ""}
        self.records.append(rec)
        return rec["id"]

    def table(self):
        rows = []
        for r in self.records:
            resp = r["response"].replace("\n", " ")
            rows.append([r["id"], r["time"][11:], r["mode"], r["model"], r["prompt"],
                         resp[:160] + ("..." if len(resp) > 160 else ""), r["feedback"]])
        return rows

    def export(self):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.dir, "inferences.jsonl"), "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        keys = ["id", "time", "mode", "model", "image", "turn", "prompt", "response",
                "latency_s", "temperature", "max_tokens", "feedback", "compare_id"]
        with open(os.path.join(self.dir, "inferences.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in self.records:
                w.writerow({k: r.get(k, "") for k in keys})
        with open(os.path.join(self.dir, "report.html"), "w") as f:
            f.write(self._report())
        zpath = os.path.join(self.dir, f"alzvlm_inferences_{stamp}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("inferences.jsonl", "inferences.csv", "report.html"):
                z.write(os.path.join(self.dir, name), name)
            for img in sorted({r["image"] for r in self.records}):
                z.write(os.path.join(self.dir, img), img)
        return zpath

    def _report(self):
        rows = []
        for r in self.records:
            rows.append(
                f"<tr><td>{r['id']}</td><td>{r['time']}</td>"
                f"<td><img src='{r['image']}' width='120'></td>"
                f"<td>{html.escape(r['mode'])}<br><b>{html.escape(r['model'])}</b></td>"
                f"<td>{html.escape(r['prompt'])}</td>"
                f"<td style='white-space:pre-wrap'>{html.escape(r['response'])}</td>"
                f"<td>{r['latency_s']}</td><td>{html.escape(r['feedback'])}</td></tr>")
        return ("<!doctype html><meta charset='utf-8'><title>Inference report</title>"
                "<style>body{font-family:system-ui,sans-serif;margin:24px}"
                "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;"
                "padding:8px;vertical-align:top;font-size:14px}th{background:#f3f4f6}</style>"
                f"<h2>Alzheimer's VLM inference report</h2><p>{len(self.records)} inferences. "
                "Research use only, not for clinical decisions.</p>"
                "<table><tr><th>#</th><th>Time</th><th>Image</th><th>Mode / model</th>"
                "<th>Prompt</th><th>Response</th><th>Latency (s)</th><th>Feedback</th></tr>"
                + "".join(rows) + "</table>")


# ----------------------------------------------------------------------------- UI

CSS = """
.app-header {background: linear-gradient(120deg, #312e81 0%, #4338ca 55%, #0e7490 100%);
  color: #fff; border-radius: 16px; padding: 22px 26px; margin-bottom: 6px;}
.app-header h1 {margin: 0 0 4px 0; font-size: 26px; font-weight: 700; color: #fff;}
.app-header p {margin: 0; opacity: .9; font-size: 14px; color: #fff;}
.app-header .tag {display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 999px;
  background: rgba(255,255,255,.18); font-size: 12px; font-weight: 600; letter-spacing: .3px;
  color: #fff !important;}
.pill {border-radius: 12px; padding: 10px 14px; font-size: 14px; font-weight: 500; line-height: 1.4;}
.pill-idle {background: #f1f5f9; color: #475569; border: 1px dashed #cbd5e1;}
.pill-checking {background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;}
.pill-ok {background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;}
.pill-bad {background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca;}
.pill-warn {background: #fffbeb; color: #b45309; border: 1px solid #fde68a;}
.counter {font-size: 13px; color: #64748b;}
.answer-card {border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 14px; min-height: 220px;}
.answer-title {font-weight: 600; font-size: 14px; margin-bottom: 6px;}
footer {display: none !important;}
.step {font-size: 13px; font-weight: 600; color: #4338ca; letter-spacing: .4px; text-transform: uppercase;
  margin: 4px 0 2px 2px;}
.preview {text-align: center; margin-top: 8px;}
.preview img {display: block; margin: 0 auto; max-width: 100%; max-height: 340px;}
.preview-name {font-size: 12.5px; color: #64748b; margin-top: 6px;}
"""

HEADER = """
<div class="app-header">
  <h1>Alzheimer's MRI Assistant</h1>
  <p>Upload a brain MRI slice and ask about findings, differentials or disease trajectory.
     Answers stream in as the model writes them.</p>
  <span class="tag">RESEARCH USE ONLY &middot; NOT FOR CLINICAL DECISIONS</span>
</div>
"""


def pill(kind, text):
    icon = {"idle": "&#9675;", "checking": "&#8987;", "ok": "&#10003;",
            "bad": "&#10007;", "warn": "&#9888;"}[kind]
    return f'<div class="pill pill-{kind}">{icon}&nbsp; {text}</div>'


def make_chatbot():
    kw = dict(label="Conversation", height=470, show_label=False,
              placeholder="Upload a brain MRI slice on the left, then ask a question below.")
    for extra in ({"type": "messages"}, {}):
        try:
            return gr.Chatbot(**kw, **extra)
        except TypeError:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backend", choices=["vllm", "local"], default="vllm")
    ap.add_argument("--adapter", default=None, help="LoRA dir for --backend local")
    ap.add_argument("--image-check", choices=["full", "heuristic", "off"], default="full",
                    help="full = pixel heuristics + model confirmation")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    size = cfg["image_size"]
    system_prompt = cfg["data"].get("system_prompt")
    export_root = os.path.join(cfg["eval"]["out_dir"], "demo_sessions")
    os.makedirs(export_root, exist_ok=True)

    if args.backend == "vllm":
        be = VLLMBackend(cfg["eval"]["base_url"], size)
        default = cfg["serve"]["served_model_name"]
        default = default if default in be.models else be.models[0]
    else:
        be = LocalBackend(cfg["model_name"], args.adapter, size)
        default = be.models[0]
    can_compare = len(be.models) >= 2
    tuned = default if default != "base" else next((m for m in be.models if m != "base"), default)
    # The base model judges "is this a brain MRI?"; the fine-tuned one has only seen MRIs.
    gate_model = "base" if "base" in be.models else default
    gate_cache = {}
    last_warn = {"key": None, "t": 0.0}

    def warn_once(image):
        # Gradio can fire two change events per upload; show the toast once.
        key = image_hash(image)
        if key != last_warn["key"] or time.time() - last_warn["t"] > 3:
            gr.Warning(REJECT_MSG)
        last_warn.update(key=key, t=time.time())

    def sess_of(s):
        return s if s is not None else Session(export_root)

    def summary(s):
        n = len(s.records) if s else 0
        return f'<div class="counter">{n} inference{"s" if n != 1 else ""} saved in this session</div>'

    def download_update(s):
        if s is None or not s.records:
            return gr.update(value=None, interactive=False)
        return gr.update(value=s.export(), interactive=True)

    # -- image check ----------------------------------------------------------------
    def check_image(image):
        if image is None:
            yield pill("idle", "Upload a brain MRI slice to begin."), {"ok": False, "reason": ""}, []
            return
        if args.image_check == "off":
            yield pill("ok", "Image loaded (checks disabled)."), {"ok": True}, []
            return
        ok, reason = heuristic_check(image)
        if not ok:
            warn_once(image)
            yield pill("bad", f"Not a brain MRI. {reason} {REJECT_MSG}"), {"ok": False, "reason": reason}, []
            return
        if args.image_check == "heuristic":
            yield pill("ok", "Grayscale scan detected. Ready for questions."), {"ok": True}, []
            return
        key = image_hash(image)
        if key not in gate_cache:
            yield pill("checking", "Checking that this is a brain MRI..."), {"ok": False, "reason": "checking"}, []
            try:
                ans = complete(be, [{"role": "user", "text": GATE_PROMPT, "image": image}],
                               gate_model, 4, 0.0)
                gate_cache[key] = ans.strip().lower().startswith("yes")
            except Exception as e:
                yield (pill("warn", f"Couldn't verify the image ({e.__class__.__name__}). "
                                    "You can still ask, but make sure it is a brain MRI."),
                       {"ok": True}, [])
                return
        if gate_cache[key]:
            yield pill("ok", "Brain MRI confirmed. Ask a question or pick an example."), {"ok": True}, []
        else:
            warn_once(image)
            yield pill("bad", f"This doesn't look like a brain MRI. {REJECT_MSG}"), {"ok": False, "reason": "gate"}, []

    def on_upload(path):
        if isinstance(path, (list, tuple)):
            path = path[0] if path else None
        path = getattr(path, "name", path)
        btn = gr.update(label="Upload a different image")
        if not path:
            yield (pill("idle", "Upload a brain MRI slice to begin."), {"ok": False, "reason": ""},
                   [], None, gr.update(), "")
            return
        try:
            image = Image.open(path)
            image.load()
        except Exception:
            yield (pill("bad", "That file couldn't be opened as an image. Upload a JPG or PNG "
                               "brain MRI slice."), {"ok": False, "reason": "unreadable"},
                   [], None, btn, "")
            return
        view = viewer_html(image, path)
        for st, verdict, chat in check_image(image):
            yield st, verdict, chat, image, btn, view

    def require_mri(image, verdict):
        if image is None:
            raise gr.Error("Upload a brain MRI slice first.")
        if not (verdict or {}).get("ok"):
            if (verdict or {}).get("reason") == "checking":
                raise gr.Error("Still checking the image, try again in a moment.")
            raise gr.Error(REJECT_MSG)

    # -- chat -----------------------------------------------------------------------
    def respond(image, prompt, history, model, max_tokens, temperature, verdict, s, chatmap):
        require_mri(image, verdict)
        if not prompt or not prompt.strip():
            raise gr.Error("Type a question first.")
        s = sess_of(s)
        history = [{"role": m["role"], "content": as_text(m["content"])} for m in (history or [])]
        history.append({"role": "user", "content": prompt.strip()})
        msgs = build_messages(history, image, system_prompt)
        history.append({"role": "assistant", "content": ""})
        skip = gr.update()
        yield history, "", s, skip, skip, skip, chatmap
        t0 = time.time()
        try:
            for text in be.stream(msgs, model, int(max_tokens), float(temperature)):
                history[-1]["content"] = text
                yield history, "", s, skip, skip, skip, chatmap
        except Exception as e:
            history[-1]["content"] = f"Generation failed: {e}"
            yield history, prompt, s, skip, skip, skip, chatmap
            return
        rid = s.add(mode="chat", model=model, image=s.save_image(image),
                    turn=sum(m["role"] == "user" for m in history), prompt=prompt.strip(),
                    response=history[-1]["content"], latency_s=round(time.time() - t0, 2),
                    temperature=float(temperature), max_tokens=int(max_tokens), compare_id="")
        chatmap = dict(chatmap or {})
        chatmap[str(len(history) - 1)] = rid
        yield history, "", s, download_update(s), summary(s), s.table(), chatmap

    def on_like(s, chatmap, evt: gr.LikeData):
        if s is None:
            return s, gr.update(), gr.update()
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        rid = (chatmap or {}).get(str(idx))
        for r in s.records:
            if r["id"] == rid:
                r["feedback"] = "up" if evt.liked is True else "down" if evt.liked is False else ""
        return s, s.table(), download_update(s)

    # -- compare --------------------------------------------------------------------
    def compare(image, prompt, max_tokens, temperature, verdict, s):
        require_mri(image, verdict)
        if not prompt or not prompt.strip():
            raise gr.Error("Type a question first.")
        s = sess_of(s)
        msgs = build_messages([{"role": "user", "content": prompt.strip()}], image, system_prompt)
        img_name = s.save_image(image)
        cid = f"c{len(s.records) + 1}"
        outs = {"base": "", tuned: ""}
        skip = gr.update()
        for name in ("base", tuned):
            t0 = time.time()
            try:
                for text in be.stream(msgs, name, int(max_tokens), float(temperature)):
                    outs[name] = text
                    yield outs["base"], outs[tuned], s, skip, skip, skip
            except Exception as e:
                outs[name] = f"Generation failed: {e}"
            s.add(mode="compare", model=name, image=img_name, turn=1, prompt=prompt.strip(),
                  response=outs[name], latency_s=round(time.time() - t0, 2),
                  temperature=float(temperature), max_tokens=int(max_tokens), compare_id=cid)
        yield outs["base"], outs[tuned], s, download_update(s), summary(s), s.table()

    def clear_session(s):
        fresh = Session(export_root)
        return (fresh, gr.update(value=None, interactive=False), summary(fresh), [], [], {}, "", "")

    # -- layout ---------------------------------------------------------------------
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
                           radius_size="lg", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    major = int(gr.__version__.split(".")[0])
    blocks_kw = {} if major >= 6 else {"theme": theme, "css": CSS}
    launch_kw = {"theme": theme, "css": CSS} if major >= 6 else {}

    with gr.Blocks(title="Alzheimer's MRI Assistant", **blocks_kw) as demo:
        sess = gr.State(None)
        verdict = gr.State({"ok": False, "reason": ""})
        chatmap = gr.State({})
        img = gr.State(None)
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                upload_btn = gr.UploadButton("Upload brain MRI", file_types=["image"],
                                             file_count="single", type="filepath", variant="primary")
                viewer = gr.HTML()
                status = gr.HTML(pill("idle", "Upload a brain MRI slice to begin."))
                with gr.Accordion("Model and generation settings", open=False):
                    model = gr.Dropdown(be.models, value=default, label="Model",
                                        info="'base' is the original model before fine-tuning")
                    max_tokens = gr.Slider(64, 1024, value=cfg["eval"]["max_tokens"], step=32,
                                           label="Max answer length (tokens)")
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                            label="Temperature", info="0 gives the same answer every time")
                with gr.Group():
                    counter = gr.HTML(summary(None))
                    download = gr.DownloadButton("Download all inferences (.zip)", value=None,
                                                 interactive=False, variant="primary")
                    clear_all = gr.Button("Start a new session", variant="secondary", size="sm")
                gr.Markdown("<span class='counter'>The zip holds a CSV, a JSONL file, an HTML report "
                            "with thumbnails, and every image you queried.</span>")

            with gr.Column(scale=6, min_width=420):
                with gr.Tabs():
                    with gr.Tab("Ask"):
                        chat = make_chatbot()
                        with gr.Row():
                            prompt = gr.Textbox(show_label=False, lines=2, max_lines=6, scale=8,
                                                placeholder="Ask about this scan. Press Enter to send, "
                                                            "Shift+Enter for a new line.")
                            with gr.Column(scale=2, min_width=110):
                                send = gr.Button("Send", variant="primary")
                                stop = gr.Button("Stop", variant="stop", size="sm")
                        with gr.Row():
                            new_chat = gr.Button("New conversation", size="sm")
                        gr.Examples(EXAMPLE_PROMPTS, inputs=[prompt], label="Example questions (click to use)")
                        gr.Markdown("<span class='counter'>Follow-up questions keep the same image. "
                                    "Use the thumbs on an answer to rate it; ratings go into the export."
                                    "</span>")

                    with gr.Tab("Compare base vs fine-tuned", visible=can_compare):
                        cprompt = gr.Textbox(label="Question", lines=2,
                                             placeholder="One question, answered by both models")
                        gr.Examples(EXAMPLE_PROMPTS[:3], inputs=[cprompt], label="Examples")
                        cbtn = gr.Button("Run both models", variant="primary")
                        with gr.Row(equal_height=True):
                            with gr.Column(elem_classes="answer-card"):
                                gr.HTML("<div class='answer-title'>Base model</div>")
                                out_base = gr.Markdown()
                            with gr.Column(elem_classes="answer-card"):
                                gr.HTML(f"<div class='answer-title'>Fine-tuned ({html.escape(tuned)})</div>")
                                out_tuned = gr.Markdown()

                    with gr.Tab("Session history"):
                        table = gr.Dataframe(
                            headers=["#", "time", "mode", "model", "prompt", "response", "feedback"],
                            value=[], wrap=True, interactive=False)

        # -- events -----------------------------------------------------------------
        upload_btn.upload(on_upload, [upload_btn],
                          [status, verdict, chat, img, upload_btn, viewer]).then(
            lambda: {}, None, chatmap)
        chat_in = [img, prompt, chat, model, max_tokens, temperature, verdict, sess, chatmap]
        chat_out = [chat, prompt, sess, download, counter, table, chatmap]
        ev1 = send.click(respond, chat_in, chat_out)
        ev2 = prompt.submit(respond, chat_in, chat_out)
        stop.click(None, None, None, cancels=[ev1, ev2])
        new_chat.click(lambda: ([], "", {}), None, [chat, prompt, chatmap])
        chat.like(on_like, [sess, chatmap], [sess, table, download])
        cbtn.click(compare, [img, cprompt, max_tokens, temperature, verdict, sess],
                   [out_base, out_tuned, sess, download, counter, table])
        clear_all.click(clear_session, [sess],
                        [sess, download, counter, table, chat, chatmap, out_base, out_tuned])

    # Session exports live outside cwd and /tmp, so Gradio must be told it may serve them.
    demo.queue(default_concurrency_limit=2).launch(server_name="127.0.0.1",
                                                   server_port=args.port,
                                                   allowed_paths=[os.path.abspath(export_root)],
                                                   **launch_kw)


if __name__ == "__main__":
    main()
