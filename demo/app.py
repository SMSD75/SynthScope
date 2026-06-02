"""AGID Gradio demo — runs any detector on an uploaded image or video.

Deployable as a Hugging Face Space (see demo/README.md) or locally:

    pip install -e ".[demo]"
    python demo/app.py

On a Space (or any machine without local weights), detectors auto-download from the model zoo;
set the ``AGID_HUB_REPO`` env var to your checkpoints repo.
"""
from __future__ import annotations

import agid

# Detectors that need gated backbones we can't host on a public demo (DINOv3 weights).
_GATED = {"dinov3", "stall"}


def _model_choices():
    """Dropdown choices as (label, key), available/runnable detectors first."""
    models = [m for m in agid.list_models() if m["key"] not in _GATED]
    models.sort(key=lambda m: (not m["available"], m["modality"], m["key"]))
    return [(f"{m['name']} · {m['paper']} ({m['modality']})", m["key"]) for m in models]


def run_detection(file_path, model_key):
    """Score ``file_path`` with detector ``model_key``; return (label_dict, details_dict)."""
    if not file_path:
        return {}, {"error": "Upload an image or video first."}
    if not model_key:
        return {}, {"error": "Pick a detector."}
    try:
        det = agid.load(model_key)
        res = det.predict(file_path)
    except FileNotFoundError:
        return {}, {"error": f"No weights available for '{model_key}' "
                            "(not in the model zoo, or gated). Try another detector."}
    except Exception as exc:  # surface decode/runtime errors to the UI
        return {}, {"error": f"{type(exc).__name__}: {exc}"}
    label = {
        "AI-generated (fake)": res["fake_probability"],
        "Real": res["real_probability"],
    }
    return label, res


def build_demo():
    import gradio as gr

    choices = _model_choices()
    default = choices[0][1] if choices else None
    with gr.Blocks(title="AGID — AI-generated media detection") as demo:
        gr.Markdown(
            "# AGID — AI-generated image & video detection\n"
            "Upload an **image or video** and pick a detector to estimate the probability that it "
            "is **AI-generated**. Each detector is a re-implementation benchmarked in "
            "[AGID](https://github.com/SMSD75/AGID)."
        )
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Image or video", type="filepath",
                                  file_types=["image", "video"])
                model_in = gr.Dropdown(choices=choices, value=default, label="Detector")
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Prediction", num_top_classes=2)
                json_out = gr.JSON(label="Details")
        btn.click(run_detection, inputs=[file_in, model_in], outputs=[label_out, json_out])
        gr.Markdown(
            "_Per-generator generalization is the hard part of this task; a single score is a demo, "
            "not a verdict. See the repo's reproduction table for benchmarked accuracy._"
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0")
