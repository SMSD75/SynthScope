---
title: SynthScope AI-Generated Media Detection
emoji: 🔎
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# SynthScope demo (Gradio / Hugging Face Spaces)

Upload an image or video, pick a detector, and get the probability it is AI-generated. Built on
the [`synthscope`](https://github.com/SMSD75/SynthScope) Python API (`synthscope.load(key).predict(file)`).

## Run locally

```bash
pip install -e ".[demo]"        # from the SynthScope repo root
python demo/app.py              # serves on http://localhost:7860
```

## Deploy as a Hugging Face Space

1. Create a new **Gradio** Space.
2. Add these three files at the Space **root**: `app.py`, `requirements.txt`, and this `README.md`
   (the YAML header above configures the Space).
3. Set a Space **variable** `SYNTHSCOPE_HUB_REPO=<your-username>/synthscope-checkpoints` so detectors
   auto-download their weights from your [model zoo](../README.md#model-zoo-hugging-face-hub). For a
   **private** zoo, also add an `HF_TOKEN` Space **secret**.
4. Choose hardware: the large backbones (ConvNeXt-XXL, SigLIP, CLIP-L) want a **GPU**; CPU works but
   is slow.

## Notes

- Detectors needing **gated** backbones (DINOv3 → `dinov3`, `stall`) are hidden in this demo.
- Official-weights detectors (`fat_former`, `aide`, `npr`) only appear if their weights are in the
  configured zoo (they are license-restricted and not published by default — see `LICENSES.md`).
- A single score is a demo, not a verdict — see the repo's per-generator reproduction table for
  benchmarked accuracy.
