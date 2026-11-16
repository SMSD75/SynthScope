"""Veritas++ (arXiv 2607.27113): a Qwen3-VL-8B detector that reasons about an image, then answers real or fake.

Official code: github.com/EricTan7/VeritasPP @ 9a1f00b (an ms-swift fork, Apache-2.0); weights: HF EricTan7/VeritasPP @
22ba6c6b (Apache-2.0; base Qwen3-VL-8B, Apache-2.0). The released model is a full ``Qwen3VLForConditionalGeneration``
(750 tensors, 17.5 GB bf16); SynthScope loads it with transformers into that class (``models/vlm/vlm.py``, Qwen-VL
family).

The official inference is a vLLM OpenAI server (``self_scripts/deploy/deploy_model.sh``: bf16, TP 4) queried by
``self_scripts/infer/infer_vllm_single.py``, which sends the image file's bytes. The request is reproduced here:

* image: vLLM's loading (``models/vlm/image.py``: EXIF transpose, transparency composited on white, RGB), then the
  model's own Qwen3-VL processor at its defaults (65,536 to 16,777,216 pixels, i.e. native resolution for almost every
  benchmark image; patch 16, merge 2), as vLLM's HF processor;
* system prompt and user prompt: ``SYSTEM_PROMPT`` / ``USER_PROMPT`` of ``infer_vllm_single.py``, verbatim (the
  apostrophe in "It’s" is U+2019, as in the official file); user content = [image, text];
* ``chat_template_kwargs={"enable_thinking": False}`` (the release's chat template does not read it);
* decoding: the official request sets ``temperature=0.7``, the rest comes from ``generation_config.json``
  (``top_k=20``, ``top_p=0.8``, ``repetition_penalty=1.0``): ``OFFICIAL_DECODING``. SynthScope scores with greedy
  decoding (``GREEDY_DECODING``) so that results are deterministic; the sampled setting stays available
  (``decoding="official"``) and the README compares the two.

The output is a plain-text rationale (about 400-650 tokens) followed by ``<answer> real </answer>`` or
``<answer> fake </answer>``. The repo has no parser; the verdict is parsed with the authors' Veritas (v1) rule
(``models/vlm/parsers.py``). ``max_new_tokens=2048`` bounds a run-away generation (vLLM's default is the rest of the
context); a truncated output has no answer and is reported as unparsed. Greedy decoding here has no logits processor
(no repetition penalty), so the raw logits are the ones the argmax is taken over.
"""
from models.vlm.image import vllm_rgb
from models.vlm.parsers import parse_veritas_v1
from models.vlm.vlm import QWEN_VL, VLMConfig

# self_scripts/infer/infer_vllm_single.py @ 9a1f00b, verbatim.
SYSTEM_PROMPT = """You are an expert visual analyst.
Please think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags."""
USER_PROMPT = """Is this image real or fake?"""

GREEDY_DECODING = {"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0}
OFFICIAL_DECODING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 2048,
                     "repetition_penalty": 1.0}

VERITASPP = VLMConfig(
    family=QWEN_VL,
    model_class="Qwen3VLForConditionalGeneration",
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    parser=parse_veritas_v1,
    answer_marker="<answer>",
    answer_end="</answer>",
    verdict_pairs=(("fake", "real"), ("Fake", "Real"), ("FAKE", "REAL")),
    verdict_mode="generate",
    logits="raw",
    decoding=GREEDY_DECODING,
    official_decoding=OFFICIAL_DECODING,
    chat_template=True,
    chat_template_kwargs={"enable_thinking": False},
    image_loader=vllm_rgb,
    max_pixels=None,                       # the processor default (16,777,216); see the README's memory section
    hf_repo="EricTan7/VeritasPP",
    hf_revision="22ba6c6b9f623755c1a68a9816b83a0540039829",
)
