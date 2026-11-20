"""Align, Don't Blend (AlignGemini; arXiv 2512.06746, ACM MM 2026): the VLM branch, a LoRA on Qwen2.5-VL-7B-Instruct
that writes a short rationale and then answers "This is an authentic image." or "This is an AI-generated image.".

Official code: github.com/GzZning/Align-Dont-Blend @ f4b4794 (Apache-2.0 since 2026-11-20); adapter: HF
Gaffeyzz/Align-Dont-Blend-VLM @ 49975ccd (r16, alpha 32, 196 LoRA modules on the language model's attention and MLP
projections; Apache-2.0 on the card since ff7973df, same files); base: HF
Qwen/Qwen2.5-VL-7B-Instruct @ cc594898 (Apache-2.0). SynthScope loads the base with transformers into
``Qwen2_5_VLForConditionalGeneration`` from a LOCAL folder (``base_model``, relative to $SYNTHSCOPE_CKPT_DIR) and keeps
the adapter UNMERGED (peft), as ms-swift's pt backend runs it. Only the VLM branch is released: the paper's
final verdict is VLM OR a pixel-artifact expert, which is not released, so this detector is the VLM branch alone.

The official inference is ``predict.py`` -> ``swift infer`` (ms-swift; vLLM backend by default, or ``--infer_backend
pt``). The request is reproduced here (checked token for token against ms-swift 4.4.2 with the pt backend; the vLLM
backend was not compared; repro/adb_vlm_fidelity.json):

* image: ms-swift's loading (``models/vlm/image.py``: ``swift_rgb``, a plain RGB conversion without EXIF rotation and with
  the alpha channel dropped), then the ``--max_pixels 1003520`` (1280 x 28 x 28) resize of swift's Qwen-VL template
  (``swift_qwen_vl_resize``: ``rescale_image`` above the cap, then qwen-vl-utils ``fetch_image`` to multiples of 28,
  3,136 to 1,003,520 pixels) as the ``preprocess`` hook; the processor is then called with ``do_resize=False``, as
  swift calls it;
* system prompt and user prompt: ``VLM_SYSTEM_PROMPT`` / ``VLM_USER_PROMPT`` of ``detector/core.py``, verbatim; swift
  replaces the user prompt's ``<image>`` tag with the vision tokens, i.e. user content = [image, the text after the tag];
* decoding: ``--temperature 0`` (greedy) and ``--max_new_tokens 512`` from ``predict.py``; ms-swift leaves the repetition
  penalty unset, so the base model's ``generation_config.json`` value 1.05 applies (it changes the text and the score);
  ``--seed 42`` does not matter for greedy decoding. ``GREEDY_DECODING`` is therefore both the default and the
  official setting.

The output is ``<rationale> ... </rationale>`` then ``<answer>\nThis is an AI-generated image.\n</answer>`` (or
"authentic"); about 2% of outputs omit the tags and end with the bare line "This is an AI-generated / authentic image.",
which the official parser also accepts. The verdict is parsed with the official ``parse_vlm_verdict``
(``models/vlm/parsers.py``); an output it rejects (the official script stops on it) is unparsed. P(fake) is read where
the parser read the verdict: at the first "AI" / "authentic" token after ``<answer>``, or, for a tag-less output, at the
"AI" / "authentic" token of the final line (``adb_verdict_offset``). It is the sigmoid of the RAW LM-head logit gap
l(AI) - l(authentic) of the emitted token and its same-spelling counterpart ("ĠAI" / "Ġauthentic"); the repetition
penalty only drives generation (it shrinks both logits, as both words are in the prompt), so the score does not depend
on it beyond the generated text.
"""
from models.vlm.image import swift_qwen_vl_resize, swift_rgb
from models.vlm.parsers import adb_verdict_offset, parse_adb
from models.vlm.vlm import QWEN_VL, VLMConfig

# detector/core.py @ f4b4794, verbatim (the user prompt without its leading "<image>" tag, which becomes the image).
SYSTEM_PROMPT = """Role: AI visual assistant.
You are an AI visual assistant that analyzes a single image to determine whether it is authentic or AI-generated.
Tasks
1. <rationale> section:
• Provide concise, evidence-based reasoning in fewer than 120 words.
• Focus strictly on semantic or logical reasoning.
2. <answer> section:
• If real: only output “This is an authentic image.”
• If AI-generated: only output “This is an AI-generated image.”
Output Structure (must match exactly):
<rationale></rationale>
<answer></answer>"""
USER_PROMPT_OFFICIAL = "<image>\nWhether the image is authentic or AI-generated?"
USER_PROMPT = USER_PROMPT_OFFICIAL[len("<image>"):]

MAX_PIXELS = 1280 * 28 * 28                            # predict.py DEFAULT_MAX_PIXELS (--max_pixels, MAX_PIXELS)
GREEDY_DECODING = {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.05}


ADB_VLM = VLMConfig(
    family=QWEN_VL,
    model_class="Qwen2_5_VLForConditionalGeneration",
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    parser=parse_adb,
    verdict_offset=adb_verdict_offset,     # a tag-less final line: the score is read at its verdict word
    answer_marker="<answer>",
    answer_end="</answer>",
    verdict_pairs=(("AI", "authentic"), ("Ai", "Authentic"), ("ai", "authentic")),
    verdict_mode="generate",
    logits="raw",
    decoding=GREEDY_DECODING,
    official_decoding=GREEDY_DECODING,     # the official request is greedy too (temperature 0)
    chat_template=True,
    image_loader=swift_rgb,
    preprocess=swift_qwen_vl_resize,       # swift's Qwen2.5-VL resize at --max_pixels (rescale_image + fetch_image)
    preprocess_kwargs={"max_pixels": MAX_PIXELS},
    processor_kwargs={"images_kwargs": {"do_resize": False}},
    max_pixels=None,                       # the cap is applied by the preprocess hook (preprocess_kwargs)
    base_model="qwen2.5-vl-7b-instruct",
    hf_repo="Gaffeyzz/Align-Dont-Blend-VLM",
    hf_revision="49975ccd8b28eaab8ef14650ffee0c545934d02b",
    base_hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
    base_hf_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
)
