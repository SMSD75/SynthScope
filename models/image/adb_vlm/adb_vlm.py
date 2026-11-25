"""Align, Don't Blend (AlignGemini; arXiv 2512.06746, ACM MM 2026): the VLM branch, a LoRA on Qwen2.5-VL-7B-Instruct
that writes a short rationale and then answers "This is an authentic image." or "This is an AI-generated image.".

Official code: github.com/GzZning/Align-Dont-Blend @ cf168db (Apache-2.0 since 2026-11-20; first integrated at f4b4794);
adapter: HF Gaffeyzz/Align-Dont-Blend-VLM @ 49975ccd (r16, alpha 32, 196 LoRA modules on the language model's attention and
MLP projections; Apache-2.0 on the card since ff7973df, same files); base: HF
Qwen/Qwen2.5-VL-7B-Instruct @ cc594898 (Apache-2.0). SynthScope loads the base with transformers into
``Qwen2_5_VLForConditionalGeneration`` from a LOCAL folder (``base_model``, relative to $SYNTHSCOPE_CKPT_DIR) and keeps
the adapter UNMERGED (peft), as ms-swift's pt backend runs it. This detector is the VLM branch alone: the paper's final
verdict is VLM OR a pixel-artifact expert (the released DDA checkpoint), which is ``adb_full`` (models/image/adb_full).

The official inference is ``predict.py`` -> ``swift infer`` (ms-swift; vLLM backend by default, or ``--infer_backend
pt``). The request is reproduced here as ``predict.py`` builds it at cf168db (the f4b4794 request was checked token for
token against ms-swift 4.4.2 with the pt backend, repro/adb_vlm_fidelity.json; the cf168db request end to end against
the unmodified ``predict.py``, repro/adb_full_fidelity.json; the vLLM backend was not compared):

* image: ms-swift's loading (``models/vlm/image.py``: ``swift_rgb``, a plain RGB conversion without EXIF rotation and with
  the alpha channel dropped), then the ``--max_pixels 1210000`` resize of swift's Qwen-VL template
  (``swift_qwen_vl_resize``: ``rescale_image`` above 1,210,000 pixels, then qwen-vl-utils ``fetch_image`` to multiples of
  28, 3,136 to 1,209,712 pixels: swift turns ``MAX_PIXELS=1210000`` into 1,543 tokens of 28 x 28) as the ``preprocess``
  hook; the processor is then called with ``do_resize=False``, as swift calls it. Commit 2f0adbc raised the cap from
  1,003,520 (1280 x 28 x 28);
* prompts (commit 2f0adbc, "Use the DPO user prompt for inference"): the system message is "You are a helpful
  assistant." and the task prompt is the user message, directly after ``<image>`` with no newline (``VLM_SYSTEM_PROMPT``
  / ``VLM_USER_PROMPT`` of ``detector/core.py``, verbatim); swift replaces the ``<image>`` tag with the vision tokens,
  i.e. user content = [image, the text after the tag]. Before 2f0adbc the task prompt was the system message and the
  user message "<image>\nWhether the image is authentic or AI-generated?" (``SYSTEM_PROMPT_F4B4794`` /
  ``USER_PROMPT_F4B4794``, kept for the record);
* decoding: ``--temperature 0`` (greedy), ``--max_new_tokens 512``, ``--repetition_penalty 1.0`` and ``--top_p 1.0`` from
  ``predict.py`` (the last two since 2f0adbc; before, swift left the repetition penalty unset and the base model's
  ``generation_config.json`` value 1.05 applied); ``--seed 42`` and top_p do not matter for greedy decoding.
  ``GREEDY_DECODING`` is therefore both the default and the official setting.

The output is ``<rationale> ... </rationale>`` then ``<answer>\nThis is an AI-generated image.\n</answer>`` (or
"authentic"); some outputs omit the tags and end with the bare line "This is an AI-generated / authentic image.",
which the official parser also accepts. The verdict is the official ``predict.py`` decision
(``models/vlm/parsers.py`` ``parse_adb_default_real``): ``parse_vlm_verdict``'s verdict, and "real" for an output it
rejects (commit 898b607; before, the script stopped on it). P(fake) is read where the parser read the verdict: at the
first "AI" / "authentic" token after ``<answer>``, or, for a tag-less output, at the "AI" / "authentic" token of the
final line (``adb_verdict_offset``). It is the sigmoid of the RAW LM-head logit gap l(AI) - l(authentic) of the emitted
token and its same-spelling counterpart ("ĠAI" / "Ġauthentic"); with the repetition penalty at 1.0 the raw and the
processed logits are the same, so the emitted word is the sign of the gap. An unparsed output whose verdict word is not
found has P 0.5 (logit 0, "unscored").
"""
from models.vlm.image import swift_qwen_vl_resize, swift_rgb
from models.vlm.parsers import adb_verdict_offset, parse_adb_default_real
from models.vlm.vlm import QWEN_VL, VLMConfig

# detector/core.py @ cf168db, verbatim (the user prompt without its leading "<image>" tag, which becomes the image). The
# second line ends with a space in the official file.
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_OFFICIAL = (
    "<image>Role: AI visual assistant\n"
    "You are an AI visual assistant that analyzes a single image to decide whether it is authentic or AI-generated. \n"
    "\n"
    "Tasks:\n"
    "1. <rationale> section\n"
    "   - Provide concise, evidence-based reasoning in less than 120 words.\n"
    "   - Focus strictly on semantic or logical reasoning.\n"
    "2. <answer> section\n"
    "   - If real: output \"This is an authentic image.\"\n"
    "   - If AI-generated: output \"This is an AI-generated image.\"\n"
    "\n"
    "Output Structure (must match exactly):\n"
    "<rationale></rationale>\n"
    "<answer></answer>")
USER_PROMPT = USER_PROMPT_OFFICIAL[len("<image>"):]

# The request before commit 2f0adbc (detector/core.py @ f4b4794 / 5d83e40), which every result before the upstream sync used.
SYSTEM_PROMPT_F4B4794 = """Role: AI visual assistant.
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
USER_PROMPT_F4B4794 = "<image>\nWhether the image is authentic or AI-generated?"
MAX_PIXELS_F4B4794 = 1280 * 28 * 28
REPETITION_PENALTY_F4B4794 = 1.05                     # unset in swift: the base generation_config.json value

MAX_PIXELS = 1210000                                   # predict.py DEFAULT_MAX_PIXELS (--max_pixels, MAX_PIXELS) @ cf168db
GREEDY_DECODING = {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0}


ADB_VLM = VLMConfig(
    family=QWEN_VL,
    model_class="Qwen2_5_VLForConditionalGeneration",
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    parser=parse_adb_default_real,     # predict.py @ 898b607: an answer parse_vlm_verdict rejects is Real
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
