"""FakeVLM (Wen et al., NeurIPS 2025, arXiv 2503.14905): LLaVA-1.5-7B fine-tuned on FakeClue to answer "Does the image
looks real/fake?" with "This is a real image." or "This is a fake image." followed by an explanation of the artifacts.

Official code: github.com/opendatalab/FakeVLM @ fdcdcf1 (no licence file); weights: HF lingcco/fakeVLM @ dce9e88d (no
licence; a full LLaVA-1.5-7B in bf16, 3 safetensors shards, 14.1 GB; base LLaVA-1.5 / Vicuna-7B v1.5 under the Llama 2
Community License). SynthScope loads the folder strictly into transformers' ``LlavaForConditionalGeneration`` and the
processor from a LOCAL copy of llava-hf/llava-1.5-7b-hf @ a272c74 (tokenizer and processor files only), as the
official ``scripts/eval.py`` does (``AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')``);
FakeVLM's own tokenizer appends ``</s>`` to the prompt (``add_eos_token=true``) and is not used.

The official request (``scripts/eval.py``, transformers 4.45.2), reproduced here (repro/fakevlm_fidelity.json):

* image: ``Image.open`` as is; the processor converts to RGB (alpha dropped, no EXIF rotation), resizes the shorter side
  to 336 (PIL bicubic) and centre-crops 336 x 336. transformers 5 does the resize with torchvision, which differs, so
  the 4.45.2 resize and crop run as the ``preprocess`` hook (``models/vlm/image.py``, pixel-equal to 4.45.2) and the
  processor is called with ``do_resize=False, do_center_crop=False``;
* prompt: the raw FakeClue question ``"<image>Does the image looks real/fake?"`` (no chat template; the training
  template ``USER: <image>\\n... ASSISTANT:`` gives an empty answer), tokenised with ``padding="max_length",
  max_length=1024, truncation=True`` (left padding); on transformers 5 the a272c74 processor expands ``<image>`` to
  576 tokens given ``patch_size=14, vision_feature_select_strategy="default", num_additional_image_tokens=1``
  (4.45.2 merged the image inside the model);
* decoding: ``generate(max_new_tokens=256)`` with the model's generation config (greedy);
* verdict: ``parse_fakevlm`` (``models/vlm/parsers.py``, the official rule; its random fallback is "unparsed").

The answer starts "This is a fake / real image." (tokens "▁▁", "▁This", "▁is", "▁a", then "▁fake" / "▁real"), so the
verdict is the FIRST "▁fake" / "▁real" token generated, and P(fake) = sigmoid(l(▁fake) - l(▁real)) there (the counterpart
pair; the vocabulary has one spelling of "fake" and four of "real", so a set renormalisation would be asymmetric).

Two scorers (``verdict_mode``):

* ``FAKEVLM_GENERATE`` ("generate"): the official greedy generation (up to 256 tokens), the logits of the step that
  emitted the verdict token; the verdict is the official parser on the generated text. The reference.
* ``FAKEVLM`` ("teacher_forced", the registry default): one forward pass on the prompt + the four answer tokens before
  the verdict ("▁▁ This is a", the verdict sequences' shared prefix), the logits of the next token; the verdict is the
  sign of the gap. It reads the same position as the generated path whenever the answer starts with those four tokens
  (every fidelity image), and agrees with it to float noise (repro/fakevlm_fidelity.json); no rationale is written.
"""
import dataclasses

from models.vlm.image import clip_resize_center_crop, transformers_rgb
from models.vlm.parsers import parse_fakevlm
from models.vlm.vlm import LLAVA, VLMConfig

USER_PROMPT = "<image>Does the image looks real/fake?"      # FakeClue test.json, every row, verbatim
GREEDY_DECODING = {"do_sample": False, "max_new_tokens": 256}   # eval.py: generate(**inputs, max_new_tokens=256)
# The answer's first tokens ("  This is a" + the verdict word) under the a272c74 Llama tokenizer.
VERDICT_SEQUENCES = (("▁▁", "▁This", "▁is", "▁a", "▁fake"), ("▁▁", "▁This", "▁is", "▁a", "▁real"))
PROCESSOR = "llava-1.5-7b-hf_a272c74_processor"             # folder under $SYNTHSCOPE_CKPT_DIR


FAKEVLM_GENERATE = VLMConfig(
    family=LLAVA,
    model_class="LlavaForConditionalGeneration",
    system_prompt="",
    user_prompt=USER_PROMPT,
    parser=parse_fakevlm,
    answer_marker=None,                    # the verdict is the first "▁fake" / "▁real" generated
    verdict_pairs=(("fake", "real"),),
    verdict_sequences=VERDICT_SEQUENCES,
    verdict_mode="generate",
    logits="raw",
    decoding=GREEDY_DECODING,
    official_decoding=GREEDY_DECODING,     # the official request is greedy too
    chat_template=False,
    image_loader=transformers_rgb,
    preprocess=clip_resize_center_crop,    # transformers 4.45.2's CLIP resize (PIL bicubic) + centre crop
    preprocess_kwargs={"shortest_edge": 336, "crop_size": 336},
    processor=PROCESSOR,
    processor_overrides={"patch_size": 14, "vision_feature_select_strategy": "default",
                         "num_additional_image_tokens": 1},
    processor_kwargs={"padding": "max_length", "max_length": 1024, "truncation": True,
                      "images_kwargs": {"do_resize": False, "do_center_crop": False}},
    max_pixels=None,
    hf_repo="lingcco/fakeVLM",
    hf_revision="dce9e88df7d1053e72af197c1bd3c61c94089e96",
    processor_hf_repo="llava-hf/llava-1.5-7b-hf",
    processor_hf_revision="a272c74b2481d8aff3aa6fc2c4bf891fe57334fb",
    processor_hf_include=("*.json", "tokenizer.model"),
)

FAKEVLM = dataclasses.replace(FAKEVLM_GENERATE, verdict_mode="teacher_forced")
