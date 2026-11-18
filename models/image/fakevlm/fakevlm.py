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
  max_length=1024, truncation=True`` (left padding, ``<image>`` one token); 4.45.2 then expands ``<image>`` inside
  the model into the 576 image features, so the model sees 1,599 positions, 1,013 of them padding. On transformers 5
  the a272c74 processor does the expansion (given ``patch_size=14, vision_feature_select_strategy="default",
  num_additional_image_tokens=1``) before padding, so ``max_length=1599`` (= 1024 - 1 + 576) gives the same layout.
  The padding is masked, but its length still moves the bf16 numerics: with ``max_length=1024`` (438 pads) the greedy
  text equalled the official one-image-at-a-time 4.45.2 run on 12 of 16 fidelity images, with 1599 on 16 of 16;
* decoding: ``generate(max_new_tokens=256)`` with the model's generation config (greedy);
* verdict: ``parse_fakevlm`` (``models/vlm/parsers.py``, the official rule; its random fallback is "unparsed").

The answer starts "This is a fake / real image." (usually "▁▁", "▁This", "▁is", "▁a", then "▁fake" / "▁real"; sometimes
without the "▁▁", with more spaces or a newline first), so the verdict is the FIRST "▁fake" / "▁real" token generated,
and P(fake) = sigmoid(l(▁fake) - l(▁real)) of that step's raw logits (the counterpart pair; the vocabulary has one
spelling of "fake" and four of "real", so a set renormalisation would be asymmetric).

Two scorers (``verdict_mode``):

* ``FAKEVLM`` ("generate", the registry default): the official greedy generation (up to 256 tokens), the logits of the
  step that emitted the verdict token; the verdict is the official parser on the generated text; the rest of the text
  is the model's explanation. About 0.5 s for a bare verdict (9 tokens), up to 7.5 s for 256 tokens on an A5000.
* ``FAKEVLM_TEACHER_FORCED`` ("teacher_forced", NOT the default): one forward pass on the prompt + "▁▁ This is a" (the
  verdict sequences' shared prefix), the logits of the next token; the verdict is the sign of the gap; about 0.27 s per
  image. It failed the default rule on the fidelity check (repro/fakevlm_fidelity.json): the generated answer does not
  always start with those four tokens (1 of the 16 fidelity images, 5 of 224 extended images), and P(fake) differs from
  the generated path's by up to 0.011 (0.052 on the extended set; bf16 logits, cached decoding vs one forward pass).
  Kept for fast screening; results scored with it are not the official method.
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
# eval.py pads the text to 1024 tokens with <image> as one token; transformers 4.45.2 expands it in the model to 576
# image features: 1024 - 1 + 576 positions. transformers 5 expands in the processor, before padding.
MAX_LENGTH = 1024 - 1 + 576


FAKEVLM = VLMConfig(
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
    processor_kwargs={"padding": "max_length", "max_length": MAX_LENGTH, "truncation": True,
                      "images_kwargs": {"do_resize": False, "do_center_crop": False}},
    max_pixels=None,
    hf_repo="lingcco/fakeVLM",
    hf_revision="dce9e88df7d1053e72af197c1bd3c61c94089e96",
    processor_hf_repo="llava-hf/llava-1.5-7b-hf",
    processor_hf_revision="a272c74b2481d8aff3aa6fc2c4bf891fe57334fb",
    processor_hf_include=("*.json", "tokenizer.model"),
)

FAKEVLM_TEACHER_FORCED = dataclasses.replace(FAKEVLM, verdict_mode="teacher_forced")
