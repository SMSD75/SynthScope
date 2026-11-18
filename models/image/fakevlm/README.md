# FakeVLM (`fakevlm`)

**Paper:** Wen et al., *Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact
Explanation*, NeurIPS 2025 ([arXiv 2503.14905](https://arxiv.org/abs/2503.14905)).
**Code:** [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM) @ `fdcdcf1` (no licence file).
**Weights:** [lingcco/fakeVLM](https://huggingface.co/lingcco/fakeVLM) @ `dce9e88d` (no licence, no model card): a full
LLaVA-1.5-7B in bf16, 14.1 GB. **Processor:** the tokenizer and processor files of
[llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) @ `a272c74`, as the official `eval.py`
loads them.

**Status.** Inference only, with the official weights. Not on the Space and not in the model zoo: neither the code nor
the weights has a licence, and the base LLaVA-1.5 / Vicuna is under the Llama 2 Community License ([Licence](#licence)).
**GenImage is in its training data** (FakeClue contains GenImage real and fake images, and Chameleon fakes). The
paper's benchmarks (FakeClue, LOKI, DMimage, DD-VQA, FF++) are not held here, so there is no paper reproduction claim.

- **Fidelity** (16 images): the greedy text equals the official code run one image at a time under the authors'
  transformers 4.45.2 on 16 / 16 images; the first sentence and verdict equal every official run, including the
  batched `eval.py`, whose explanations change with the batch size ([Fidelity](#fidelity)).
- **Benchmarks:** pending (the subsampled VLM protocol of `repro/vlm_bench/`).

## Method
FakeVLM fine-tunes LLaVA-1.5-7B on FakeClue (over 100,000 real and synthetic images in seven categories, with artifact
explanations in natural language). Asked "Does the image looks real/fake?", it answers "This is a real image." or
"This is a fake image." and then, usually for fakes, explains the artifacts.

## Official inference, as SynthScope reproduces it
`scripts/eval.py` (transformers 4.45.2, batches of 16 in `eval.sh`). SynthScope sends the same request through
transformers 5 (`models/image/fakevlm/fakevlm.py`, shared VLM kind `models/vlm/`, `LlavaFamily`).

| step | official (`scripts/eval.py`, transformers 4.45.2) | SynthScope |
|---|---|---|
| model | `LlavaForConditionalGeneration`, bf16, flash-attention 2 | the same class, bf16, SDPA (flash-attn is not installed); loaded strictly, offline |
| processor | `AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')` | the same files from a local folder (`llava-1.5-7b-hf_a272c74_processor/`), plus `patch_size=14`, `vision_feature_select_strategy="default"`, `num_additional_image_tokens=1`, which transformers 5 needs to expand `<image>` to the 576 tokens 4.45.2 merged in the model. FakeVLM's own tokenizer appends `</s>` to the prompt (`add_eos_token=true`) and is refused |
| image | `Image.open`; the processor converts to RGB (alpha dropped, no EXIF rotation), resizes the shorter side to 336 (PIL bicubic) and centre-crops 336 | `transformers_rgb` + `clip_resize_center_crop` (transformers 4.45.2's code, ported in `models/vlm/image.py`; pixel values equal to 4.45.2's to 2.4e-7); the processor then only rescales and normalises. transformers 5's own resize uses torchvision and differs by up to 0.03 |
| prompt | the raw FakeClue question `"<image>Does the image looks real/fake?"` (no chat template), `padding="max_length", max_length=1024, truncation=True`, left padding; 4.45.2 expands `<image>` inside the model, so the model sees 1,599 positions | the same prompt; `max_length=1599` after the processor's expansion, i.e. the same 1,599-position layout with 1,013 pads |
| decoding | `generate(max_new_tokens=256)`, the model's generation config (greedy) | greedy, 256 new tokens |
| verdict | the text after the last "?"; "real" in the first sentence, else "fake", else the same on the second sentence, else `random.choice` | the same rule (`parse_fakevlm`, `models/vlm/parsers.py`); where the official code picks at random, "unparsed" (counted wrong) |

**Score.** P(fake) = sigmoid(l(▁fake) - l(▁real)) of the raw LM-head logits at the **first** "▁fake" / "▁real" token
the model generates (the counterpart pair: the vocabulary has one spelling of "fake" and four of "real"). It is
usually the 5th token ("▁▁ This is a" first), sometimes the 4th or 6th.

**Teacher-forced variant (not the default).** `FAKEVLM_TEACHER_FORCED` reads the same gap from one forward pass on the
prompt + "▁▁ This is a", about 13x faster. It failed the rule for a default scorer: the greedy answer does not always
start with those four tokens, and its P(fake) differs from the generated one by more than 1e-3 on some images
([Fidelity](#fidelity)). Results scored with it are not the official method.

```python
import synthscope
det = synthscope.load("fakevlm")              # needs both folders in $SYNTHSCOPE_CKPT_DIR (see Licence)
r = det.predict("photo.jpg")                  # fake_probability, logit, verdict, response (the explanation), ...
```

## Fidelity
`repro/fakevlm_fidelity.py`, `repro/fakevlm_official.py` / `repro/fakevlm_fidelity.sbatch`. Results:
`repro/fakevlm_fidelity.json` and `repro/logs/fakevlm_fidelity_293873.out` (job 293873, RTX A5000, bf16, SDPA,
transformers 5.5.4; the official runs under transformers 4.45.2 from a `pip --target` folder, and under 5.5.4).

16 images: the gate's 6 GenImage val images, 7 SynthWildX, 2 RGBA WildRF PNGs and one 8 MP WildRF JPEG (256x256 to
3264x2448). Plus an **extended set** of 224 images (8 per class folder of GenImage val, SynthWildX and WildRF) for the
teacher-forced check, where only the first 8 generated tokens are needed.

The official script runs in batches (`eval.sh`: 16) and cannot run one image at a time: its `.squeeze()` drops the
batch axis. Its greedy text depends on the batch size: under 4.45.2, batches of 4 and of 2 agree on 14/16 texts, and
batches of 4 and single images on 11/16 (bf16 numerics; the padding and the batch change the reductions). So the
engine (one image at a time) is compared with five official runs: `eval.py` unmodified in batches of 4 and 2 under
4.45.2 and of 4 under 5.5.4, and eval.py's own dataset item, model and `generate` call one image at a time under 4.45.2
and 5.5.4.

| check | result |
|---|---|
| strict load | 686/686 tensors equal to the shards; no network access |
| inputs vs the official 4.45.2 processor | pixel values max abs diff 2.4e-7; token ids equal 16/16 (1,599 positions) |
| vs the official code one image at a time, transformers 4.45.2 (the authors' pin) | **identical text 16/16** |
| vs `eval.py` unmodified, 4.45.2, batches of 4 / of 2 | identical text 11/16 / 12/16; first sentence and parsed verdict 16/16 |
| vs `eval.py` unmodified, 5.5.4, batches of 4; the code one at a time, 5.5.4 | identical text 10/16 / 13/16; first sentence and parsed verdict 16/16 |
| vs a direct transformers call (the reference) | identical tokens 16/16, verdict step 16/16, verdict 16/16 |
| verdict logit gap, engine vs the direct call | max abs diff 0.0 (generated); 0.0 (teacher-forced) |
| official accuracy on the 16 | 13/16 in every official batched run and in the engine (3 reals called fake) |
| speed (engine, bs 1, generated) | 0.294 img/s (3.4 s/img, 29 tok/s): 0.63 s for a bare verdict (9 tokens), 8.3 s for 256 tokens |
| speed (teacher-forced variant) | 2.6 img/s (0.38 s/img) |
| peak memory (engine) | 15.1 GB allocated (weights 14.1 GB) |

Every text difference is in the explanation after the first sentence ("This is a fake / real image."), or in the
whitespace before it (one image under 5.5.4).

**Teacher-forced variant vs the generated score.** The rule for making it the default: the same verdict-token
position, |P(fake) diff| <= 1e-3 on every image, and every answer starting "This is a". Result: on the 16, the answer
starts "▁▁ This is a" 16/16, but |P diff| > 1e-3 on 2 (max 0.0023; logit gaps differ by up to 0.25). On the 224 extended
images, 219 start with those tokens (the rest lack the "▁▁" or start with a newline); |P diff| > 1e-3 on 29 (max 0.031,
gaps up to 0.75 apart); the verdicts agree on 223. Rule not met, so the **generated path is the default**.

**Three transformers-version steps** make the request equal to 4.45.2's:
- the processor attributes `patch_size`, `vision_feature_select_strategy`, `num_additional_image_tokens` (otherwise
  transformers 5 fails, or inserts 575 tokens);
- the PIL resize and crop of 4.45.2's CLIP processor (transformers 5's torchvision resize differs by up to 0.03);
- `max_length=1599`: 4.45.2 pads the 1024-token text and then expands `<image>` in the model, so it sees 1,599
  positions (1,013 pads). With `max_length=1024` (438 pads) the text equalled the official single-image 4.45.2 run on
  12/16 (job 293846); with 1599 on 16/16.

## SynthScope adaptations

| choice | reason |
|---|---|
| transformers 5.5.4 instead of 4.45.2 | the SynthScope environment. The three steps above make the request the same; the text then equals the official code run one image at a time under 4.45.2 on 16/16 |
| SDPA instead of flash-attention 2 | flash-attn is not installed; the official runs here used SDPA too |
| one image at a time | `eval.py` batches, and its text depends on the batch size (above); a single image is the batch-independent reference |
| P(fake) from the raw logits at the first "▁fake" / "▁real" | the official output is text only; AUROC needs a continuous score that agrees with the text |
| unparsed answers kept and flagged | the official script picks a random label for them; SynthScope counts them wrong (logit 0). None in the fidelity run |
| inference only, still images only | the training (LLaVA fine-tuning on FakeClue) is not ported |

## Licence
- Code: no licence (no LICENSE file; GitHub reports none). SynthScope vendors only the verdict rule of
  `scripts/eval.py` (`parse_fakevlm`) and copies the prompt.
- Weights (`lingcco/fakeVLM`): no licence tag and no model card. The base LLaVA-1.5-7B / Vicuna-7B v1.5 and the
  llava-hf tokenizer are under the Llama 2 Community License. FakeClue (the training data) is Apache-2.0.
- So: not redistributed (`NEVER_REDISTRIBUTE` in `synthscope/hub.py`), not in the model zoo, not on the Space. Download
  both folders yourself at the pinned revisions:

```bash
hf download lingcco/fakeVLM --revision dce9e88df7d1053e72af197c1bd3c61c94089e96 --local-dir $SYNTHSCOPE_CKPT_DIR/fakevlm
hf download llava-hf/llava-1.5-7b-hf --revision a272c74b2481d8aff3aa6fc2c4bf891fe57334fb --include '*.json' \
    --include 'tokenizer.model' --local-dir $SYNTHSCOPE_CKPT_DIR/llava-1.5-7b-hf_a272c74_processor
```
