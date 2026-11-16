# Veritas++ (`veritaspp`)

**Veritas++** (Tan et al., arXiv 2607.27113, July 2026) — [paper](https://arxiv.org/abs/2607.27113) ·
[official code](https://github.com/EricTan7/VeritasPP) (Apache-2.0) ·
[weights](https://huggingface.co/EricTan7/VeritasPP) (Apache-2.0)

A multimodal-LLM detector. The released model is Qwen3-VL-8B fine-tuned with the authors' ms-swift fork (the released
`args.json` records the last stage, a GKD distillation run) to look at an image, write a human-like rationale with self-checks, and end with a verdict in
`<answer> real </answer>` or `<answer> fake </answer>`. It is the first detector of SynthScope's **VLM kind**
(`models/vlm/vlm.py`, a family-adapter design: Veritas++ uses the Qwen-VL family), which later VLM detectors reuse.
Still images only (no video).

**Status.** Inference only, with the official weights (no training path in SynthScope: the official training runs in
the authors' ms-swift fork). Not on the Space (an 8B model does not run on the CPU Space) and not in the model
zoo (a 17.5 GB folder). Fidelity against a direct transformers call with the official request: see
[Fidelity check](#fidelity-check). Benchmark results: pending (the subsampled VLM protocol, a later task).

## Pipeline (official request, reproduced)

The official inference is a vLLM OpenAI server (`self_scripts/deploy/deploy_model.sh`: bf16, tensor-parallel 4) queried
by `self_scripts/infer/infer_vllm_single.py`, one image per request (the file's bytes, base64). vLLM is not in the
SynthScope environment, so the same request goes through transformers (`Qwen3VLForConditionalGeneration` + the model's
own `Qwen3VLProcessor`), with vLLM's image loading reproduced:

```
image file ─ decoded without an RGB conversion (media.decode_image_raw)
  → vLLM's loading (vLLM 0.31.0 ImageMediaIO.load_bytes, ported in models/vlm/image.py):
      ImageOps.exif_transpose (pixels upright as displayed), then RGB passes, RGBA is composited on white, other modes
      with transparency (LA, PA, palette tRNS) go through RGBA onto white, everything else is convert("RGB")
  → Qwen3-VL processor at its defaults: resized only to multiples of 32 px within 65,536 - 16,777,216 pixels
    (native resolution for almost every benchmark image), normalized, 16x16 patches, 2x2 merge
  → chat request: system  "You are an expert visual analyst. Please think about the question as if you were a human
                           pondering deeply. It’s encouraged to include self-reflection or verification in the
                           reasoning process. Then, give a final verdict within <answer> </answer> tags."
                  user    [image, "Is this image real or fake?"]
                  chat_template_kwargs {"enable_thinking": False}
  → generate (SynthScope: greedy; official: temperature 0.7, top-p 0.8, top-k 20 from generation_config.json)
  → "...rationale (about 400-650 tokens)... <answer> real </answer>"
```

The prompts are copied verbatim from `infer_vllm_single.py` @ 9a1f00b (the apostrophe in "It’s" is U+2019 there too);
`tests/test_veritaspp_smoke.py` compares them with the official file. The release's chat template does not use
`enable_thinking`, so that flag changes nothing; it is passed because the official request passes it.

The image loading equals vLLM 0.31.0's own `ImageMediaIO.load_bytes` on 16 synthetic files that cover every branch
(the eight EXIF orientations, RGBA, LA, palette with and without tRNS, L, CMYK, RGB, RGBA with EXIF):
`repro/vlm_image_vllm_parity.py`, run in a vLLM environment, records 16/16 identical (`repro/vlm_image_vllm_parity.json`).
The authors install the latest vLLM (`vllm>=0.5.1 -U`), so 0.31.0 is the reference. On the held benchmarks this
matters for transparent PNGs only: none of their JPEGs has an EXIF rotation (orientation 2-8; checked over WildRF,
SynthWildX, BFreeOnline, RealChain, RRDataset and AIGIBench), while WildRF has RGBA PNGs (PIL's plain `convert("RGB")`
would show the model the colours under the transparent pixels; vLLM shows white).

**Score.** P(fake) = sigmoid(l(Ġfake) - l(Ġreal)) at the verdict token: the first token after `<answer>` (and before
`</answer>`) that spells fake or real, compared with its same-spelling counterpart (`Ġreal` ↔ `Ġfake`, `ĠReal` ↔ `ĠFake`, ...). `logit` is that
gap, so benchmarks can rank by it. The logits are the raw LM-head outputs of the step that produced the verdict token
(read with a forward hook; identical to transformers' `output_logits`). Because the model has already argued its way to
an answer, the score is near 0 or 1, and the paper reports accuracy: the primary metric is the accuracy of the parsed
verdict, with AUROC / AP as extras. No verdict token (no `<answer>` generated, e.g. a truncated output, or no fake / real
word inside the answer): P(fake) = 0.5, logit 0, `score_source: "unparsed"` (ruling U1). Scoring holds a lock: the
logit recorder hooks the model's shared LM head, so concurrent requests are served one after the other.

**Verdict.** The official repo has no parser or metric script. The verdict is parsed with the rule of the authors'
earlier Veritas (v1) `self_scripts/infer/cal_metrics.py` (vendored in `models/vlm/parsers.py`): the first
`<answer>(.*?)</answer>`, lower-cased, "real" checked before "fake". The official script drops an unparsed answer
from the accuracy; SynthScope reports it as `verdict: "unparsed"` (ruling U1: counted as wrong for accuracy; its score
is the verdict token's gap when one was generated, e.g. a truncated `<answer> fake`, else 0.5).

Weights: `EricTan7/VeritasPP` @ `22ba6c6b9f623755c1a68a9816b83a0540039829` (4 safetensors shards, 750 tensors, 17.5 GB
bf16, plus the processor and tokenizer files), unchanged, as the folder `$SYNTHSCOPE_CKPT_DIR/veritaspp/`:

```bash
hf download EricTan7/VeritasPP --revision 22ba6c6b9f623755c1a68a9816b83a0540039829 --local-dir $SYNTHSCOPE_CKPT_DIR/veritaspp
```

```python
import synthscope
det = synthscope.load("veritaspp")             # or checkpoint="/path/to/VeritasPP-folder"
r = det.predict("photo.jpg")                   # fake_probability, logit, verdict, rationale, response, ...
```

## SynthScope adaptations

| choice | reason |
|---|---|
| transformers (`Qwen3VLForConditionalGeneration`, bf16, SDPA) instead of the vLLM server | vLLM is not in the SynthScope environment. The prompts, chat template and processor are the official ones and the image loading is vLLM's (above); the generated text can still differ from a vLLM run by kernel-level numerics. |
| **greedy decoding** (`do_sample=False`) by default | the official request samples at temperature 0.7 (top-p 0.8, top-k 20), so its verdicts are not deterministic. The sampled setting stays available (`Engine.vlm_score(..., decoding="official", seed=...)`); the fidelity check below compares the two. |
| `max_new_tokens=2048` | vLLM's default is the rest of the context. The rationales are 400-650 tokens; the bound only stops a run-away generation (it is then reported, `finished: false`, and the answer is unparsed). |
| the Veritas v1 parser; unparsed answers kept and flagged | Veritas++ releases no parser (see Verdict above). |
| P(fake) from the verdict-token logits | the official output is text only; the logit gap is the continuous score that agrees with the text (under greedy decoding the emitted token is the argmax, so the sign of the logit is the verdict). |
| no pixel cap (`max_pixels=None`) | the official processor default (16.8 MP) is used; see Memory below. |
| inference only, still images only, one image per call | no `trainer.py` dispatch (the official training in the ms-swift fork is not reproduced); the engine, the API and the web app refuse videos for this detector (a rationale per frame would take minutes per video); the web app scores it in a worker thread and caps `/api/score_batch` at 20 images. |

## Fidelity check

Pending (job on ivi-cn029): `repro/veritaspp_fidelity.py` (`repro/veritaspp_fidelity.sbatch`, which first checks that
every file of the folder came from the pinned revision and passes its sha256) compares the engine path with a direct
transformers `generate` that uses the official prompt, processor and vLLM image loading on 15 images (GenImage val,
SynthWildX and two transparent WildRF PNGs, real and fake, 256x256 to 2048x2048): strict load (750 tensors), identical
greedy tokens and verdict, the max |diff| of the verdict logit gap, greedy vs the official sampling, throughput and peak
memory on an RTX A5000, and a memory probe at 8, 16.8 and 64 MP.

## Speed and memory (provisional)

Until that job reports, the only measurement is the gate check's (6 GenImage images up to 1024x1024, RTX A5000, bf16,
a different script): about 17 s per image (0.057 images/s, about 29 tokens/s for 400-650-token rationales) and 18.5 GB
peak. Larger images cost more (a 16.8 MP image is about 16k visual tokens).

## Availability

- Not on the Hugging Face Space: an 8B model needs a GPU (18.5 GB in bf16 at 1024x1024, provisional).
- Not in the model zoo: the weights are Apache-2.0, so redistribution is allowed, but the release is a 17.5 GB
  multi-file folder that the zoo's one-file-per-key layout does not hold (`NEVER_REDISTRIBUTE` in
  `synthscope/hub.py`); `synthscope.load("veritaspp")` without the folder prints the download command above.

## Licence

Code: Apache-2.0 (EricTan7/VeritasPP, an ms-swift fork). Weights: Apache-2.0 (HF model card; ModelScope
`EricTanh/VeritasPP`). Base model Qwen3-VL-8B: Apache-2.0. The parser rule is from EricTan7/Veritas (Apache-2.0). See
[LICENSES.md](../../../LICENSES.md).
