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
zoo (a 17.5 GB folder). Fidelity against a direct transformers call with the official request: 15 / 15 identical greedy
tokens and verdicts, max |logit-gap diff| 0.0 ([Fidelity check](#fidelity-check)). Benchmark results: pending (the subsampled VLM protocol, a later task).

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

`repro/veritaspp_fidelity.py` (`repro/veritaspp_fidelity.sbatch`, job 292005 on ivi-cn029, commit bd4d8d2, RTX A5000,
torch 2.11, transformers 5.5.4, bf16, SDPA; results `repro/veritaspp_fidelity.json`, log
`repro/logs/veritaspp_fidelity_292005.out`) compares SynthScope's engine path (`Engine._get_model` + `Engine.vlm_score`
on the file decoded as the API decodes it) with an independent reference: a direct transformers `generate` with the
official prompts (read from `infer_vllm_single.py`), the released processor and vLLM's image loading written out.
The job first checked that every file of the folder came from the pinned revision 22ba6c6b (hf download metadata; the
shard etags equal `SHA256SUMS`) and that all 15 files pass `sha256sum -c`.

Images: 15 (8 real, 7 fake) from GenImage val (the gate check's 3 real + 3 fake, 256x256 to
1024x1024), SynthWildX (3 real + 4 fake, 662x1000 to 2048x2048) and WildRF (2 real RGBA PNGs, transparency composited
on white).

| check | result |
|---|---|
| (a) strict load | `from_pretrained` loading info empty (missing 0, unexpected 0, mismatched 0); 750 / 750 tensors of the four shards equal to the loaded parameters (`torch.equal`), no other parameter |
| request | the engine's rendered prompt == the reference's |
| (b) greedy tokens | **15 / 15 identical** to the reference (341-653 new tokens each) |
| (b) parsed verdict | **15 / 15 identical**; verdict step identical on all 15; 0 unparsed |
| (c) verdict logit gap | **max \|diff\| 0.0** (engine hook vs reference `output_logits`) |
| accuracy, greedy | 15 / 15 correct; P(fake) 0.00016-0.00063 on the 8 reals, 0.99945-0.99997 on the 7 fakes |
| (d) official sampling (T 0.7, top-p 0.8, top-k 20, seed 0) | 15 / 15 correct; verdict agreement with greedy **15 / 15** |
| `predict_image` wrapper | same P(fake) and verdict as `vlm_score` |

## Speed and memory (RTX A5000, bf16, one image at a time)

| | measured |
|---|---|
| weights resident | 17.54 GB; engine load 3.9 s (node-local SSD, files already in the page cache from the reference run) |
| 15 fidelity images (up to 2048x2048, about 4.2k visual tokens) | **17.35 s per image (0.0576 images/s)**, 12.2-21.8 s, 29.2 generated tokens/s, 341-653 tokens per answer; peak allocated **18.91 GB** (at 2048x2048) |
| memory probe, WildRF 2448x3264 (8.0 MP) | 22.8 s, 7,826 prompt tokens, peak 20.12 GB |
| memory probe, WildRF 4096x4096 (16.8 MP, the processor's cap) | 37.8 s, 16,458 prompt tokens, peak **22.96 GB**, no OOM |
| memory probe, WildRF 6936x9248 (64 MP; the processor resizes it to 3520x4704) | 38.8 s, 16,244 prompt tokens, peak 22.89 GB, no OOM |

Nothing ran out of memory on the 24 GB card, so `max_pixels` stays at the processor's default (16,777,216 pixels): no
cap and no deviation from the official preprocessing. The cost grows with the image: about 17 s per image up to 4 MP,
about 38 s at the 16.8 MP cap.

One probe image is a model error, not a fidelity issue: the WildRF fake `reddit/1_fake/000325_0325.jpg` (4096x4096) was
called "real" (logit -6.375). The probe images are not part of the parity comparison (engine path only).

## Availability

- Not on the Hugging Face Space: an 8B model needs a GPU (17.5 GB of weights in bf16; 18.9 GB peak at 2048x2048, 23.0 GB at the
  16.8 MP processor cap).
- Not in the model zoo: the weights are Apache-2.0, so redistribution is allowed, but the release is a 17.5 GB
  multi-file folder that the zoo's one-file-per-key layout does not hold (`NEVER_REDISTRIBUTE` in
  `synthscope/hub.py`); `synthscope.load("veritaspp")` without the folder prints the download command above.

## Licence

Code: Apache-2.0 (EricTan7/VeritasPP, an ms-swift fork). Weights: Apache-2.0 (HF model card; ModelScope
`EricTanh/VeritasPP`). Base model Qwen3-VL-8B: Apache-2.0. The parser rule is from EricTan7/Veritas (Apache-2.0). See
[LICENSES.md](../../../LICENSES.md).
