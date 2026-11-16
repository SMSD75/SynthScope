# Veritas++ (`veritaspp`)

**Veritas++** (Tan et al., arXiv 2607.27113, July 2026) — [paper](https://arxiv.org/abs/2607.27113) ·
[official code](https://github.com/EricTan7/VeritasPP) (Apache-2.0) ·
[weights](https://huggingface.co/EricTan7/VeritasPP) (Apache-2.0)

A multimodal-LLM detector. The released model is Qwen3-VL-8B fine-tuned with the authors' ms-swift fork (the released
`args.json` records the last stage, a GKD distillation run) to look at an image, write a human-like rationale with self-checks, and end with a verdict in
`<answer> real </answer>` or `<answer> fake </answer>`. It is the first detector of SynthScope's shared **VLM kind**
(`models/vlm/vlm.py`), which later VLM detectors reuse.

**Status.** Inference only, with the official weights (no training path in SynthScope: the official training runs in
the authors' ms-swift fork). Not on the Space (an 8B model does not run on the CPU Space) and not in the model
zoo (a 17.5 GB folder). Fidelity against a direct transformers call with the official request: see
[Fidelity check](#fidelity-check). Benchmark results: pending (the subsampled VLM protocol, a later task).

## Pipeline (official request, reproduced)

The official inference is a vLLM OpenAI server (`self_scripts/deploy/deploy_model.sh`: bf16, tensor-parallel 4) queried
by `self_scripts/infer/infer_vllm_single.py`, one image per request. vLLM is not in the SynthScope environment, so the
identical request goes through transformers (`Qwen3VLForConditionalGeneration` + the model's own `Qwen3VLProcessor`):

```
image file ─ PIL decode, RGB                         (official: the file's bytes, base64, decoded by the server)
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

**Score.** P(fake) = sigmoid(l(Ġfake) - l(Ġreal)) at the verdict token: the first token after `<answer>` that spells
fake or real, compared with its same-spelling counterpart (`Ġreal` ↔ `Ġfake`, `ĠReal` ↔ `ĠFake`, ...). `logit` is that
gap, so benchmarks can rank by it. The logits are the raw LM-head outputs of the step that produced the verdict token
(read with a forward hook; identical to transformers' `output_logits`). Because the model has already argued its way to
an answer, the score is near 0 or 1, and the paper reports accuracy: the primary metric is the accuracy of the parsed
verdict, with AUROC / AP as extras.

**Verdict.** The official repo has no parser or metric script. The verdict is parsed with the rule of the authors'
earlier Veritas (v1) `self_scripts/infer/cal_metrics.py` (vendored in `models/vlm/parsers.py`): the first
`<answer>(.*?)</answer>`, lower-cased, "real" checked before "fake". The official script drops an unparsed answer
from the accuracy; SynthScope reports it as `verdict: "unparsed"` (P(fake) then comes from the step after `<answer>`, or
is 0.5 when no `<answer>` was generated) so that the caller decides.

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
| transformers (`Qwen3VLForConditionalGeneration`, bf16, SDPA) instead of the vLLM server | vLLM is not in the SynthScope environment. The request (prompts, chat template, processor, image) is the official one; the generated text can still differ from a vLLM run by kernel-level numerics. |
| **greedy decoding** (`do_sample=False`) by default | the official request samples at temperature 0.7 (top-p 0.8, top-k 20), so its verdicts are not deterministic. The sampled setting stays available (`Engine.vlm_score(..., decoding="official", seed=...)`); the fidelity check below compares the two. |
| `max_new_tokens=2048` | vLLM's default is the rest of the context. The rationales are 400-650 tokens; the bound only stops a run-away generation (it is then reported, `finished: false`, and the answer is unparsed). |
| the Veritas v1 parser; unparsed answers kept and flagged | Veritas++ releases no parser (see Verdict above). |
| P(fake) from the verdict-token logits | the official output is text only; the logit gap is the continuous score that agrees with the text (under greedy decoding the emitted token is the argmax, so the sign of the logit is the verdict). |
| no pixel cap (`max_pixels=None`) | the official processor default (16.8 MP) is used; see Memory below. |
| an RGBA / palette image is converted with PIL's `convert("RGB")` | as every SynthScope detector; how the official server treats transparency was not checked (no vLLM here). Benchmark images are almost all RGB. |
| inference only | no `trainer.py` dispatch; the official training (ms-swift fork) is not reproduced. |

## Fidelity check

Pending (job on ivi-cn029): `repro/veritaspp_fidelity.py` (`repro/veritaspp_fidelity.sbatch`) compares the engine path
with a direct transformers `generate` that uses the official prompt and processor on 13 images (GenImage val and
SynthWildX, real and fake, 256x256 to 2048x2048): strict load (750 tensors), identical greedy tokens and verdict, the
max |diff| of the verdict logit gap, greedy vs the official sampling, throughput and peak memory on an RTX A5000.

## Availability

- Not on the Hugging Face Space: an 8B model needs a GPU (18.5 GB in bf16).
- Not in the model zoo: the weights are Apache-2.0, so redistribution is allowed, but the release is a 17.5 GB
  multi-file folder that the zoo's one-file-per-key layout does not hold (`NEVER_REDISTRIBUTE` in
  `synthscope/hub.py`); `synthscope.load("veritaspp")` without the folder prints the download command above.

## Licence

Code: Apache-2.0 (EricTan7/VeritasPP, an ms-swift fork). Weights: Apache-2.0 (HF model card; ModelScope
`EricTanh/VeritasPP`). Base model Qwen3-VL-8B: Apache-2.0. The parser rule is from EricTan7/Veritas (Apache-2.0). See
[LICENSES.md](../../../LICENSES.md).
