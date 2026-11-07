# VINA (`vina`, `vina_video`)

**Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection** (arXiv 2605.21977) —
[paper](https://arxiv.org/abs/2605.21977) · [official code](https://github.com/mgiant/VINA) ·
[checkpoint](https://huggingface.co/mgiant/VINA_checkpoints)

VINA detects AI-generated images and AI-generated videos with one model. A DINOv3 ViT-L/16 is fine-tuned end to end
with a linear head on its CLS output, on images and video frames together: the frames of real and generated videos
act as a natural augmentation (compression, motion blur, varied content) that makes the image detector generalize,
and the same model then scores video frames. The released checkpoint `VINA_DINOv3-L_140k+DDA` is the one the official
README points to for both the image and the video benchmarks.

## Pipeline (official, reproduced exactly)

```
image ─ Resize(256, bicubic) on the short side ─ CenterCrop(256) ─ ToImage, ToDtype(float32) ─ ImageNet normalization
      → DINOv3 ViT-L/16 (eager attention, fp32) → pooler output (CLS after the final norm) [1024]
      → Linear(1024, 1) → logit;  P(fake) = sigmoid(logit)                 (official test.py / util.eval_model)

video ─ ffmpeg -vf fps=2 → JPEG frames 1.jpg, 2.jpg, ...                  (official preprocess/video2frame.py)
      ─ the central select_frame_nums = 1 frame (index (n - 1) // 2)       (configs/VINA_DINOv3.yaml, dataloader.py)
      ─ the image preprocessing above → logit per frame
      → mean of the frame logits → P(fake) = sigmoid(mean)
```

In SynthScope the two paths are two registry entries that share the model and the checkpoint:

* `vina` (image detector): the image path. Like every SynthScope image detector it also accepts a video, scored by the
  engine's generic frame sampling (16 frames spread over the clip, mean P(fake)), which is *not* the paper's protocol.
* `vina_video` (video detector): the official video protocol above, run by the engine's video path
  (`media.ffmpeg_fps_frames` + the image transform, sigmoid of the mean frame logit); `benchmark/bench_eval_video.py
  --model vina_video` uses it. It needs an ffmpeg binary, found as `$SYNTHSCOPE_FFMPEG`, else `ffmpeg` on the PATH,
  else the static build of imageio-ffmpeg. The SynthScope environment ships none: `pip install imageio-ffmpeg` or set
  `SYNTHSCOPE_FFMPEG=/path/to/ffmpeg`. Without one, `vina_video` is reported as unavailable. The parity below used the
  static ffmpeg 7.0.2 (the imageio-ffmpeg build); the whole clip is extracted because ffmpeg's JPEG rate control runs
  across frames, so other ffmpeg versions may encode the frames slightly differently.

Weights: `VINA_DINOv3-L_140k+DDA.pth` (HF `mgiant/VINA_checkpoints`, 1.21 GB, sha256 `1c46c9d8…c1d59d`) holds the
full model (all 415 backbone tensors + the head) and loads key-for-key as `vina.pth`; the gated DINOv3 weights are not
needed for inference.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `DINOv3` class of `models/DINOv3.py` | `SynthScopeVINAModel` subclasses it, so the checkpoint loads strictly; attribution and licence status in the header. |
| backbone built from its config (`DINOV3_VITL16`), not `from_pretrained('weights/dinov3-vitl16-pretrain-lvd1689m')` | the checkpoint already holds the backbone weights; `backbone_path` still loads a pretrained HF-format DINOv3 (for training). |
| ViT block prefix renamed on load (`backbone.layer` ↔ `backbone.model.layer`) | the checkpoint was saved with transformers 4.x; transformers 5 moved the blocks. |
| `remap_vina` unwraps `model_state_dict` and drops `module.` | the official `util.load_checkpoint` rule. |
| `forward` averages the logits of a `[B, T, 3, H, W]` clip | the official `util.eval_model` / `test.py` aggregation, so a clip is one call. |
| new transform steps `resize_short_256_bicubic` and `to_image` | the official eval is torchvision v2 `Resize(256, interpolation=3)` then `ToImage` + `ToDtype(float32, scale=True)`; `to_tensor`-style conversion differs from it by one float ulp, so the v2 order is kept for bit-exact inputs. |
| the video protocol uses the frame-directory path, not the `_mp4` path | the official benchmark keys (`genvideo`, `magic`, `reward`, ...) read ffmpeg frame directories; the `_mp4` keys decode with decord and, at test time too, pick a frame at `random.randint` (unseeded), so they are not reproducible. |
| image files of every format PIL reads | the official test loader only accepts `.jpg`, `.jpeg`, `.png`, `.webp` (other files become a black image); the pixels of the accepted formats are the same. |
| `compute_loss` = BCE with logits, `outputs` = `[-logit, logit]` | the official `util._classification_loss` for one output; the two columns let the trainer's metrics (argmax, softmax[:, 1]) read the single logit, as DDA / D3QE / B-Free. The training code is not released. |

## Fidelity check

The official code (`vina_ref` at commit 225d331, in a separate process with transformers 4.57.1, `test.py`'s
`build_inference_loader` + `infer_inputs` and configs/VINA_DINOv3.yaml) against SynthScope's full pipeline, both with
the same static ffmpeg 7.0.2:

* **images** (24: PNG, JPEG, WebP, BMP; 128×128 to 2048×1365, square and non-square): raw file →
  `bench_eval._open_rgb` → registry transform → engine-built model with `vina.pth` loaded by the engine → sigmoid;
* **videos** (8 GenVideo-Val clips: 3 MSR-VTT reals, Sora, Gen2, Crafter, Lavie, MorphStudio; 3 to 61 frames at
  2 fps): the official `preprocess/video2frame.py` then `test.py` on the frame directories, against SynthScope's
  `vina_video` on the mp4 (`engine.video_clip` → `engine._video_fake_prob`), with the served `select_frame_nums = 1`
  and with 8 (exercising the window, the last-frame padding of 3- to 5-frame clips and the logit mean).

Results (CPU, commit d0b87f6; logs `repro/logs/vinapar_cpu_{official,ours}_d0b87f6.log`):

* state dict: **all 417 tensors identical** (sha256 of the bytes; the only difference is the block-prefix name),
* model inputs: identical on all 24 images and on all 8 clips for both frame counts,
* P(fake) and logits: **max |diff| 0.0, 0 flips** on the 23 images the official loader reads (against its batch-1 run;
  1.2e-7 against its default batch-64 run, the official batch-64 vs batch-1 difference itself) and on all 8 videos for
  T = 1 and T = 8. The BMP file, which the official loader replaces by a black image, equals the official transform
  applied directly (max |diff| 0.0).

The official `_mp4` path (decord, a random frame) gives different scores on every run for the same clip (e.g. Sora
P(fake) 0.995963 / 0.997796 for two seeds vs 0.999641 from the frame directory); SynthScope follows the frame-directory
protocol.

Memory and speed (CPU, 16 threads on a shared login node, 256×256 input, fp32): 276 ms per image at batch 1 (3.6
img/s), 4.4 img/s at batch 8, peak RSS 2.96 GiB (`repro/logs/vinapar_cpu_speed_d0b87f6.log`); a `vina_video` clip
(ffmpeg extraction + one frame) took 0.80 s on average over the 8 parity clips.

## Results

Benchmark reproduction: pending (Task 2 of the VINA integration plan).

## Licence

The official repository has no LICENSE file and the Hugging Face checkpoint repository has no licence tag, so the
licence is **unclear** (LICENSES.md): the code is vendored with attribution for research use, and the weights are not
redistributed (not in the SynthScope model zoo, not on the Space).
