# WaveRep (`waverep`, `waverep_g4`)

**Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation** (Corvi,
Cozzolino, Prashnani, De Mello, Nagano, Verdoliva; NeurIPS 2025) —
[paper (arXiv 2506.16802)](https://arxiv.org/abs/2506.16802) ·
[official code](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection) ·
[project page](https://grip-unina.github.io/WaveRep-SyntheticVideoDetection/)

A video detector built from an image backbone. A DINOv2 ViT-B/14 with registers is fine-tuned end to end on single
video frames. The method's contribution is the training data and an augmentation, not the architecture: in a wavelet
decomposition, a fake frame's low-pass and horizontal / vertical bands are replaced by those of a real frame, which steers
the detector towards the diagonal mid-high frequencies that generators leave behind and that survive compression. The
paper trains on ONE generator (Pyramid Flow) and tests on 15 unseen ones.

Two released checkpoints, both registered:

| key | checkpoint | trained on |
|---|---|---|
| `waverep` | `weights_dinov2_G1.ckpt` (`waverep.pth`) | one generator, Pyramid Flow (the paper's detector, Tables 1-2) |
| `waverep_g4` | `weights_dinov2_G4.ckpt` (`waverep_g4.pth`) | four generators: Pyramid Flow, CogVideoX 1.5, Allegro, Open-Sora Plan (Table 5; the demo's default) |

## Pipeline (official, reproduced exactly)

```
video ─ OpenCV VideoCapture, every frame in order, BGR -> RGB      (official demo/utils.py ReadVideoIteratorCV)
  → the first 64 frames                                               (paper, App. D; the demo's --limit 64)
  → per frame: CenterCrop(504) on the PIL frame (no resize; a smaller frame is zero-padded), ToTensor, ImageNet norm
  → DINOv2 ViT-B/14 + 4 registers (timm vit_base_patch14_reg4_dinov2, img_size 504) → one logit per frame
  → video logit = mean of the frame logits;  P(fake) = sigmoid(logit)  (logit > 0: synthetic)
```

Weights: `weights_dinov2_G{1,4}.ckpt` from grip.unina.it (md5 `bc9852a7…` / `8bf19e6f…`, as listed in the demo
README; sha256 `1c7566b2…a897` / `50d63904…5d53`). Each is a flat timm state dict of the full model, loaded strictly.

## SynthScope adaptations

| choice | reason |
|---|---|
| `SynthScopeWaveRepModel` holds the timm model as `self.model`; `registry.remap_waverep` prefixes the checkpoint keys with `model.` | The official `create_model` builds exactly this timm model and loads the checkpoint into it strictly. The remap also accepts the Lightning `{"state_dict": {"model.*"}}` form the official loader handles. |
| `pretrained_backbone: false` | The official code first downloads the timm DINOv2 weights, then overwrites every one of them with the checkpoint; building the model empty gives the same tensors without the download. |
| `official_utils.py` = the official `demo/utils.py`, verbatim | Reference for the tests and the parity check (GRIP license: nonprofit use; notices kept). SynthScope's reader `media.cv2_frames` is the same loop as `ReadVideoIteratorCV`, written out so that the package does not import pandas. |
| video sampler `cv2_frames`, `num_frames=64` | The paper scores the first 64 frames (App. D, Fig. 9). The demo scores every frame unless `--limit` is given; `dataclasses.replace(spec, num_frames=0)` reproduces that default. |
| frames scored `frame_batch` (16) at a time; the mean is taken in float64 | The demo scores one frame at a time and averages the float32 logits with pandas (float64). Batching changes the CPU logits by at most ~1e-5 (measured below) and bounds memory on long videos (`Engine.video_frame_logits` streams). |
| new transform step `center_crop_504` | torchvision `CenterCrop(504)` on the PIL frame: bit-exact with the official `CenterCrop((504, 504))`, zero padding included. |
| `compute_loss` = BCE with logits on single frames | The training code is not released (only the augmentation); no training recipe is reproduced. |

## Fidelity check (identical to the official model)

The unmodified official demo (`demo/main_avideo.py` @ `0fd6010`, CPU) against SynthScope's full pipeline (raw file ->
`media.cv2_frames` -> registry transform -> engine-built model, strict load -> `Engine.video_frame_logits` /
`predict_video_clip`), both checkpoints, same environment (torch 2.11, timm 1.0.27, OpenCV 4.13), on 15 videos:
GenVideo-Val (MSR-VTT reals 298×224, Sora 1280×720 × 600 frames, WildScrape `.mov` 2160×544 and `.gif`), EvalCrafter
(Hot Shot 672×384 × 8, MoonValley 1184×672 at 50 fps), VideoPhy (Dream Machine 1360×752 × 121), VidProM (ModelScope
256×256 MPEG-4 part 2, Text2Video-Zero 512×512), the demo's own two videos, and three re-encodes (VP9 `.webm`, MJPEG
`.avi`, HEVC `.mkv`); codecs H.264, MPEG-4 part 2, GIF, VP9, MJPEG, HEVC; 8 to 600 frames.

| check | result |
|---|---|
| state dict | 177/177 tensors identical (sha256), 0 missing / 0 unexpected, no transformers key translation (timm model), both checkpoints |
| model inputs (every frame of every video) | bit-exact: same frame count, every 504×504 input tensor identical (max abs diff 0.0) |
| per-frame logits, every frame (demo default) | max abs diff 4.8e-6 (frames batched 16 at a time; the demo runs one at a time) |
| video logit / P(fake), every frame | max abs diff 6.0e-7 / 2.8e-8 |
| served protocol (first 64 frames) vs the demo with `--limit 64` | max abs logit diff 2.7e-7 |
| `frame_batch=1` (the demo's batch) vs the demo | per-frame max abs diff 1.8e-15: identical; the 4.8e-6 above is CPU batching only |

Gate check on the demo's own videos (official demo output, logit > 0 = synthetic): `opensoraplan_-J24pksGK64_2.mp4`
4.80 (G1) / 9.33 (G4), `real_-D8sIkdnSkM_0.mp4` -9.03 / -8.49.

**Frame decoder.** SynthScope reads frames with OpenCV exactly as the demo, so there is no decoder difference. Measured
for reference (first 64 frames, both checkpoints): the demo's alternative PyAV reader (`ReadVideoIteratorAV`) gives the
same logits on all 15 videos (diff 0.0); decord (SynthScope's sampler for image detectors on video) gives the same
logits on 13, fails on the GIF, and on the WildScrape `.mov` returns 58 frames where OpenCV and PyAV decode 52 (OpenCV
reports 58 in its header but stops reading after 52), which moves the G4 logit by 0.26 (P(fake) by 0.064). Protocol
matters too: on the 600-frame Sora clip the G4 logit is -2.61 over the first 64 frames and -1.20 over every frame.

## Results

Benchmarks: see Task 2 of the integration (to be added).

## License

GRIP-UNINA license (`LICENSE.md` of the official repository): free for informational and nonprofit purposes only,
original notices kept, and the authors credited whenever results are made public. The checkpoints (distributed by GRIP
alongside the code) are treated as covered by the same terms, as for B-Free. Citation (official, from the project page):

```
@InProceedings{corvi2025seeing,
    title = {Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation},
    author = {Riccardo Corvi and Davide Cozzolino and Ekta Prashnani and Shalini De Mello and Koki Nagano and Luisa Verdoliva},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year = {2025},
    url = {https://openreview.net/forum?id=dOGXKBL7IE}
}
```
