# NSG-VD (`nsg_vd`, `nsg_vd_pika`, `nsg_vd_unbalance`)

**Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection** (Zhang, Lian, Yang, Li, Pang, Liu, Han, Li,
Tan; NeurIPS 2025 Spotlight) — [paper (arXiv 2510.08073)](https://arxiv.org/abs/2510.08073) ·
[official code](https://github.com/ZSHsh98/NSG-VD) (Apache-2.0)

A physics-motivated video detector. Under probability-flow conservation, a real video's density changes over time only
by moving probability mass around; the paper's statistic, the **Normalized Spatiotemporal Gradient** (NSG), is the ratio
of the spatial score (estimated by a pre-trained diffusion model) to the temporal change of the frames. Generated videos
violate the conservation and their NSG drifts. A video is scored by a deep-kernel **Maximum Mean Discrepancy** between
its NSG features and those of a fixed set of real reference videos: above 1 it is called generated.

Three released checkpoints are registered, the paper's main method (MMD-MP, Eq. 13) in its three tables:

| key | checkpoint (official `ckpts/`) | trained on | paper table |
|---|---|---|---|
| `nsg_vd` (default) | `standard-SEINE-mp.pth` (`nsg_vd.pth`) | Kinetics-400 + SEINE | Table 2 |
| `nsg_vd_pika` | `standard-Pika-mp.pth` (`nsg_vd_pika.pth`) | Kinetics-400 + Pika | Table 1 |
| `nsg_vd_unbalance` | `unbalance-SEINE-mp.pth` (`nsg_vd_unbalance.pth`) | Kinetics-400 + 1,000 SEINE | Table 3 |

The default is the SEINE model, which has the best released results (`assets/nsg-vd-test_results`). The three MMD-D
checkpoints (`*-d.pth`) are not registered (they appear only in the released results files, not in the paper's tables);
the parity check below covers them too.

## Pipeline (official, reproduced exactly)

```
video ─ moviepy frame count n = int(fps * duration); k = max(1, n // 8)          (data/video_frames_dataset.py)
  → ffmpeg -vf select='not(mod(n\,k))',setpts=N/FRAME_RATE/TB -vframes 8 → frame1..8.jpg  (fewer than 8: video skipped)
  → each JPEG: PIL resize (224, 224) LANCZOS, /255                       (score_feature_dataset.py)
  → guided-diffusion 256x256 uncond (fp16), t = (5 + 1) / 1000: s = -eps(2x - 1, 999 t) / std(t), first 3 channels
  → NSG = s / (<s, central frame difference> + 1e-10) per frame; first and last frame x100     [8, 3, 224, 224]
  → TALL single Swin block (8 frames tiled 1x8, resized to 224) → 300-d feature
  → MMD_batch2 vs the 100 reference videos (deep kernel, trained sigma / sigma0 / epsilon); fake if > 1
```

The reference set is the official one: the first 100 ids of `split/real/MSR-VTT/val_ids.txt` (released
`assets/split.zip`; `test.yaml`: `ref_model: MSR-VTT`, `ref_mode: val`, `ref_load_len: 100`). Its NSG features
(100 x 8 x 3 x 224 x 224 float32, 482 MB) are precomputed once with `build_reference.py` and shipped as
`nsg_vd_reference_msrvtt_val100.safetensors`, the same precedent as STALL's calibration statistics. The reference
discriminator features depend on the checkpoint and are computed from the file at the first score.

Files in `$SYNTHSCOPE_CKPT_DIR`:

| file | what | source |
|---|---|---|
| `nsg_vd.pth`, `nsg_vd_pika.pth`, `nsg_vd_unbalance.pth` | the released checkpoints, unchanged (a `deep_MMD` state dict, 47 tensors) | official repo `ckpts/` |
| `256x256_diffusion_uncond.pt` | OpenAI guided-diffusion (MIT), 2.2 GB, md5 (base64) `/Z3SM1uHNtUh3grtVL2Qyg==` | openaipublic.blob.core.windows.net/diffusion/jul-2021/ |
| `nsg_vd_reference_msrvtt_val100.safetensors` | the reference NSG features | `build_reference.py` from the official split + MSR-VTT clips |

## SynthScope adaptations

| choice | reason |
|---|---|
| `SynthScopeNSGVDModel` holds the official `deep_MMD` as `mmd` (`registry.remap_nsg_vd` adds `mmd.`) and the diffusion UNet as `unet` (loaded in `__init__` from its own file, declared in `pretrained_prefixes`) | The checkpoints hold only the kernel parameters and the discriminator; strict load of all 47 tensors. |
| `official_tall.py`, `official_mmd.py`, `guided_diffusion/` vendored unchanged (only the needed parts; see headers); `official_nsg.py` ports `_get_score_features` / `_get_score_velocity` / `get_score_fn` line for line | `tests/test_nsg_vd_smoke.py` [D] checks them bit-identical against the official modules. |
| reference features shipped as a file, built by `build_reference.py` | The official test recomputes them from the 100 videos on every run; a deterministic port needs them fixed. |
| `media.ffmpeg_frame_count` re-implements moviepy's frame count; `media.ffmpeg_select_frames` runs the official ffmpeg command | SynthScope does not depend on moviepy; both checked against the official code (counts and decoded frames identical). |
| a video that gives fewer than 8 frames is a `VideoDecodeError` | The official dataset silently drops it. |
| the UNet's convolutions are converted to fp16 before each diffusion pass | The official `convert_to_fp16`; the engine casts models to fp32 after loading. On a CPU the UNet runs in fp32 (`cpu_fp32`), a deviation. |
| P(fake) = sigmoid((MMD - 1) / `prob_scale`) | The official output is the raw MMD thresholded at 1; the probability is for display (0.5 at the threshold). The engine reports the raw statistic as `logit`. |
| no training (`compute_loss` raises) | Use the official `train_dMMD.py`. |

## Fidelity check

The unmodified official code (`ZSHsh98/NSG-VD` @ `9a60f57`: `get_ref_dataloader` + `get_ref_features`,
`get_score_datasets`, the body of `test_dMMD.test_dMMD`) against SynthScope's full pipeline (raw `.mp4` ->
`media.ffmpeg_select_frames` -> `nsgvd_frame_224` -> engine-built model, strict load -> MMD against the shipped
reference file), all six released checkpoints, on 75 GenVideo-Val test videos in the official test order (15 MSR-VTT
reals from index 200 of the test ids, 15 ModelScope and 5 each of MorphStudio, MoonValley, HotShot, Show-1, Gen2,
Crafter, LaVie, Sora, WildScrape), RTX A5000, same environment (torch 2.11), same ffmpeg (static 4.2.2) on both
sides. Record: `repro/nsg_vd_official_parity_genvideo75.json` (per-video scores; jobs 286253 official, 286389 SynthScope).

| check | result |
|---|---|
| state dict | all 47 tensors of each of the 6 checkpoints load strictly (0 missing / 0 unexpected) and equal the file |
| reference set | the shipped file (`build_reference.py`) is bit-identical to the official `ref_data` (100 x 8 x 3 x 224 x 224, max abs diff 0.0), same 100 ids in the same order (none dropped) |
| frames | 75/75 videos: the 8 decoded frames identical to the official `video_frames/` JPEGs |
| frame count | `media.ffmpeg_frame_count` == moviepy 2.2.1 `int(fps * duration)` on all 18,228 val + test videos of the split (0 mismatches; job 286254, `repro/nsg_vd_framecount_genvideo_val.json`) |
| NSG features, official grouping (3 videos per diffusion pass, the official order) | bit-identical on 75/75 videos |
| scores, official grouping + the official MMD batches | **bit-identical (max abs diff 0.0) for all 6 checkpoints** |
| scores, the served engine path (1 video per pass, 1 per MMD batch) | max abs diff 0.038 / 0.024 / 0.022 (Pika-mp / SEINE-mp / unbalance-mp), 0.015 / 0.002 / 0.008 (the -d ones); official verdict (> 1) reproduced on 75/75 for every checkpoint |
| the official code against itself: its own NSG features scored one at a time, or in reshuffled batches (`test_dMMD` uses `shuffle=True`) | max spread 0.037 / 0.035 / 0.022 (mp), 0.0001 / 0.0015 / 0.005 (d) |

So the port is exact; the engine's per-video score differs from a given official run only through fp16 / batch numerics
(which videos share a diffusion pass and an MMD batch), by at most 0.038, about as much as the official test itself
varies with the batch a video lands in for the MMD-MP checkpoints (for the -d ones the diffusion batch dominates). The statistic is sensitive to them because
the trained kernel width `sigma0` is tiny (5e-7 to 1e-6 for the MMD-MP checkpoints).

**Frame extraction (ffmpeg build).** The official code shells out to whatever `ffmpeg` is installed; its version is not
pinned, and it matters. On the same 75 videos, with SynthScope's pipeline:

| ffmpeg | videos giving 8 frames | score vs the official run with 4.2.2 (SEINE-mp / Pika-mp / unbalance-mp) |
|---|---|---|
| 4.2.2 (static, imageio-ffmpeg 0.4.2) — used for parity | 75/75 | (exact, above) |
| 9.0.2 (static, imageio-ffmpeg 0.8.1) | 75/75 | mean abs diff 0.19 / 0.07 / 0.04, max 0.79 / 0.83 / 0.44; verdicts agree 70 / 70 / 73 of 75 |
| 7.0.2 (static, the repo's VINA / D3 build) | 68/75 (drops frames on all 5 MoonValley and 2 Sora clips) | mean abs diff 0.13 / 0.17 / 0.12 on the 68; verdicts agree 59 / 57 / 59 |

The JPEG encoders of 4.2.2 and 9.0.2 put the same frames ~0.8 grey levels apart on average, and the NSG amplifies it.
ffmpeg 7.0.x is unusable for this protocol. Point `$SYNTHSCOPE_FFMPEG` at a 4.x build to match the parity setting; which
build the authors used is unknown.
