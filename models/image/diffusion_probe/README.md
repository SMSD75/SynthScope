# Diffusion-feature probe (`diffusion_probe`)

**An AGID-original baseline, not a published detector.** It asks the UnivFD question for a
*diffusion model*: are the internal features of a frozen text-to-image diffusion model linearly
separable into real vs. AI-generated? Same recipe as AGID's CLIP / DINOv2 linear probes — a frozen
backbone + a trainable `Linear(C, 2)` head — but the backbone is **Stable Diffusion v1.4**.

- Feature source — `backbone: unet` (default, **DIFT**-style, Tang et al., *Emergent Correspondence
  from Image Diffusion*, NeurIPS'23): `x∈[0,1] → ×2−1 → VAE.encode → z0`; add noise at one fixed
  timestep `t` (DDIM); run the UNet **one** step with an empty-prompt context; a forward hook on
  `unet.up_blocks[up_ft_index]` captures the feature map; spatial-pool → feature vector. `n_noise`
  averages features over N noise draws.
- `backbone: vae` — a deliberately cheap baseline: the SD-VAE latent pooled to a `GxG` grid, flattened.
- Backbone frozen; only the head trains. `predict_kind=ce` (engine reads `softmax(logits)[:,1]`).
- **Input is raw `[0,1]`** (registry `tf_resize_crop(256,224)`, no normalize) — the `×2−1` SD-VAE
  mapping happens *inside* `forward`. Do not add a `normalize_*` transform.

## Results (AGID-trained baseline — honest, per `repro/bench_results/dprobe_*.json`)

| benchmark | train data | mAP | mAUROC | notes |
|---|---|---|---|---|
| GenImage-8 | GenImage (SD-v1.4) | **78.3** | 79.0 | strong **in-distribution** SD (sdv1.4 96.5 / sdv1.5 96.8 / wukong 96.9 / midjourney 81.6 / glide 76.9), weak cross-family (adm 55.2 / vqdm 60.8 / biggan 61.5) |
| ForenSynths-13 | ProGAN | 75.4 | 76.6 | progan 97.0 in-dist; GANs moderate (gaugan 93.6 / cyclegan 87.9 / stargan 84.8), weak on san 53.9 / stylegan2 60.8 / whichfaceisreal 55.2 |
| UFD-8 | ProGAN | 74.8 | 75.9 | diffusion gens moderate (glide/ldm/dalle 74–80), ldm_200_cfg 59.9 |

**Takeaway:** frozen SD/DIFT features *are* linearly separable for **same-family** generations
(in-distribution AP 96–97), but generalize poorly across generator families — so this probe is a
useful diagnostic baseline, **not** competitive with AGID's SOTA detectors (PGC/SAFE/AIDE reach
97–100 mAP on GenImage). It is **not** in `trending.json` (no published SOTA paper backs it).

## Config knobs (`conf/model/image/diffusion_probe.yaml`)
`backbone` (`unet`|`vae`), `sd_id`, `image_size` (224 → 28×28 latent), `timestep` (50), `up_ft_index`
(1), `pool` (`mean`|`mean_std`), `n_noise` (noise-ensemble), `eval_fixed_noise`/`noise_seed`
(deterministic eval), `vae_pool_grid` (vae backbone).

## AGID adaptations
| choice | reason |
|---|---|
| Empty-prompt context encoded once, text encoder dropped | DIFT uses the `""` prompt; keeps the frozen checkpoint small (no CLIP text encoder weights). |
| Fixed-seed eval noise (`eval_fixed_noise`) | deterministic scores; `n_noise>1` averages draws to cut noise variance. |
| Raw `[0,1]` input, `×2−1` inside `forward` | the SD-VAE expects `[-1,1]`; keeps the dataset/transform pipeline standard. |
