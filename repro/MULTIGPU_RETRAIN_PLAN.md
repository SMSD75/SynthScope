# Multi-GPU retrain plan — push the "near" detectors to official (tomorrow)

*Goal: close the remaining AGID-vs-official gaps that this session showed are training/recipe-bound, using
multi-GPU so the heavyweight frozen backbones (SigLIP-384, MFM-ViT-448, ConvNeXt-XXL, dinov2-L) are
tractable. The diagnosis is done; this is execution.*

## 🚀 LAUNCHED overnight (2026-06-07 ~22:25) — all 4 retrains + auto-eval running
| Job | Node | GPUs | Status | Saves to | Eval tag |
|---|---|---|---|---|---|
| AIDE-faithful (182106) | cn029 | 4 | running | `checkpoints_retrain/aide_faithful` | `aide_faithful_g8/_fs` |
| Co-Spy-full (182107) | cn029 | 4 | running | `checkpoints_retrain/cospy_full` | `cospy_full_g8/_cospybench` |
| SPAI-full (182108) | cn025 | 4 | queued (8-GPU cap) | `checkpoints_retrain/spai_full` | `spai_full_g8` |
| SAFE-full (182109) | cn028 | 4 | queued (8-GPU cap) | `checkpoints_retrain/safe_full` | `safe_full_g8` |
- **8-GPU/account cap**: AIDE+Co-Spy hold all 8; SPAI+SAFE auto-start when they finish (~2-3 h).
- **Auto-eval**: `repro/gap_autopilot_v2.sh` (cn023 GPU4) evals each new checkpoint on GenImage-8 (+ForenSynths
  for AIDE, +CoSpyBench for Co-Spy) with `--limit`. Results stream to `repro/bench_results/`.
- **On wake**: read those tags, promote any that beat the shipped checkpoint (relink
  `checkpoints/<m>.pth`), refresh `REPRODUCED_MODELS_BENCHMARKS.md` + READMEs + registry.

## ✅ Done tonight (2026-06-07 eve)
- **AIDE preprocessing code-fix — DONE + CPU-tested.** `models/image/aide/aide.py` now takes a **raw
  [0,1] 256** image and applies, internally: **CLIP** normalization to the ConvNeXt-XXL semantic branch
  (256→8×8) and **ImageNet** normalization to the SRM/HPF patch branch, with the DCT run in pixel space —
  matching official AIDE (`repro/official_evals/aide_official_eval.py`). Registry transform →
  `tf_square_resize(256,None,None,None)`; `experiments/image/aide.sh` train/eval → raw-256 (no clip-norm,
  256 crops). Forward + `compute_loss` verified on CPU. **Retrain script ready:**
  `repro/retrain_aide_faithful_mgpu.sbatch` (4-GPU DeepSpeed, GenImage SD-v1.4). Just `sbatch` it.
- **`co_spy.pth` promoted** to the retrained model (CoSpyBench 95.88).

## Targets (priority order)
| # | Model | Now → target | Lever | Needs |
|---|---|---|---|---|
| 1 | **AIDE** | 91.75 → ~97 | code fix **DONE** → retrain | `sbatch repro/retrain_aide_faithful_mgpu.sbatch` |
| 2 | **Co-Spy** | GenImage-8 88.85 → ~92 | full-data + broader generators | full GenImage SDv1.4 (already staged) |
| 3 | **SPAI** | 88.8 → ~91 | full dmid_real + more epochs (448) | dmid_real (staged) + MFM ckpt |
| 4 | **SAFE** | 92.9 → ~99 | full recipe, more epochs | ProGAN/CNNDetection (stage to node) |
| 5 | PGC | 97.6 → ~99 | longer schedule (marginal) | GenImage (staged) |

CLIP, CNN, NPR, FatFormer, Co-Spy(CoSpyBench) already reproduce — no work.

## Optional speed optimization (NOT required — 4-GPU makes the retrains feasible already)
*Left for tomorrow with a validation pass — a buggy feature-cache would silently corrupt training, so it
must be checked against the live-forward numbers before relying on it. At 4 GPUs Co-Spy(full) ≈ 45 min/epoch
and the others are tractable, so this is a nice-to-have, not a blocker.*

**The backbones are frozen, yet every epoch recomputes them** → Co-Spy was 6 h/epoch, SPAI 8 h/epoch.
Two fixes, either makes the rest ~10× faster:
- **(a) Precompute frozen features once.** One forward pass over the training set caching
  `[SigLIP feat, VAE residual]` (Co-Spy) / `[MFM intermediate feats]` (SPAI) / `[dinov2 feat]` (PGC) /
  `[CLIP feat]` (VIB), then train only the head from the cache (seconds/epoch, many epochs cheaply).
  ~150 lines; biggest win. Lets us sweep LR/epochs/data mixes fast.
- **(b) Parallelise eval.** `benchmark/bench_eval.py` `score()` loads images single-threaded (~8 img/s →
  3 h on the 96k full set). Wrap in a `DataLoader(num_workers=16)`; ~10× faster evals. Small, safe change.

## Compute (nodes seen idle today)
- **cees `cn025`/`cn029`** — 8× rtx_a5000 each. GenImage + dmid_real already staged on cn029 /ssdstore.
- **cees6000 `cn031`** — 8× rtx_6000. **comcrim `cn007`/`cn009`** — 2× rtx_pro_6000 (new, fast).
  **biggpu `cn019`**. Submit with `--account=ceesusers` (cees) / per-partition accounts; DeepSpeed ZeRO-2
  `accelerate_config.yaml`, `--num_processes 4 --gres=gpu:4` (the prior AIDE 3-GPU runs used this pattern:
  `repro/retrain_aide_genimage_mgpu.sbatch`).
- **Data staging:** GenImage/dmid_real already on cn029. SAFE needs ProGAN → rsync
  `ivi-cn023:/hddstore/ssalehi/AGID/CNNDetection` to the node's scratch (~72G). Evals that need
  CoSpyBench/AIGCDetect (440G+) stay on **cn023 GPUs 4&7** (data is node-local there).

## Per-model recipe
1. **AIDE — fix then train.** Edit `models/image/aide/aide.py`: (i) restore the DCT/IDCT level-filter
   reconstruction in `_DCTPatchSelector`, (ii) feed native-res `to_tensor`+ImageNet (drop the 224
   CLIP-norm crop), (iii) convnext on 256 not 224 — per `ARCH_AUDIT_GAPMODELS.md` §3. CPU-test forward,
   then 4-GPU DeepSpeed on GenImage SDv1.4, 5 ep. Target: adm 58→~94, mAP →~97.
2. **Co-Spy — full data.** Same recipe as `retrain_cospy_genimage_gpu7.sh` but `root_dir=GenImage`
   (full 162k, not the 12k subset), 4-GPU, 3 ep. Optionally add a second generator (sdv1.5/wukong dirs)
   to lift adm/vqdm. Target GenImage-8 →~92 (CoSpyBench already 95.88).
3. **SPAI — full + more epochs.** `retrain_spai_cees.sbatch` with 4 GPUs (`--num_processes 4`), full
   dmid_real, 8–10 ep at 448; if precompute (a) is done, also try the native-arbitrary-res path
   (`MINIMUM_PATCHES=4`) for the last ~1 mAP. Target →~91.
4. **SAFE — full recipe.** Stage ProGAN, 4-GPU, train to the paper's epoch count. Target →~99.
5. **PGC** (optional) — longer cosine schedule on GenImage. Marginal.

## Order for tomorrow
1. Land infra fix (a) precompute + (b) eval DataLoader (1–2 h, unblocks speed).
2. Kick off AIDE code-fix + 4-GPU retrain (the only code change).
3. Parallel 4-GPU retrains: Co-Spy(full), SPAI(full), SAFE(ProGAN) on cn029/cn025/cn031.
4. Evals on cn023 GPUs 4&7 as checkpoints land; promote what closes; update
   `REPRODUCED_MODELS_BENCHMARKS.md` + READMEs + registry.

*Note: legally only cn023 GPUs 4 & 7 for us on cn023; everything else via the cees/cees6000/comcrim
partitions with the right `--account`.*
