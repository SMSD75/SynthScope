# SynthScope Dataset Integrity Audit

- **Node:** ivi-cn023 (all data node-local under `/hddstore/ssalehi/SynthScope/`)
- **Date:** 2026-06-07
- **Disk:** `/hddstore` 14T total, 9.2T used, 4.9T avail (66%) — no space pressure.
- **Method:** recursive image counts (follow symlinks), `find -L <dir> -type l` for broken symlinks, PIL open+`.load()` with `LOAD_TRUNCATED_IMAGES=True` on ~20 random images/dataset. Counts are real, not estimated.

## Summary table

### Evaluation datasets

| # | Dataset | Root | Subsets found | Real total | Fake total | Broken links | Corrupt (spot) | Status |
|---|---------|------|--------------:|-----------:|-----------:|-------------:|---------------:|--------|
| 1 | CNNDetection / ForenSynths test | `…/CNNDetection/test` | 13 gens (expected 13) | 45169 | 45160 | 0 | 0/20 | OK |
| 2 | GenImage 8-gen test | `…/GenImage_full/test` | 8 gens (expected 8) | 16500 | 16500 | 0 | 0/20 | OK |
| 3 | UniversalFakeDetect | `…/universialfakedetection/test` | 10 dirs (8 fake gens + imagenet/laion reals) | 2000 | 8000 | 0 | 0/20 | OK (shared-reals layout) |
| 4 | AIGCDetectBenchmark | `…/AIGCDetectBenchmark/test` | 17 gens (expected 17) | 17000 | 17000 | 0 | 0/20 | OK |
| 5 | CoSpyBench | `…/CoSpyBench/test` | 110 real gen-subsets + `_real_pool` | see below | see below | 0 | 0/20 | OK — reals RESTORED |
| 6 | AIGIBench | `…/AIGIBench/test` | 25 gens | 106502 | 106300 | 0 | 0/20 | OK (minor imbalance) |
| 7 | DiffusionForensics | `…/DiffusionForensics/test` | 8 gens | 8000 | 7768 | 0 | 0/20 | OK (ddpm short) |
| 8 | Diffusion1kStep | `…/Diffusion1kStep/test` | 5 gens | 5000 | 5000 | 0 | 0/20 | OK |
| 9 | CommGen15 | `…/CommGen15/test` | 15 gens | 12983 | 12983 | 0 | 0/20 | OK (3 small gens) |
| 10 | freqnet / GANGen | `…/freqnet/test` | 9 GAN gens | 18000 | 18000 | 0 | 0/20 | OK |

### Training datasets

| # | Dataset | Root (path consumed) | Real | Fake | Broken links | Corrupt | Status |
|---|---------|----------------------|-----:|-----:|-------------:|---------|--------|
| 11 | GenImage SD-v1.4 train | `…/GenImage/train` (real/, fake/) | 162000 | 162000 | 0 | 0/20 | OK, balanced |
| 12 | CNNDetection ProGAN train (20-class) | `…/CNNDetection/train` | 360059 | 360060 | 0 | 0/20 | OK, balanced (val 4000/4000) |
| 13a | SPAI `dmid_real` (used by latest retrain) | `…/spai_train/dmid_real` | 81000 (tr) | 81000 (tr) | 0 | 0/20 | OK, balanced |
| 13b | SPAI `staged` (referenced by default config) | `…/spai_train/staged` | **0 (20000 dead)** | 20000 | **22000** | fakes OK | **BROKEN — zero reals** |
| 14 | Co-Spy train (`cospy_subset`) | `…/spai_train/cospy_subset/train` | 60000 | 60000 | 0 | links resolve | OK, balanced (ProGAN-sourced) |

## Per-dataset detail

### 1. CNNDetection / ForenSynths test — OK
13 generators, all with paired reals/fakes. progan/cyclegan/stylegan/stylegan2 use the nested `<gen>/<class>/{0_real,1_fake}` layout and were counted correctly. Per-gen counts: biggan 2000/2000, crn 6382/6382, cyclegan 1321/1321, deepfake 2707/2698, gaugan 5000/5000, imle 6382/6382, progan 4000/4000, san 219/219, seeingdark 180/180, stargan 1999/1999, stylegan 5991/5991, stylegan2 7988/7988, whichfaceisreal 1000/1000. Resolutions native/varied (256², 512×256, 512×384) — fine for NPR/SPAI.

### 2. GenImage 8-gen test — OK
adm, biggan, glide, midjourney, sdv1.4, sdv1.5, vqdm, wukong — each `{0_real,1_fake}` balanced. sdv1.4 has 6000/6000, the rest 1500/1500.

### 3. UniversalFakeDetect — OK (by-design layout)
Standard UniversalFakeDetect layout: the 8 fake generators (dalle, glide_100_10, glide_100_27, glide_50_27, guided, ldm_100, ldm_200, ldm_200_cfg) ship **only `1_fake`** (1000 each); reals live in shared pools `imagenet/0_real` (1000) and `laion/0_real` (1000). The SynthScope loader pairs fakes against these shared reals, so the lopsided 2000-real/8000-fake on-disk count is expected, not a defect.

### 4. AIGCDetectBenchmark — OK
17 generators, every one balanced 1000/1000.

### 5. CoSpyBench — OK, reals fully RESTORED (prior concern resolved)
- `test/` has 111 entries: 110 real-source×generator subsets + 1 `_real_pool` backing store.
- **110/110 generator subsets have BOTH `0_real`(>0) and `1_fake`(>0).** 0 subsets with only fakes, 0 with only reals.
- **All 5 real sources are present on disk:** cc3m (22 subsets, reals 2000 each), flickr (22, reals 2500), mscoco (22, reals 2500), sbu (22, reals 2000), textcaps (22, reals 2000). Each subset has 5000 fakes.
- The prior note "only cc3m's reals survived a disk rebuild" is **NO LONGER TRUE** — flickr/mscoco reals were rebuilt on Jun 5 (`_real_pool/{flickr,mscoco}` timestamped Jun 5). Reals are real files / hardlinks (link count 23, shared via `_real_pool`), not symlinks. `find -L … -type l` = 0 broken.

### 6. AIGIBench — OK (minor)
25 generators, ~4500/4500 each. Minor under-fill: FaceSwap 4500/4400, InSwap 4500/4400, DALLE-3 4002/4000. Totals 106502/106300. Not metric-relevant.

### 7. DiffusionForensics — OK (one short generator)
8 gens balanced 1000/1000 **except ddpm = 1000 real / 768 fake**. Minor.

### 8. Diffusion1kStep — OK
5 gens (DALLE, ddpm, guided-diffusion, improved-diffusion, midjourney), all 1000/1000.

### 9. CommGen15 — OK (3 small gens, balanced)
15 gens, balanced. Three have fewer samples but balanced: ChatGPT 355/355, Hunyuan 240/240, NanoBanana 388/388. (Note: dir lists 15 names; "Lexica/Veo" etc. all present.)

### 10. freqnet / GANGen — OK
9 GAN generators (AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN), all 2000/2000.

### 11. GenImage SD-v1.4 train — OK
`GenImage/train/{real,fake}` = 162000 / 162000, no symlinks, no corruption. Paired `GenImage/test` = 6000/6000.

### 12. CNNDetection / ProGAN train — OK
20 object classes, `<class>/{0_real,1_fake}`. Train 360059 real / 360060 fake (balanced); val 360060? → val 4000/4000. No broken symlinks, no corruption.

### 13. SPAI training set — MIXED (one broken layout, one good layout)
Two parallel real sources exist:
- **`spai_train/staged/{train,test}/ldm/{0_real,1_fake}`** — the ImageFolder layout the **default config `conf/dataset/image/spai_ldm.yaml` points at** (`root_dir=…/spai_train/staged`). `0_real` is **20000 (train) + 2000 (test) BROKEN symlinks** → `/hddstore/ssalehi/ocvlm_datasets/coco_raw/images/train2017/…`, a directory that **no longer exists** on the node. `1_fake` (LDM, 256², 20000+2000) resolves fine. **Net: training via the default config sees ZERO reals — exactly the prior bug.** 22000 broken symlinks total.
- **`spai_train/dmid_real/{train,val,test}/ldm/{0_real,1_fake}`** — the Jun 5 rebuild. train 81000/81000, val 4500/4500, test 4500/4500, **all symlinks resolve** (reals → `/ssdstore/mdorkenw/DETECTRON2_DATASETS/coco/train2017/`, which *is* reachable on cn023; fakes → the local dmid LDM set). 0 broken.
- The **latest retrain script `repro/retrain_spai_cees.sbatch` overrides the config** to `DatasetConfig.root_dir=$DATA/spai_train/dmid_real`, so the *current* retrain uses the good data. The hazard is the stale default config.
- Repair source for `staged` reals is available: the exact broken target `000000405271.jpg` exists at `/hddstore/ssalehi/Paneco_dataset/coco/dataset/images/train2017/` (118287 COCO train2017 images).
- `dmid` LDM fakes (the fake source) are all 256², intact: 7 train subsets (45000+13500+13500+18000+45000+22500+22500 = 180000) + valid splits, 0 broken, 0 corrupt.

### 14. Co-Spy training set — OK
`spai_train/cospy_subset/train/<20 ProGAN classes>/{0_real,1_fake}` = 60000 real / 60000 fake. All symlinks resolve into the intact `CNNDetection/train` ProGAN data. Note for faithfulness: Co-Spy here is trained on **ProGAN-sourced** images (symlinks into CNNDetection), not DRCT — consistent with the gap-closing notes but not the paper's exact DRCT source.

## Prioritized problems

| Pri | Problem | Affected model / benchmark | Detail & fix |
|-----|---------|----------------------------|--------------|
| **P1** | **`spai_train/staged/.../0_real` = 22000 BROKEN symlinks (zero reals)**; default SPAI config `conf/dataset/image/spai_ldm.yaml` still points `root_dir` at this `staged` dir. | **SPAI from-scratch / gap-closing retrain** | Any SPAI run using the plain config trains on fakes-only reals → collapse. Latest sbatch already overrides to `dmid_real` (good). **Fix:** repoint the config to `…/spai_train/dmid_real`, OR re-link `staged/.../0_real` to `/hddstore/ssalehi/Paneco_dataset/coco/dataset/images/train2017/` (targets verified present). |
| P3 | DiffusionForensics **ddpm** has 768 fakes vs 1000 reals (imbalanced single generator). | DiffusionForensics eval (any model) | Minor per-gen AUC noise only; total set still 8000/7768. Optionally re-download ddpm fakes. |
| P3 | AIGIBench FaceSwap/InSwap 4400 fakes (vs 4500), DALLE-3 4000 fakes (vs 4002). | AIGIBench eval | Negligible (<2.5% short on 3 of 25 gens). |
| P4 (note) | `dmid_real` reals resolve via `/ssdstore/mdorkenw/…` (works on cn023). If `dmid_real` is rsynced to another node (cees), those symlinks may not resolve there. | SPAI cross-node retrain | The retrain sbatch rsyncs `dmid_real`; verify reals materialize on the target node, else rsync with `-L` (deref) or restage reals from Paneco COCO. |
| P4 (note) | UniversalFakeDetect uses shared `imagenet`/`laion` reals (no per-gen reals). | UniversalFakeDetect eval | By design; confirm the SynthScope loader pairs shared reals (it does) — not a defect. |
| P4 (note) | Co-Spy trained on ProGAN (symlinked CNNDetection), not DRCT. | Co-Spy faithfulness | Data-source mismatch vs paper, not an integrity error. |

## Bottom line
No corrupted images and no missing generators in any **evaluation** set; all eval sets are class-paired and openable. **CoSpyBench is fully restored** (all 5 real sources, 110/110 paired subsets) — the earlier "only cc3m survived" worry is resolved. The single real integrity defect is **P1: the `spai_train/staged` real symlinks are 100% broken (22000 dead links → deleted `ocvlm_datasets/coco_raw`)**, and the default SPAI config still points there. The healthy `dmid_real` (81000/81000) exists and the latest retrain script already uses it, but the stale config is a live foot-gun and the prime suspect for any SPAI under-performance run through the default path.
