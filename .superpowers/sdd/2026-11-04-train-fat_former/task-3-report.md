# Task 3 report: FatFormer trained-checkpoint evaluation (status: DONE)

## Where it ran
The IVI jobs 281678/281680 never started and were cancelled. With the user's approval the evaluation ran on the DAS6 cluster
(job 66626, node409, RTX A6000) at commit a2a4d2b (clean worktree), same protocol and tags as the planned script.
Log: /ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/logs/ffevt_das6_66626.out. The checkpoint/data paths in the JSONs are DAS6 cache copies.
Both checkpoints load strictly into the served spec (719 tensors, 0 missing, 0 unexpected, num_vit_adapter=3); no --model-config needed.
model_best = epoch 0 (val acc 1.0000, strict > never replaced it); model_last = epoch 24. Headline = model_last (final model, as the paper/official code).

## Results (mAP / acc, from repro/bench_results/fat_former_trained_*.json)
| | GANs-8 | UFD DMs-8 |
|---|---|---|
| paper | 99.7 / 98.4 | DMs-10: 98.8 / 95.0 (same-8 mean of the paper's per-subset numbers: 98.54 / 93.80) |
| official ckpt in SynthScope | 99.70 / 98.39 | 98.60 / 93.72 |
| trained, last (epoch 24) | 99.56 / 97.29 | 98.88 / 92.09 |
| trained, best (epoch 0) | 99.66 / 95.71 | 98.67 / 90.89 |

GANs-8 per-subset acc (last): ProGAN 99.90, StyleGAN 94.73, StyleGAN2 94.08, BigGAN 99.45, CycleGAN 99.62, StarGAN 99.97, GauGAN 99.59, Deepfake 90.96.
Gap vs paper concentrated in StyleGAN (-2.5), StyleGAN2 (-4.7), Deepfake (-2.2). Tolerance rule in build_repro_full (2 points) marks the GANs row reproduced.
DMs: not like-for-like (UFD-8 vs the paper's 10 subsets, PNDM/VQ-Diffusion unavailable); on the same 8 subsets the trained model is +0.3 mAP, -1.7 acc vs the paper; row stays "official", not "reproduced".

## Files changed
MASTER_RESULTS.jsonl already had the 4 tags (4 lines, one per tag, written by bench_eval on DAS6); bench_results JSONs (4);
models/image/fat_former/README.md; repro/benchmark_specs/fat_former.json (synthscope_train entries); interface/backend/build_repro_full.py (VAL);
interface/backend/repro_full.json regenerated (only the two fat_former rows changed); repro/RESULTS_SUMMARY.md; README.md reproduction row.
Checkpoints not uploaded.

## Concerns
- Accuracy at the 0.5 threshold is 1.1 below the paper on GANs-8 (mAP -0.14); 10-subset DM mean cannot be formed.
- Training val accuracy saturates at 1.0 from epoch 0, so val-based selection is uninformative.
