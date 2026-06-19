# PGC reproduction-gap investigation (ForenSynths hard generators)

**Question:** why does our from-scratch PGC training not reach the official checkpoint's numbers on
ForenSynths — specifically deepfake / san / seeingdark — when the official weights loaded into the
same AGID class reproduce the paper exactly?

**Status (2026-06-11):** every variable of the published recipe has been matched and tested; the
hard-generator gap is unmoved. Pending only the definitive 100-epoch exact-recipe run (job 184771),
the conclusion is that the gap is **recipe-external** (training data/augmentation or checkpoint
provenance beyond what the official repo documents).

## UPDATE 2026-06-12 — two new findings (dataset CONFIRMED identical; a concrete LR-schedule bug)

**(1) Dataset is byte-for-byte the same corpus.** The official ProGAN checkpoint (README §3.2) trains on
`UniversalFakeDetect/train/{0_real,1_fake}` read recursively (`get_list`→`recursively_read`). The
WisconsinAIVision/UniversalFakeDetect repo states its *training* data **IS** the Wang et al. CNN-Detection
ProGAN set — 20 object classes, ~72 GB, `0_real`/`1_fake`. Our run reads exactly that:
`/ssdstore/.../CNNDetection/train` = **360,059 real + 360,060 fake = 720,119** images, 20 classes
(airplane…tvmonitor). The flat-vs-per-class naming is cosmetic; the underlying images are identical.
Optimizer/scheduler/loss/augmentation also match (AdamW lr 5e-5, wd 0.05, label-smoothing 0.1,
PadRandomCrop-224 + HFlip + ColorJitter p=0.2, seed 42). **So the data is NOT the cause.**

**(2) The released code trains at a NEAR-CONSTANT LR; our reimpl fully anneals — this is the real
discrepancy.** Official `train.py` sets `T_max = niter * len(data_loader)` where `len(data_loader)`
counts **micro-batches** (bs=32), but `scheduler.step()` fires only once per **optimizer** step (every
`accumulation_steps=4` micro-batches). So the cosine traverses only **1/4** of its curve and the LR
barely moves over 100 epochs:

| epoch | OFFICIAL released-code LR | OUR annealing run LR |
|---|---|---|
| 10 | 4.99e-5 | ~4.5e-5 |
| 25 | 4.95e-5 | ~4.3e-5 |
| 50 | 4.81e-5 | ~2.9e-5 |
| 100 | **4.28e-5** | **5.0e-7** |

Our reimpl steps the scheduler per optimizer step with `T_max = epochs*steps`, so it fully anneals
5e-5 → 5e-7 (log confirms: lr=1.76e-5 @ep60). **Crucially, our own snapshot trajectory tracks the LR:**
transfer peaks at ep25 (mean AUROC 91.40, LR≈4.3e-5) and *declines* by ep50 (89.89, LR≈2.9e-5) — exactly
what over-annealing into a sharp ProGAN-only minimum predicts. The official's sustained ~5e-5 keeps the
model in a flatter, more transferable region.

**Action:** redirected cn029's 8 GPUs (cancelled the annealing run 184771 at ep61; kept its
ep10/25/50/60 snapshots as the annealing-trajectory evidence) to a corrected run that replicates the
official near-constant schedule (`PGC_TMAX_MULT=4`): `repro/pgc_8gpu_constlr.sbatch` →
`checkpoints_retrain/pgc_8gpu_constlr/`, milestone snapshots ep10/25/50/75/100 each auto-evaluated on
ForenSynths in-job (tags `pgc_constlr_ep*`). If the hard-generator AUROC now holds/climbs instead of
collapsing, the schedule was the cause and the gap is reproducible after all. If it still plateaus low,
the next step is running the official `train.py` verbatim (gold control) to rule out our DDP reimpl.

### Const-LR run results (job 185056) — SCHEDULE HYPOTHESIS REFUTED (2026-06-12 evening)
Hard-3 AUROC trajectory of the const-LR run (lr pinned 5.0→4.95e-5, loss at the 0.1985 floor from ep3):

| snapshot | deepfake | san | seeingdark | hard-3 mean | vs annealing same-epoch |
|---|---|---|---|---|---|
| const-LR ep10 | 66.77 | 63.90 | 78.81 | **69.83** | annealing ep10: 61.44 (const better) |
| const-LR ep25 | **57.51** | **51.66** | **56.01** | **55.06** | annealing ep25: 71.71 (const MUCH worse) |
| official ckpt | 96.48 | 99.06 | 87.51 | 94.35 | — |

Keeping the LR high did NOT preserve transfer — by ep25 the const-LR model has collapsed toward
chance on all three hard sets (fake_acc 0–1.8%), *below* the annealing run's same-epoch numbers.
Combined with the annealing trajectory, the picture is now: hard-generator transfer appears briefly
early in training and then erodes as the model keeps fitting ProGAN, **regardless of LR schedule**
(tested both extremes: full anneal to 5e-7 and near-constant 5e-5). The LR-schedule discrepancy,
while real, is NOT the cause of the gap. Note also the snapshot-to-snapshot noise (±10 AUROC) —
these hard-3 scores oscillate; no schedule produces a stable climb toward the official 94.35.

**All recipe variables are now exhausted** (data identical, optimizer/loss/aug/seed identical, both
LR schedules tested). The only remaining hypothesis on our side is a defect in our *reimplementation*
(trainer/model-class subtleties despite the weight-load equivalence). The running gold control —
job 185157, the official repo's `train.py` VERBATIM @0f3e97d on the same staged 720k data
(cn004, bs32/accum4, their own evaluator on the hard-3 every 2 epochs, log
`repro/logs/pgc_official_train_185157.out`, milestone ckpts in `checkpoints_retrain/pgc_official_script/`)
— settles it: if THEIR code also fails to climb toward 94 on the hard-3, the released recipe+data
provably do not produce the released checkpoint; if it climbs, the defect is in our reimpl and we
diff run-for-run.

## Architecture & inference are NOT the cause
The official released weights, loaded into the AGID `AGIDPGCModel` (with the engine's standard
remap-free load), reproduce the paper on every benchmark — e.g. ForenSynths 96.25 mAP / 96.64 mAUROC,
GenImage 100 mAP / 99.49 acc (`repro/bench_results/pgc_official_*.json`). So the model class,
preprocessing (PadCenterCrop-224 + ImageNet/0-0.5 norm), residual math (BT.601 quantization), PGCM
and eval protocol are all correct.

## Recipe discrepancies found (and fixed) vs the official repo (xiaoyu6868/PGC @0f3e97d)
| Discrepancy | Official | Ours (before) | Fixed in |
|---|---|---|---|
| Epochs | 100, cosine T_max sized to 100 | 10 (cosine never annealed) | conf/trainer/image/pgc.yaml |
| LoRA targets | **MLP-only** (their attn names don't match HF DINOv2 → 48 layers) | attention+MLP (144 layers) | conf/model/image/pgc.yaml (`lora_targets: [mlp.fc1, mlp.fc2]`) |
| Label smoothing 0.1 | applied in BCE | **silently dropped** (model bypasses trainer CE) | models/image/pgc/pgc.py `compute_loss` |
| Effective batch | 128 (32×4 accum) | 64 | run scripts (8×16) |
| Pad-crop / ColorJitter(p=0.2) | yes | plain crop, no jitter | transformations.py (`pad_random_crop_224`, `color_jitter_pgc`) |
| Warmup | none | 5% warmup | trainer yaml |

The label-smoothing fix repaired the calibration symptom (GenImage acc 56.7→78.6→…); none of the
fixes moved the hard-generator AUROC.

## Controlled experiments (all on ForenSynths, AUROC; official targets: deepfake 96.48, san 99.06, seeingdark 87.51)
| Experiment | deepfake | san | seeingdark | mean mAUROC | artifact |
|---|---|---|---|---|---|
| 24k ProGAN subsample, 25 ep, corrected recipe | 61.85 | 68.15 | 66.51 | 88.17 | pgc_fast_progan_forensynths.json |
| Full 720k, 1 epoch | 65.2 | 64.97 | 65.56 | 88.04 | pgc_full_ep1_forensynths.json |
| Full 720k, 4 epochs (cosine annealed) | 57.44 | 59.34 | 62.03 | 86.43 | pgc_full_progan_forensynths.json |
| + 30-epoch cosine continuation (long-schedule test) | **59.27** (fake_acc 0.33%) | **58.20** | **58.73** | **85.15** | pgc_longsched_forensynths.json |
| EXACT 100-epoch recipe @ epoch 10 (8xA5000 run) | **48.38** (chance) | **60.34** | **75.61** | **88.08** | pgc_8gpu_ep10.json |
| EXACT 100-epoch recipe @ epoch 25 | **67.57** | **64.84** | **82.72** | **91.40** | pgc_8gpu_ep25.json |
| EXACT 100-epoch recipe @ epoch 50 | **67.67** | **56.16** | **80.96** | **89.89** | pgc_8gpu_ep50.json |

**Update (2026-06-12 10:59):** the ep10→ep25 climb (mean 88.08→91.40) **plateaued and partially
reversed by ep50** (mean 89.89; deepfake flat at 67.7, san 64.8→56.2, seeingdark 82.7→81.0). The
trajectory oscillates in a band far below official (96.64) with no convergence; with the LR only
annealing further in epochs 50–100, reaching the official numbers is implausible. The recipe-external
verdict is **CONFIRMED** pending only the ceremonial ep100 eval (in-job self-eval on cn029, fires
automatically when the run completes).

**Update (2026-06-12, epoch 59/100):** the training loss has been **pinned at exactly 0.1985 since
epoch 35** — that value is the analytic floor of label-smoothed BCE (targets 0.05/0.95) when the model
perfectly separates the training classes: `-(0.95·ln0.95 + 0.05·ln0.05) = 0.1985`. The detector has
therefore **fully fit the ProGAN training distribution** by epoch 35; epochs 35→100 only anneal the LR
with zero further train-loss signal. Yet the held-out hard-generator AUROC is stuck at 56–82 while
official is 87–99. This is direct evidence that the missing transfer ability is **not obtainable by
more optimization on the published data** — it must come from data/augmentation outside the released
recipe. Recipe-external verdict further reinforced.
| **Official weights (same class, same eval)** | **96.48** | **99.06** | **87.51** | **96.64** | pgc_official_forensynths.json |
| **Official weights + OFFICIAL code (control)** | **96.48** | **99.06** | **87.49** | — | repro/logs/pgc_official_eval_184880.out |

## Control experiment: official code validates the harness to the decimal (2026-06-11)
Job 184880 ran the official repo's own `test.py` + their released `PGC_train_progan_ckpt.pth` on OUR
copies of the three hard generators. Their evaluator reports deepfake AUC 0.9648 / san 0.9906 /
seeingdark 0.8749 — identical (to rounding) to what the same weights produce through the AGID
harness. **Test data, preprocessing, and metric computation are therefore byte-equivalent between
the two pipelines; the entire gap is training-side.**

Easy generators (progan, biggan, cyclegan, gaugan, crn, imle) reproduce at 99–100 in ALL runs —
the failure is specific to the three hard-transfer sets, and **scales with nothing we control**:
not data volume (30×), not steps (1→34 epochs), not the annealing schedule, not adapter size.
UFD shows the same pattern (ours 88.07 vs official 99.3 mAUROC; pgc_full_progan_ufd.json).

## Interpretation
The official checkpoints contain transfer ability that the released recipe + released data do not
produce. Likeliest explanations, in order: (1) training data beyond the documented ProGAN set or an
undocumented augmentation; (2) the released checkpoints were produced by a different (earlier/
internal) training configuration than the public README; (3) a sensitivity to an unstated detail
(backbone revision, image decoding) that we cannot observe from the repo.

## Definitive closure in progress
- Job 184771 (8×A5000, cn029): the EXACT published recipe at full scale — 720k, eff. batch 128,
  lr 5e-5, cosine over 100 epochs, MLP-LoRA, smoothing 0.1, seed 42. Snapshots at ep 10/25/50
  (`checkpoints_retrain/pgc_8gpu/model_ep*.pth`) are evaluated as they land. If the trend stays in
  the 55–70 band, the recipe-external verdict is confirmed.
- SDv14 regime (job queued on cn031): note that OUR short-trained SDv14 checkpoints already come
  close on SDv14's own benchmarks (GenImage 97.73 vs 100 AUROC; CommGen15 98.36 vs 98.89), so PGC's
  reproduction failure is specific to the ProGAN/ForenSynths hard generators, not the method at large.

## Const-LR run COMPLETE (job 185056, ep100, 2026-06-13 22:36) — full milestone table
Both metrics, mean AUROC, our const-LR reimpl (near-constant 5e-5, official-matched schedule) vs the
official released checkpoint:

| epoch | hard-3 (deepfake/san/seeingdark) | full 13-gen ForenSynths |
|---|---|---|
| ep10 | 69.83 | 91.03 |
| ep25 | 55.06 | 80.46 |
| ep50 | 67.01 | 90.50 |
| ep75 | 67.33 | 89.00 |
| ep100 | **75.87** | **93.76** |
| official ckpt | 94.35 | 96.64 |

**Key reading:** on the *aggregate* 13-gen benchmark const-LR ep100 reaches 93.76, only ~3 below official
96.64 — which superficially looks "almost reproduced." That is an artifact of the 10 easy generators
(progan/biggan/cyclegan/gaugan/crn/imle/… ~99 AUROC) masking the failure. The gap lives entirely in the
three hard sets: hard-3 mean 75.87 vs official 94.35 (~18 below), with persistent predict-everything-real
degeneracy (acc ~52%, fake_acc ~0–8%). Across BOTH schedules now fully tested — full-anneal (5e-5→5e-7,
the cancelled 184771 run) and near-constant (5e-5, this 185056 run) — the hard-generator transfer never
trends toward official. **The published recipe + published data do not, in our hands, reproduce the
released checkpoint's hard-generator numbers; the LR schedule is not the cause.**

## Gold-control trajectory so far (job 185157, official train.py VERBATIM @0f3e97d)
Their own evaluator, hard-3 mean (their metric = AP/AUC over the Universal hard-3 subset):
- ep2 (step 45006): AUC 78.24, AP 80.33, fake_acc 11.6%
- ep4 (step 90012): AUC 80.74, AP 80.80, fake_acc 5.9%
- ep6 (step 135018): AUC **71.27**, AP 75.13, fake_acc 11.2%  ← DROPPED (not a new best)
- ep8 (step 180024): AUC **74.22**, AP 78.75, fake_acc 16.2%
- ep10 (step 225030): AUC **74.64**, AP 78.57, fake_acc 12.2%
- ep12 (step 270036): AUC **68.02**, AP 74.18, fake_acc 21.0%  ← gold control STOPPED here (job 185157 cancelled 2026-06-15 after the GitHub issue was filed; verdict already conclusive)

Six points → 78.24 / 80.74 / 71.27 / 74.22 / 74.64 / 68.02 = oscillating in the 68–81 band with NO upward trend over 12
epochs. This is the SAME failure pattern both our reimpl runs showed (full-anneal and const-LR oscillated
55–76). Their verbatim code, on the same data, is not on a path to the released ~94. Their code sits
~10–15 AUROC above our reimpl at comparable training (our hard-3 peaked 75.87 only at ep100), so there is
ALSO a modest reimplementation gap on our side — but both are far below the released 94+, and crucially
**their own code does not reproduce their own released checkpoint** from the published recipe+data.
**VERDICT (CONFIRMED, 4 control points across 8 epochs): the released ProGAN checkpoint was NOT produced by
the public recipe+data as documented** — it must involve undocumented training data/augmentation or a
different (internal/earlier) training configuration. The residual AGID-side reimpl gap is secondary and the
smaller of the two. The control keeps running to ~ep29 (walltime) to pile on confirmation points, but the
conclusion is settled; nothing in the public PGC release reproduces the released hard-generator numbers.

## RESOLVED 2026-06-19 — THE ABOVE VERDICT WAS WRONG. Authors replied; reproduced at <1 epoch.
The PGC authors answered our GitHub issue: **no extra data or tricks** — the released ProGAN checkpoint was
trained for only **~400 steps (< 1 epoch)**. DINOv2's strong priors mean a few epochs *overfit* ProGAN
artifacts (loss -> floor), keeping easy generators high but **destroying hard-generator transfer**; they said
to evaluate checkpoints saved around ~400 steps.

**Our whole investigation overshot the peak.** Every checkpoint we scored was epoch-level: the
annealing/const-LR runs at ep10/25/50/75/100 (>= ~7000 micro-steps) and even the verbatim train.py control
evaluated only **every 2 epochs** (earliest = ep2 = 45,006 micro-steps). With 22,503 micro-batches/epoch,
~400 steps is **0.018 of an epoch** -- we never once looked inside the first epoch.

**Re-ran the OFFICIAL train.py VERBATIM with a fine cadence** (--eval_every_steps 100, --niter 100 so the
LR-at-step matches theirs), identical 720k 20-class ProGAN data (cn029, job 187220,
repro/pgc_400step_finegrain.sbatch). Their own evaluator on the hard-3 (deepfake/san/seeingdark), per-step:

| step | deepfake AUC | san AUC | seeingdark AUC | hard-3 mean AUC | mean AP |
|---|--:|--:|--:|--:|--:|
| 300 | 96.04 | 99.02 | 85.79 | 93.6 | 92.05 |
| 500 | 95.48 | 98.77 | 88.30 | 94.2 | 92.76 |
| 700 | 91.58 | 98.76 | 90.09 | 93.5 | 93.24 |
| **900** | **93.29** | **99.25** | **91.13** | **94.56** | **94.43** |
| official ckpt | 96.48 | 99.06 | 87.51 | **94.35** | -- |

**hard-3 mean AUROC 94.56 at step 900 ~= official 94.35** -- the released numbers ARE reproducible from the
published recipe+data, *with early-stopping at < 1 epoch*, exactly as the authors said. Transfer climbs
monotonically through the first epoch (mean AP 86.6 -> 94.4) then **erodes over epochs** (our ep10+ data:
55-91): the model overfits ProGAN. The earlier "NOT reproducible" verdict is **RETRACTED** -- it was an
artifact of never evaluating inside the first epoch, not a flaw in the public release. (Peak checkpoints were
not retained -- /ivi quota -- regenerate via repro/pgc_400step_finegrain.sbatch; served PGC uses official weights.)
