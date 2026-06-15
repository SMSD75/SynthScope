Thanks for releasing the code, checkpoints, and CommGen15 — really nice work.

I'm trying to reproduce the released **ProGAN** checkpoint (`PGC_train_progan_ckpt.pth`) by training from scratch with your code, and I can't get close to its numbers on the ForenSynths hard generators, even running your `train.py` verbatim. I wanted to check whether the released checkpoint was trained with the exact recipe/data in the README, or with something additional.

## Setup
- Repo @ `0f3e97d` (latest), unmodified.
- DINOv2-Large weights from `facebook/dinov2-large`.
- Training data = the ProGAN training set (Wang et al. / CNN-Detection 20-class, the same set linked from `WisconsinAIVision/UniversalFakeDetect`), arranged as `UniversalFakeDetect/train/{0_real,1_fake}`.
- Exact command from README §3.2 (ProGAN), single GPU: `--batch_size 32 --accumulation_steps 4 --niter 100 --lr 5e-5 --optim adam --label_smoothing 0.1 ...`

## What works (so eval/data are fine on my side)
Loading your released `PGC_train_progan_ckpt.pth` reproduces the paper — e.g. on the three hard generators I get AUROC ≈ deepfake **0.965**, san **0.991**, seeingdark **0.875**.

## What I can't reproduce (training from scratch)
With the command above, the validation metric on those same generators plateaus far below the released checkpoint and shows no upward trend. Using **your own evaluator** (`--test_root` pointed at deepfake/san/seeingdark), the mean over the three is:

| epoch | mean AUC | mean AP | fake_acc |
|---|---|---|---|
| 2 | 0.782 | 0.803 | 0.116 |
| 4 | 0.807 | 0.808 | 0.059 |
| 6 | 0.713 | 0.751 | 0.112 |
| 8 | 0.742 | 0.788 | 0.162 |

The training loss reaches the label-smoothing floor (~0.1985) within a few epochs (the model fits the ProGAN training set), but it doesn't transfer to these generators — it keeps predicting almost everything as "real" (fake-accuracy stays ~6–16%). I also ran a full 100-epoch reproduction and it plateaus around AUROC ~0.76 on these three, vs ~0.92–0.94 from your released weights. The easy generators (progan/biggan/cyclegan/…) reproduce at ~0.99 in every run — the gap is specific to the hard-transfer generators.

## Questions
1. Was the released ProGAN checkpoint trained on exactly the ProGAN training set above, or with additional data / a different augmentation?
2. Are there any training details that differ from the README (number of epochs, LR schedule, additional preprocessing, a different DINOv2 revision)?
3. Could you share the training log (or the `best.pth` selection epoch/step) for the released ProGAN checkpoint, so I can compare trajectories?

## Possibly unrelated minor note
In `engine/trainer.py` / `train.py`, `scheduler = CosineAnnealingLR(T_max = niter * len(data_loader))` counts micro-batches, but `scheduler.step()` is only called once per optimizer step (every `accumulation_steps`), so the cosine traverses ~¼ of its range and the LR stays near-constant (~5e-5 → ~4.3e-5 over 100 epochs). I tested both the schedule as-written and a fully-annealed one; neither changed the outcome, so it doesn't seem to be the cause — just flagging in case it's unintended.

Thanks again — happy to share full configs/logs if useful.
