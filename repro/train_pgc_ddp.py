"""Multi-GPU (accelerate/DDP) PGC trainer for cees runs.

Same corrected recipe as train_pgc_fast.py (MLP-only LoRA, label smoothing 0.1), but DDP-ready:
- loss is computed via model(images) (NOT model.compute_loss) so DDP's forward hooks fire and grads sync
- only the main process logs + saves; checkpoint is a raw state_dict on shared /ivi/zfs (engine-loadable)
- data root is configurable (PGC_DATA_ROOT) so the cees job points at node-local extracted data
Launch: accelerate launch --num_processes N repro/train_pgc_ddp.py
"""
import os, sys, glob, time, random
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from accelerate import Accelerator

from config_utils import compose_config
from models.image.pgc.pgc import SynthScopePGCModel

DATA_ROOT = os.environ.get("PGC_DATA_ROOT", "/hddstore/ssalehi/AGID/CNNDetection/train")
N = int(os.environ.get("PGC_N", "400000"))
EPOCHS = int(os.environ.get("PGC_EPOCHS", "30"))
BS = int(os.environ.get("PGC_BS", "64"))            # per-GPU; 64 fits a 24GB card for DINOv2-large bf16
OUT = os.environ.get("PGC_OUT", "/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_ddp/model.pth")
SMOOTH = float(os.environ.get("PGC_SMOOTH", "0.1"))
# The official PGC train.py passes T_max = niter*len(micro_batches) but steps the scheduler once per
# OPTIMIZER step (every accumulation_steps=4 micro-batches), so its cosine traverses only 1/4 and the
# LR stays near-constant (~5e-5 -> 4.28e-5 over 100 ep). Set PGC_TMAX_MULT=4 to replicate that exact
# schedule; the default 1.0 fully anneals (5e-5 -> 5e-7) as our earlier runs did.
TMAX_MULT = float(os.environ.get("PGC_TMAX_MULT", "1.0"))
MILESTONES = {int(x) for x in os.environ.get("PGC_MILESTONES", "10,25,50,75,100").split(",") if x.strip()}
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".JPEG")


def collect():
    """Multiple data roots separated by ':' (joint ProGAN+SDv14 training = official 3rd checkpoint).
    Supports both layouts: CNNDetection (**/0_real|1_fake/*) and GenImage (real/ + fake/)."""
    reals, fakes = [], []
    for root in DATA_ROOT.split(":"):
        if os.path.isdir(os.path.join(root, "real")) and os.path.isdir(os.path.join(root, "fake")):
            reals += [f for f in glob.glob(root + "/real/*") if f.endswith(EXT)]
            fakes += [f for f in glob.glob(root + "/fake/*") if f.endswith(EXT)]
        else:
            reals += [f for f in glob.glob(root + "/**/0_real/*", recursive=True) if f.endswith(EXT)]
            fakes += [f for f in glob.glob(root + "/**/1_fake/*", recursive=True) if f.endswith(EXT)]
    random.seed(42)                                   # official seed
    random.shuffle(reals); random.shuffle(fakes)
    reals, fakes = reals[:N], fakes[:N]
    items = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
    random.shuffle(items)
    return items


class PGCData(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = v2.Compose([
            v2.RandomCrop(224, pad_if_needed=True), v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.2),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMNET_MEAN, IMNET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        for _ in range(4):
            p, y = self.items[i]
            try:
                return self.tf(Image.open(p).convert("RGB")), y
            except Exception:
                i = random.randrange(len(self.items))
        return torch.zeros(3, 224, 224), 0


def main():
    acc = Accelerator()
    items = collect()
    dl = DataLoader(PGCData(items), batch_size=BS, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    cfg = compose_config(["model@ModelConfig=image/pgc", "trainer@TrainerConfig=image/pgc",
                          "dataset@DatasetConfig=image/cnndetection"])
    model = SynthScopePGCModel(cfg)
    init = os.environ.get("PGC_INIT", "")
    if init and os.path.isfile(init):                 # resume from a prior epoch checkpoint
        sd = torch.load(init, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
    eff_bs = BS * acc.num_processes
    # EXACT official recipe: lr 5e-5 at effective batch 128 (8 GPUs x bs 16); scale only if larger.
    lr = 5e-5 * max(1.0, eff_bs / 128)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(EPOCHS * len(dl) * TMAX_MULT), eta_min=lr * 0.01)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)

    if acc.is_main_process:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        print(f"[ddp] gpus={acc.num_processes} per_gpu_bs={BS} eff_bs={eff_bs} lr={lr:.2e} "
              f"items={len(items)} steps/epoch={len(dl)} epochs={EPOCHS} data={DATA_ROOT} "
              f"tmax_mult={TMAX_MULT} milestones={sorted(MILESTONES)}", flush=True)

    model.train()
    for ep in range(EPOCHS):
        t0 = time.time(); tot = 0.0; n = 0
        for images, labels in dl:
            with acc.autocast():
                logits = model(images)               # DDP forward (grads sync) — NOT compute_loss
                target = labels.float()
                if SMOOTH > 0:
                    target = target * (1 - SMOOTH) + 0.5 * SMOOTH
                loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += acc.gather(loss.detach()).mean().item(); n += 1
        if acc.is_main_process:
            print(f"[epoch {ep+1}/{EPOCHS}] loss={tot/max(n,1):.4f} time={time.time()-t0:.0f}s "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            acc.wait_for_everyone()
            sd = acc.unwrap_model(model).state_dict()
            torch.save(sd, OUT)
            if (ep + 1) in MILESTONES:
                snap = os.path.join(os.path.dirname(OUT), f"model_ep{ep+1}.pth")
                torch.save(sd, snap)
                print(f"[snapshot] saved {snap}", flush=True)
        else:
            acc.wait_for_everyone()
    if acc.is_main_process:
        print(f"[done] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
