"""Fast standalone PGC trainer for the official-reproduction attempt.

Trains the AGID PGC model (corrected recipe: MLP-only LoRA, label smoothing 0.1) on a SUBSAMPLE of a
data regime with parallel decode + bf16 + corrupt-image skipping — no per-epoch test eval. Saves a raw
state_dict that the AGID engine / bench_eval load directly. Prints per-epoch time so we can extrapolate.

env: IDEO unused. Knobs: PGC_REGIME(progan|sdv14), PGC_N(per class), PGC_EPOCHS, PGC_OUT, CUDA_VISIBLE_DEVICES.
"""
import os, sys, glob, time, random
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True            # tolerate the ProGAN PNGs' truncated/odd zlib

from config_utils import compose_config
from models.image.pgc.pgc import AGIDPGCModel

REGIME = os.environ.get("PGC_REGIME", "progan")
N = int(os.environ.get("PGC_N", "12000"))
EPOCHS = int(os.environ.get("PGC_EPOCHS", "25"))
OUT = os.environ.get("PGC_OUT", "/ivi/zfs/s0/original_homes/ssalehi/AGID/checkpoints_retrain/pgc_progan_fast/model.pth")
BS = int(os.environ.get("PGC_BS", "128"))
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".JPEG")


def log(*a):
    print(*a, flush=True)


def collect(regime):
    if regime == "progan":
        root = "/hddstore/ssalehi/AGID/CNNDetection/train"
        reals = [f for f in glob.glob(root + "/**/0_real/*", recursive=True) if f.endswith(EXT)]
        fakes = [f for f in glob.glob(root + "/**/1_fake/*", recursive=True) if f.endswith(EXT)]
    elif regime == "sdv14":
        root = "/hddstore/ssalehi/AGID/GenImage/train"
        reals = [f for f in glob.glob(root + "/real/*") if f.endswith(EXT)]
        fakes = [f for f in glob.glob(root + "/fake/*") if f.endswith(EXT)]
    else:
        raise ValueError(regime)
    random.seed(0); random.shuffle(reals); random.shuffle(fakes)
    reals, fakes = reals[:N], fakes[:N]
    items = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
    random.shuffle(items)
    return items


class PGCData(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = v2.Compose([
            v2.RandomCrop(224, pad_if_needed=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.2),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMNET_MEAN, IMNET_STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        for _ in range(4):
            path, label = self.items[i]
            try:
                img = Image.open(path).convert("RGB")
                return self.tf(img), label
            except Exception:
                i = random.randrange(len(self.items))   # skip corrupt → another sample
        return torch.zeros(3, 224, 224), 0


def main():
    # cn023 flakily fails CUDA init — never fall back to CPU silently (a CPU "run" looks alive but
    # is ~50x slower); retry, then abort loudly so the watcher sees a real failure.
    dev = None
    for attempt in range(5):
        try:
            if torch.cuda.is_available():
                (torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")).sum().item()
                dev = "cuda"; break
        except Exception:
            pass
        log(f"[cuda] init failed (attempt {attempt+1}/5), retrying..."); time.sleep(5)
    if dev is None:
        log("[cuda] CUDA NEVER CAME UP — aborting (no silent CPU fallback)"); sys.exit(2)
    items = collect(REGIME)
    log(f"[data] regime={REGIME} N/class={N} total={len(items)} (real+fake)")
    dl = DataLoader(PGCData(items), batch_size=BS, shuffle=True, num_workers=12,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    steps_per_epoch = len(dl)
    log(f"[data] {steps_per_epoch} steps/epoch @ bs={BS}")

    cfg = compose_config(["model@ModelConfig=image/pgc", "trainer@TrainerConfig=image/pgc",
                          "dataset@DatasetConfig=image/cnndetection"])
    model = AGIDPGCModel(cfg).to(dev)
    init = os.environ.get("PGC_INIT", "")
    if init:                                          # continue training from a prior checkpoint
        sd = torch.load(init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"[init] loaded {init} (missing={len(missing)} unexpected={len(unexpected)})")
    log(f"[model] label_smoothing={model.label_smoothing} trainable="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch, eta_min=5e-7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time(); tot = 0.0; n = 0
        for images, labels in dl:
            images, labels = images.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.compute_loss((images, labels, None))
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            tot += float(loss); n += 1
        dt = time.time() - t0
        log(f"[epoch {ep+1}/{EPOCHS}] loss={tot/max(n,1):.4f} time={dt:.0f}s lr={sched.get_last_lr()[0]:.2e}")
        torch.save(model.state_dict(), OUT)      # checkpoint after every epoch (raw state_dict)
    log(f"[done] saved {OUT}")


if __name__ == "__main__":
    main()
