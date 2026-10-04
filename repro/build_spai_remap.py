import os, torch
os.environ.setdefault("SYNTHSCOPE_MFM_CKPT", "/hddstore/ssalehi/SynthScope/mfm/mfm_pretrain_vit_base.pth")
from omegaconf import OmegaConf
import sys; sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/SynthScope")

off = torch.load("/hddstore/ssalehi/SynthScope/repro/ckpts/spai_official.pth", map_location="cpu", weights_only=False)
if isinstance(off, dict) and "model" in off:
    off = off["model"]
# official non-vit keys (the projectors/context/SCV live under mfvit.* but not mfvit.vit.*)
nonvit = {k: tuple(v.shape) for k, v in off.items() if not k.startswith("mfvit.vit.")}
print("=== OFFICIAL non-vit keys (proj/context/sca/head):", len(nonvit))
for k, s in nonvit.items():
    print(f"  {k}  {s}")

# build SynthScope SPAI model + dump its non-backbone keys
from models.image.spai.spai import SPAI
cfg = OmegaConf.create({"ModelConfig": {"freeze_backbone": True, "projection_dim": 1024,
    "input_size": 224, "mask_radius": 16, "use_sca": True, "patch_size": 224, "patch_stride": 224,
    "num_heads": 12, "attn_embed_dim": 1536, "minimum_patches": 4}})
m = SPAI(cfg)
synthscope = {k: tuple(v.shape) for k, v in m.state_dict().items() if not k.startswith("backbone.")}
print("=== SynthScope non-backbone keys:", len(synthscope))
for k, s in synthscope.items():
    print(f"  {k}  {s}")
