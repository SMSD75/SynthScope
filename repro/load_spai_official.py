"""Remap the OFFICIAL SPAI checkpoint into the AGID SPAI class and save an AGID-loadable copy.
Official key namespace (PatchBasedMFViT) -> AGID SPAI class names."""
import os, torch, sys
os.environ.setdefault("AGID_MFM_CKPT", "/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth")
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
from omegaconf import OmegaConf
from models.image.spai.spai import SPAI

off = torch.load("/hddstore/ssalehi/AGID/repro/ckpts/spai_official.pth", map_location="cpu", weights_only=False)
if isinstance(off, dict) and "model" in off:
    off = off["model"]

def remap(k):
    if k == "mfvit.frequencies_mask":
        return None  # AGID regenerates `mask` buffer
    if k.startswith("mfvit.vit.head."):
        return None  # ViT's internal 1000/2-class head — unused by SPAI (intermediate features only)
    if k.startswith("mfvit.vit."):
        return "backbone." + k[len("mfvit.vit."):]
    if k.startswith("mfvit.features_processor.patch_projector.projectors."):
        return ("proj." + k[len("mfvit.features_processor.patch_projector.projectors."):]).replace(".projector.", ".proj.")
    if k.startswith("mfvit.features_processor.original_features_processor."):
        return ("context." + k[len("mfvit.features_processor.original_features_processor."):]).replace(".projector.", ".proj.")
    if k == "patch_aggregator":
        return "sca.patch_aggregator"
    if k.startswith("to_kv.") or k.startswith("to_out."):
        return "sca." + k
    if k.startswith("norm."):
        return "attn_norm." + k[len("norm."):]
    if k.startswith("cls_head.head."):
        return "classifier." + k[len("cls_head.head."):]
    return None

new = {}
dropped = []
for k, v in off.items():
    nk = remap(k)
    if nk is None:
        dropped.append(k)
    else:
        new[nk] = v
print(f"remapped {len(new)} keys; dropped {len(dropped)}: {dropped[:5]}")

cfg = OmegaConf.create({"ModelConfig": {"freeze_backbone": True, "projection_dim": 1024,
    "input_size": 224, "mask_radius": 16, "use_sca": True, "patch_size": 224, "patch_stride": 224,
    "num_heads": 12, "attn_embed_dim": 1536, "minimum_patches": 4}})
m = SPAI(cfg)
msg = m.load_state_dict(new, strict=False)
miss = [k for k in msg.missing_keys if not k.startswith("backbone.") or "mask" in k]
print("missing (non-backbone or mask):", [k for k in msg.missing_keys if not k.startswith("backbone.")][:10])
print("# missing total:", len(msg.missing_keys), " # unexpected:", len(msg.unexpected_keys))
print("unexpected sample:", msg.unexpected_keys[:8])
# Only acceptable misses are the regenerated `mask` buffer and any pure-backbone aux buffers.
nonbb_missing = [k for k in msg.missing_keys if not k.startswith("backbone.")]
if nonbb_missing == ["mask"] or nonbb_missing == []:
    torch.save(new, "/hddstore/ssalehi/AGID/repro/ckpts/spai_official_agid.pth")
    print("SAVED spai_official_agid.pth — clean load into AGID class")
else:
    print("MISMATCH — refine remap; not saving")
