import torch, collections
sd = torch.load("/hddstore/ssalehi/AGID/repro/ckpts/spai_official.pth", map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
ks = list(sd.keys())
print("total keys", len(ks))
print("top-level prefixes:", dict(collections.Counter(k.split(".")[0] for k in ks)))
print("attn/aggregator keys:", [k for k in ks if any(t in k.lower() for t in ("atten", "aggreg", "to_kv", "to_out"))][:8])
print("head/cls keys:", [k for k in ks if any(t in k.lower() for t in ("head", "cls_", "classif"))][:8])
print("proj keys:", [k for k in ks if "proj" in k.lower()][:6])
print("norm keys:", [k for k in ks if k.lower().startswith("norm") or ".norm" in k.lower()][:6])
print("sample first 12:", ks[:12])
