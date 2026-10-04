import os, glob, random
SB = "/hddstore/ssalehi/SynthScope/Synthbuster/extracted/synthbuster"
DEST = "/hddstore/ssalehi/SynthScope/Synthbuster/test"
reals = sorted(glob.glob("/hddstore/ssalehi/SynthScope/GenImage/test/real/*"))[:1000]
random.seed(0)
gens = [d for d in sorted(os.listdir(SB)) if os.path.isdir(os.path.join(SB, d))]
print("generators:", gens, flush=True)
print("proxy reals available:", len(reals), flush=True)
exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
for g in gens:
    fakes = [f for f in sorted(glob.glob(f"{SB}/{g}/*")) if f.lower().endswith(exts)]
    of, orl = f"{DEST}/{g}/1_fake", f"{DEST}/{g}/0_real"
    os.makedirs(of, exist_ok=True); os.makedirs(orl, exist_ok=True)
    for i, f in enumerate(fakes):
        l = f"{of}/{i:05d}_{os.path.basename(f)}"
        if not os.path.exists(l):
            try: os.symlink(f, l)
            except OSError: pass
    for i, f in enumerate(reals):
        l = f"{orl}/{i:05d}_{os.path.basename(f)}"
        if not os.path.exists(l):
            try: os.symlink(f, l)
            except OSError: pass
    print(f"  {g}: {len(os.listdir(of))} fake + {len(os.listdir(orl))} real(proxy)", flush=True)
print("Synthbuster restructured ->", DEST, flush=True)
