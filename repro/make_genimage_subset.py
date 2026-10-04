import os, glob, random
src = "/hddstore/ssalehi/SynthScope/GenImage"
dst = "/hddstore/ssalehi/SynthScope/GenImage_sub"
random.seed(0)
for split, n in [("train", 12000), ("test", 2000)]:
    for cls in ["real", "fake"]:
        outdir = f"{dst}/{split}/{cls}"
        os.makedirs(outdir, exist_ok=True)
        files = glob.glob(f"{src}/{split}/{cls}/*")
        random.shuffle(files)
        files = files[:n]
        for i, f in enumerate(files):
            link = f"{outdir}/{i:06d}_{os.path.basename(f)}"
            if not os.path.exists(link):
                try:
                    os.symlink(f, link)
                except OSError:
                    pass
        print(split, cls, len(os.listdir(outdir)), flush=True)
