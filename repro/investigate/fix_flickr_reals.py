"""Replace the corrupted CoSpyBench flickr reals with clean Flickr30k images.

The authentic CoSpyBench (ruojiruoli/Co-Spy-Bench) does NOT ship flickr real images — the fakes are
caption-generated and the "reals" are the original Flickr30k photos (not redistributed; metadata holds
only captions). A prior rebuild used a Flickr30k proxy that half-failed, leaving 1,250 unique 212-byte
corrupt stubs (hardlinked into all 22 flickr subsets via _real_pool/flickr). This re-fetches clean
Flickr30k images and overwrites each corrupt inode IN PLACE (preserving the hardlinks → all 22 subsets
fixed at once). Non-destructive to the directory structure; only replaces broken file *content*.
"""
import glob, io, os
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

POOL = "/hddstore/ssalehi/SynthScope/CoSpyBench/test/_real_pool/flickr"
SUBSETS = glob.glob("/hddstore/ssalehi/SynthScope/CoSpyBench/test/flickr*/0_real")
THRESH = 1024  # bytes; clean flickr jpegs are >>100KB, corrupt stubs are 212B

# 1. gather corrupt files, group by inode (hardlinks share one inode → fix once)
by_inode = {}
total_corrupt = 0
for d in [POOL] + SUBSETS:
    for f in glob.glob(d + "/*"):
        try:
            st = os.stat(f)
        except OSError:
            continue
        if st.st_size <= THRESH:
            total_corrupt += 1
            by_inode.setdefault(st.st_ino, []).append(f)
inodes = list(by_inode)
print(f"corrupt files={total_corrupt}  unique inodes to fix={len(inodes)}", flush=True)

# 2. download one clean Flickr30k parquet shard and re-encode images
shard = hf_hub_download("lmms-lab/flickr30k", "data/test-00000-of-00009.parquet",
                        repo_type="dataset", local_dir="/hddstore/ssalehi/SynthScope/CoSpyBench/_flickr30k_dl")
import pyarrow.parquet as pq
col = pq.read_table(shard, columns=["image"]).column("image").to_pylist()
print(f"flickr30k images in shard: {len(col)}", flush=True)

clean = []
for rec in col:
    b = rec.get("bytes") if isinstance(rec, dict) else rec
    if not b:
        continue
    try:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=95)
        clean.append(buf.getvalue())
    except Exception:
        continue
    if len(clean) >= len(inodes):
        break
print(f"clean re-encoded images: {len(clean)}", flush=True)
assert len(clean) >= len(inodes), f"not enough clean images ({len(clean)}) for {len(inodes)} inodes"

# 3. overwrite each corrupt inode in place (open 'wb' truncates same inode → all hardlinks updated)
for ino, jpg in zip(inodes, clean):
    target = by_inode[ino][0]
    with open(target, "wb") as fh:
        fh.write(jpg)
print(f"overwrote {len(inodes)} inodes (fixes {total_corrupt} hardlinked files)", flush=True)

# 4. verify
bad = 0
for d in [POOL] + SUBSETS:
    for f in glob.glob(d + "/*"):
        if os.path.getsize(f) <= THRESH:
            bad += 1
print(f"remaining <=1KB files after fix: {bad}", flush=True)
# spot-check loads
import random
sample = random.sample(glob.glob(POOL + "/*"), 40)
okc = 0
for f in sample:
    try:
        Image.open(f).convert("RGB").load(); okc += 1
    except Exception:
        pass
print(f"pool spot-check: {okc}/40 load clean", flush=True)
