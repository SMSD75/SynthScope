"""Build NSG-VD's reference-set file: the NSG features of the official reference videos, through SynthScope's pipeline.

The official test (test_dMMD.py, configs/nsg-vd-224x224/test.yaml: ``ref_model: MSR-VTT``, ``ref_mode: val``,
``ref_load_len: 100``) scores every video against the first 100 ids of ``split/real/MSR-VTT/val_ids.txt`` (the
released assets/split.zip), in file order; a video that does not give 8 frames is dropped, not replaced (data/
video_frames_dataset.py). Each video goes through the same path as a scored video: media.ffmpeg_select_frames ->
``nsgvd_frame_224`` -> the diffusion NSG (end frames x100), three videos per diffusion pass as the official
ScoreFeaturesDataset (``process_batch_size=3``).

    python models/video/nsg_vd/build_reference.py --split-zip NSG-VD/assets/split.zip \
        --videos /data/GenVideo-Val/Real --diffusion-weights 256x256_diffusion_uncond.pt \
        --out $SYNTHSCOPE_CKPT_DIR/nsg_vd_reference_msrvtt_val100.safetensors

``--videos`` holds the MSR-VTT clips named as in the split (``MSRVTT_684.mp4``) or as in GenVideo-Val
(``<index>_msrvtt_684.mp4``). Writes ``ref_data`` [N, 8, 3, 224, 224] float32 plus the video ids in the metadata.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "interface")]
from backend import media                                         # noqa: E402
from config_utils import compose_config                           # noqa: E402
from models.video.nsg_vd.nsg_vd import SynthScopeNSGVDModel        # noqa: E402
from models.video.nsg_vd.preprocess import NSGVDFrame              # noqa: E402

SPLIT_MEMBER = "split/real/MSR-VTT/val_ids.txt"
REF_LEN = 100            # test.yaml ref_load_len
PROCESS_BATCH = 3        # ScoreFeaturesDataset process_batch_size (utils/data_utils.py)


def reference_ids(split_zip: str | None, ids_file: str | None) -> list[str]:
    """The official ids file read as VideoFramesDataset does: non-empty stripped lines ending in .mp4, the first 100."""
    if split_zip:
        with zipfile.ZipFile(split_zip) as z:
            text = io.TextIOWrapper(z.open(SPLIT_MEMBER), encoding="utf8").read()
    else:
        text = Path(ids_file).read_text()
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    return [v for v in ids if v.endswith(".mp4")][:REF_LEN]


def find_videos(videos: Path, ids: list[str]) -> list[Path]:
    """Each split id's file: ``<id>`` itself, else a GenVideo-Val ``<index>_<id>`` (case-insensitive)."""
    index = {}
    for p in videos.iterdir():
        index.setdefault(p.name.lower(), p)
        if "_" in p.name:
            index.setdefault(p.name.split("_", 1)[1].lower(), p)
    missing = [v for v in ids if v.lower() not in index]
    if missing:
        raise SystemExit(f"{len(missing)} reference videos not found in {videos}: {missing[:5]}")
    return [index[v.lower()] for v in ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split-zip", help="the official assets/split.zip")
    src.add_argument("--ids", help="or its split/real/MSR-VTT/val_ids.txt")
    ap.add_argument("--videos", required=True, help="directory with the MSR-VTT clips")
    ap.add_argument("--diffusion-weights", required=True, help="256x256_diffusion_uncond.pt (OpenAI guided-diffusion)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.backends.cudnn.deterministic = True      # as the official run (utils/experiment_utils.py set_seed)
    torch.backends.cudnn.benchmark = False
    ids = reference_ids(args.split_zip, args.ids)
    paths = find_videos(Path(args.videos), ids)
    cfg = compose_config(["model@ModelConfig=video/nsg_vd", f"ModelConfig.diffusion_weights={args.diffusion_weights}",
                          f"ModelConfig.process_batch={PROCESS_BATCH}"])
    model = SynthScopeNSGVDModel(cfg).to(args.device).eval()
    transform = NSGVDFrame(224)
    kept, clips, dropped = [], [], []
    for vid, p in zip(ids, paths):
        try:
            frames = media.ffmpeg_select_frames(p, num_frames=model.num_frames)
        except media.VideoDecodeError as e:          # the official dataset drops it
            dropped.append({"id": vid, "reason": str(e)})
            continue
        kept.append(vid)
        clips.append(torch.stack([transform(f) for f in frames]))
    feats = []
    for i in range(0, len(clips), PROCESS_BATCH):
        feats.append(model.nsg(torch.stack(clips[i:i + PROCESS_BATCH]).to(args.device)).cpu())
    ref = torch.cat(feats).contiguous()
    meta = {"ids": json.dumps([v.split(".")[0] for v in kept]), "dropped": json.dumps(dropped),
            "source": "NSG-VD assets/split.zip split/real/MSR-VTT/val_ids.txt, first 100 (test.yaml ref_mode val)",
            "ffmpeg": media.ffmpeg_exe(), "device": args.device, "torch": torch.__version__,
            "diffusion_weights_sha256": hashlib.sha256(Path(args.diffusion_weights).read_bytes()).hexdigest()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file({"ref_data": ref}, args.out, metadata=meta)
    print(f"wrote {args.out}: ref_data {tuple(ref.shape)} from {len(kept)} videos ({len(dropped)} dropped: {dropped})")


if __name__ == "__main__":
    main()
