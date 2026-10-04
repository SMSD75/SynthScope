"""``synthscope`` command-line interface.

    synthscope list                              # list detectors and weight availability
    synthscope detect FILE --model fat_former    # score an image or video (auto-downloads weights)
    synthscope detect FILE -m d3 --json          # machine-readable output
    synthscope pull --model clip                 # fetch a checkpoint from the model zoo
    synthscope push --repo USER/agid-checkpoints # (maintainer) publish checkpoints to the Hub
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_list(args: argparse.Namespace) -> int:
    from . import list_models

    models = list_models()
    if args.json:
        print(json.dumps(models, indent=2))
        return 0
    width = max(len(m["key"]) for m in models)
    print(f"{'KEY'.ljust(width)}  AVAIL  MODALITY  NAME")
    for m in models:
        mark = "  ok " if m["available"] else "  -- "
        print(f"{m['key'].ljust(width)}  {mark}  {m['modality']:<8}  {m['name']} ({m['paper']})")
    if not args.json:
        print("\n'--' = weights not found under $AGID_CKPT_DIR (training-free models are always available).")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    from . import hub

    path = hub.download(args.model, repo_id=args.repo)
    print(f"downloaded {args.model} -> {path}")
    return 0


def _cmd_push(args: argparse.Namespace) -> int:
    from . import hub

    keys = args.models.split(",") if args.models else None
    summary = hub.push(
        repo_id=args.repo,
        keys=keys,
        include_official=args.include_official,
        private=not args.public,
        dry_run=args.dry_run,
    )
    return 0 if (summary["uploaded"] or summary["dry_run"]) else 1


def _cmd_detect(args: argparse.Namespace) -> int:
    from . import load

    det = load(args.model, checkpoint=args.checkpoint, device=args.device,
               auto_download=not args.no_download, repo=args.repo)
    if not det.available:
        print(
            f"error: no weights for '{args.model}'. Place its checkpoint under $AGID_CKPT_DIR "
            f"or pass --checkpoint PATH (training-free models need none).",
            file=sys.stderr,
        )
        return 2
    result = det.predict(args.file, num_frames=args.frames)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        p = result["fake_probability"]
        print(f"{result['model_name']} -> {result['verdict'].upper()}  "
              f"(P(fake)={p:.4f}, confidence={result['confidence']:.4f}, {result['elapsed_ms']:.0f} ms)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthscope", description="AI-generated image & video detection.")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list available detectors")
    pl.add_argument("--json", action="store_true", help="emit JSON")
    pl.set_defaults(func=_cmd_list)

    pd = sub.add_parser("detect", help="score an image or video file")
    pd.add_argument("file", help="path to an image or video")
    pd.add_argument("-m", "--model", required=True, help="detector key (see `synthscope list`)")
    pd.add_argument("--checkpoint", default=None, help="override the .pth checkpoint path")
    pd.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    pd.add_argument("--frames", type=int, default=None,
                    help="frames to sample when running an image detector on a video")
    pd.add_argument("--repo", default=None, help="model-zoo Hub repo to download from")
    pd.add_argument("--no-download", action="store_true", help="do not auto-download weights")
    pd.add_argument("--json", action="store_true", help="emit JSON")
    pd.set_defaults(func=_cmd_detect)

    pp = sub.add_parser("pull", help="download a detector's checkpoint from the model zoo")
    pp.add_argument("-m", "--model", required=True, help="detector key")
    pp.add_argument("--repo", default=None, help="Hub repo (default: $AGID_HUB_REPO)")
    pp.set_defaults(func=_cmd_pull)

    pu = sub.add_parser("push", help="(maintainer) publish checkpoints to the Hub model zoo")
    pu.add_argument("--repo", default=None, help="target Hub repo id (default: $AGID_HUB_REPO)")
    pu.add_argument("--models", default=None, help="comma-separated keys (default: all eligible)")
    pu.add_argument("--include-official", action="store_true",
                    help="also upload official-weights checkpoints (verify their license first!)")
    pu.add_argument("--public", action="store_true", help="create a PUBLIC repo (default: private)")
    pu.add_argument("--dry-run", action="store_true", help="report the selection, upload nothing")
    pu.set_defaults(func=_cmd_push)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
