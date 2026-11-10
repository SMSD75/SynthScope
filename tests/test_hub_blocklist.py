"""``synthscope.hub.push`` refuses detectors whose weights must not be redistributed (CPU, dry runs only, no network).

* every key of ``NEVER_REDISTRIBUTE`` is a registry key and has a LICENSES.md row,
* with a dummy checkpoint for every registered detector in a temp ``$SYNTHSCOPE_CKPT_DIR`` and ``--include-official``,
  a dry-run push selects none of them and says why, while GRIP's ``bfree`` (non-profit, ``official_*`` rule) and
  SynthScope-trained detectors are still selected,
* without ``--include-official`` the blocked keys stay refused (not merely "official").
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend.registry import MODELS                       # noqa: E402
from synthscope import hub                                           # noqa: E402


def _dry_run(ckpt_dir: Path, **kw):
    hub._ckpt_dir = lambda: ckpt_dir
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        summary = hub.push(repo_id="test/none", dry_run=True, **kw)
    return summary, out.getvalue()


def test_blocklist_is_consistent() -> None:
    licenses = Path(ROOT, "LICENSES.md").read_text()
    for key in hub.NEVER_REDISTRIBUTE:
        assert key in MODELS, key
        assert re.search(rf"`{key}`", licenses), f"{key} has no LICENSES.md row"
    assert "bfree" not in hub.NEVER_REDISTRIBUTE and "waverep" not in hub.NEVER_REDISTRIBUTE
    print(f"    {len(hub.NEVER_REDISTRIBUTE)} blocked keys, all registered and listed in LICENSES.md")


def test_dry_run_push() -> None:
    original = hub._ckpt_dir
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for spec in MODELS.values():
                if spec.checkpoint:
                    Path(tmp, spec.checkpoint).write_bytes(b"x")
            for flag in (True, False):
                summary, text = _dry_run(Path(tmp), include_official=flag)
                assert summary["dry_run"]
                refused = {k for k, why in summary["skipped"] if why.startswith("REFUSED")}
                assert refused == set(hub.NEVER_REDISTRIBUTE), refused
                assert not refused & set(summary["uploaded"])
                assert "REFUSED" in text and "even with --include-official" in text
            summary, _ = _dry_run(Path(tmp), include_official=True)
            assert "bfree" in summary["uploaded"] and "clip" in summary["uploaded"], summary["uploaded"]
            summary, _ = _dry_run(Path(tmp), include_official=False)
            assert "bfree" not in summary["uploaded"] and "clip" in summary["uploaded"]
            summary, text = _dry_run(Path(tmp), keys=["probe", "ltf"], include_official=True)
            assert summary["uploaded"] == [] and len(summary["skipped"]) == 2, summary
    finally:
        hub._ckpt_dir = original
    print("    dry-run push (with and without --include-official): blocked keys refused, bfree/clip handled as before")


if __name__ == "__main__":
    test_blocklist_is_consistent()
    test_dry_run_push()
    print("hub blocklist test passed")
