"""docs/CONFERENCE_PAPERS.md passes tools/check_conference_list.py, and the web page's JSON export is in sync (CPU, no network).

* the real list parses with no format, domain, duplicate or counts-table problem,
* ``interface/static/conference_papers.json`` equals a fresh ``--export`` of the markdown (the markdown is the source of truth),
* the checker catches a duplicate title, a non-official paper link and a wrong count on a small synthetic list.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_conference_list as ccl  # noqa: E402

MD = ROOT / "docs" / "CONFERENCE_PAPERS.md"
JSON = ROOT / "interface" / "static" / "conference_papers.json"


def _problems(text: str) -> list[str]:
    data, errors = ccl.parse(text)
    return errors + ccl.check(data)


def test_real_list_is_valid() -> None:
    assert ccl.main([str(MD)]) == 0
    data, errors = ccl.parse(MD.read_text(encoding="utf-8"))
    assert not errors and not ccl.check(data)
    assert len(data["papers"]) > 100
    assert any(p["synthscope"] for p in data["papers"])


def test_exported_json_in_sync() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "papers.json"
        assert ccl.main([str(MD), "--export", str(out)]) == 0
        fresh = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads(JSON.read_text(encoding="utf-8"))
    assert committed == fresh, ("interface/static/conference_papers.json is stale: run "
                                "python tools/check_conference_list.py --export")


_GOOD = """# t
> **Last updated: 2026-11-18**

| Venue | 2026 | 2025 | Total |
| --- | ---: | ---: | ---: |
| [CVPR](#cvpr) | 1 | 1 | 2 |
| **Total** | **1** | **1** | **2** |

## CVPR

### CVPR 2026

- **Paper A** · Ann Author et al. · [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/A.html) · `Image`

### CVPR 2025

- **Paper B** · Bob Author · [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/B.html) · [[arXiv]](https://arxiv.org/abs/2501.00001) · [[code]](https://github.com/o/r) · `Video` `Benchmark` · ✅
"""


def test_checker_catches_problems() -> None:
    assert _problems(_GOOD) == []
    dup = _GOOD.replace("**Paper B**", "**Paper A**")
    assert any("duplicate title" in e for e in _problems(dup))
    bad_domain = _GOOD.replace("https://openaccess.thecvf.com/content/CVPR2026/html/A.html", "https://arxiv.org/abs/2501.00002")
    assert any("paper link must be on" in e for e in _problems(bad_domain))
    bad_count = _GOOD.replace("| [CVPR](#cvpr) | 1 | 1 | 2 |", "| [CVPR](#cvpr) | 2 | 1 | 2 |")
    assert any("counts table CVPR 2026" in e for e in _problems(bad_count))
    bad_order = _GOOD.replace("### CVPR 2026", "### CVPR 2024")
    assert _problems(bad_order)
    bad_tag = _GOOD.replace("`Image`", "`Photo`")
    assert any("unknown tag" in e for e in _problems(bad_tag))


if __name__ == "__main__":
    test_real_list_is_valid()
    test_exported_json_in_sync()
    test_checker_catches_problems()
    print("ok")
