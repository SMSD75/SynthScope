"""Verdict parsers of VLM detectors: the model's generated text -> "fake" / "real" / None (unparsed).

``parse_veritas_v1`` is vendored from the authors' Veritas (v1) metric script, ``self_scripts/infer/cal_metrics.py``
(github.com/EricTan7/Veritas @ d384e81, Apache-2.0, Copyright the Veritas authors), ``get_metrics``: the per-item
answer extraction only, unchanged (the regex, the ``lower().strip()``, and "real" checked before "fake"). The official
script drops an unparsed output from the accuracy denominator (``failed += 1; continue``); here it is returned as
None and the caller decides. Veritas++ (EricTan7/VeritasPP) releases no parser or metric script, so this is the one
its authors used for the earlier model.
"""
import re
from typing import Optional

VERITAS_ANSWER = r'<answer>\s*(.*?)\s*</answer>'


def parse_veritas_v1(output: str) -> Optional[str]:
    """"real" / "fake" from the first ``<answer> ... </answer>`` of ``output``, else None (official: "failed")."""
    try:
        answer = re.search(VERITAS_ANSWER, output)
        answer = answer.group(1).lower().strip()
        if "real" in answer:
            return "real"
        elif "fake" in answer:
            return "fake"
        return None
    except Exception:          # the official `except: failed += 1` (no match: .group of None)
        return None
