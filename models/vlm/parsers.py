"""Verdict parsers of VLM detectors: the model's generated text -> "fake" / "real" / None (unparsed).

``parse_veritas_v1`` is vendored from the authors' Veritas (v1) metric script, ``self_scripts/infer/cal_metrics.py``
(github.com/EricTan7/Veritas @ d384e81, Apache-2.0, Copyright the Veritas authors), ``get_metrics``: the per-item
answer extraction only, unchanged (the regex, the ``lower().strip()``, and "real" checked before "fake"). The official
script drops an unparsed output from the accuracy denominator (``failed += 1; continue``); here it is returned as
None and the caller decides. Veritas++ (EricTan7/VeritasPP) releases no parser or metric script, so this is the one
its authors used for the earlier model.

``parse_adb`` is vendored from Align, Don't Blend's ``detector/core.py`` ``parse_vlm_verdict`` (github.com/GzZning/
Align-Dont-Blend @ f4b4794; the repository has no licence file, see LICENSES.md), unchanged in behaviour: exactly one
``<answer> ... </answer>`` whose whitespace-normalised, case-folded text is "this is an authentic image." or "this is an
ai-generated image."; with no answer tag, a final line "This is an authentic / AI-generated image." (optionally
followed by ``</answer>``). Any other output raises ``ValueError`` in the official code (``predict.py`` then stops);
here it is returned as None (unparsed). ``adb_verdict_offset`` says where in the text that parser found a tag-less
verdict, so that the score is read at the same word (``VLMConfig.verdict_offset``).
"""
import re
from typing import Optional

VERITAS_ANSWER = r'<answer>\s*(.*?)\s*</answer>'

# Align-Dont-Blend detector/core.py @ f4b4794, verbatim.
_ADB_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ADB_FINAL_SENTENCE = re.compile(
    r"(?:^|\n)\s*This is an (authentic|AI-generated) image\.\s*(?:</answer>)?\s*$",
    re.IGNORECASE,
)
_ADB_VERDICTS = {
    "this is an authentic image.": False,
    "this is an ai-generated image.": True,
}


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


def adb_parse_vlm_verdict(response: str) -> bool:
    """The official ``parse_vlm_verdict``: True = AI-generated, False = authentic; ValueError otherwise."""
    matches = _ADB_ANSWER.findall(response)
    if len(matches) == 1:
        answer = " ".join(matches[0].split()).casefold()
        if answer not in _ADB_VERDICTS:
            raise ValueError(f"Unrecognized VLM verdict: {matches[0]!r}")
        return _ADB_VERDICTS[answer]
    if len(matches) == 0:
        final = _ADB_FINAL_SENTENCE.search(response)
        if final:
            return final.group(1).casefold() == "ai-generated"
    raise ValueError(f"Expected one explicit final verdict, found {len(matches)} tags")


def parse_adb(output: str) -> Optional[str]:
    """"fake" / "real" by the official ``parse_vlm_verdict``, None where it raises (unparsed)."""
    try:
        return "fake" if adb_parse_vlm_verdict(output) else "real"
    except ValueError:
        return None


def adb_verdict_offset(output: str) -> Optional[int]:
    """Where ``parse_vlm_verdict`` read its verdict when that is NOT the single ``<answer>`` tag: the character offset in
    ``output`` of the verdict word ("authentic" / "AI") of the tag-less final line "This is an authentic / AI-generated
    image."; None when the verdict comes from the answer tag (the engine's search after ``<answer>`` finds that word) or
    the output does not parse. The same branches as ``adb_parse_vlm_verdict``, in the same order (``VLMConfig.verdict_offset``)."""
    if len(_ADB_ANSWER.findall(output)) != 0:
        return None
    final = _ADB_FINAL_SENTENCE.search(output)
    return final.start(1) if final else None
