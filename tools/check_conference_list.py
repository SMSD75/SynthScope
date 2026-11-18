#!/usr/bin/env python
"""Check (and optionally export) docs/CONFERENCE_PAPERS.md.

The markdown file is the single source of truth for SynthScope's list of top-tier conference and journal
papers on AI-generated image and video detection. This checker parses it and verifies that

* every entry has the expected format (title, first author, paper link, optional arXiv and code links, tags,
  optional SynthScope mark),
* every paper link points to an allowed official domain for its venue (CVF open access, ecva.net,
  proceedings.neurips.cc / OpenReview, proceedings.mlr.press, ojs.aaai.org, ijcai.org, dl.acm.org,
  IEEE Xplore, Springer),
* no title appears twice,
* venue sections and year sections are in the expected order (venues as listed below, years newest first),
* the counts table at the top matches the entries, and the "last updated" date is present.

It does no network access. With ``--export PATH`` it also writes the parsed list as JSON (the web page's
"Papers" tab reads ``interface/static/conference_papers.json``).

    python tools/check_conference_list.py                       # check docs/CONFERENCE_PAPERS.md
    python tools/check_conference_list.py --export interface/static/conference_papers.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = ROOT / "docs" / "CONFERENCE_PAPERS.md"
DEFAULT_JSON = ROOT / "interface" / "static" / "conference_papers.json"

CONFERENCES = ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "AAAI", "IJCAI", "ACM MM", "KDD", "WACV"]
JOURNALS = ["TPAMI", "IJCV"]
JOURNALS_HEADING = "Journals"
TAGS = ["Image", "Video", "Benchmark", "Localization", "MLLM"]
FIRST_YEAR = 2020

ALLOWED_DOMAINS = {
    "CVPR": {"openaccess.thecvf.com"},
    "ICCV": {"openaccess.thecvf.com"},
    "WACV": {"openaccess.thecvf.com"},
    "ECCV": {"www.ecva.net", "ecva.net", "eccv.ecva.net"},
    "NeurIPS": {"proceedings.neurips.cc", "openreview.net", "neurips.cc"},
    "ICLR": {"openreview.net", "iclr.cc"},
    "ICML": {"proceedings.mlr.press", "openreview.net", "icml.cc"},
    "AAAI": {"ojs.aaai.org"},
    "IJCAI": {"www.ijcai.org", "ijcai.org"},
    "ACM MM": {"dl.acm.org"},
    "KDD": {"dl.acm.org"},
    "TPAMI": {"ieeexplore.ieee.org"},
    "IJCV": {"link.springer.com"},
}

ENTRY_RE = re.compile(
    r"^- \*\*(?P<title>.+?)\*\*"
    r" · (?P<authors>[^·\[\]]+?)"
    r" · \[\[paper\]\]\((?P<paper>https://[^)\s]+)\)"
    r"(?: · \[\[arXiv\]\]\((?P<arxiv>https://arxiv\.org/abs/[0-9]{4}\.[0-9]{4,5})\))?"
    r"(?: · \[\[code\]\]\((?P<code>https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\))?"
    r" · (?P<tags>`[A-Za-z]+`(?: `[A-Za-z]+`)*)"
    r"(?P<synthscope> · ✅)?$"
)
UPDATED_RE = re.compile(r"[Ll]ast updated:?\**\s*(\d{4}-\d{2}-\d{2})")
TABLE_ROW_RE = re.compile(r"^\|\s*(?:\[(?P<venue>[^\]]+)\]\(#[^)]+\)|\*\*(?P<total>Total)\*\*)\s*\|(?P<cells>.*)\|\s*$")


def norm_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title).lower()
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def parse(text: str) -> tuple[dict, list[str]]:
    """Parse the markdown into {last_updated, years, table, papers}; return it with a list of format errors."""
    errors: list[str] = []
    papers: list[dict] = []
    table: dict[str, dict[str, int]] = {}
    table_years: list[str] = []
    venue_order: list[str] = []
    years_seen: dict[str, list[int]] = {}

    m = UPDATED_RE.search(text)
    last_updated = m.group(1) if m else None
    if not last_updated:
        errors.append("no 'Last updated: YYYY-MM-DD' line")

    section = None          # current "## " heading
    year_venue = None       # (venue, year) of the current "### " heading
    in_table = False
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.rstrip()
        # ---- counts table -------------------------------------------------------------------------
        if line.startswith("| Venue |"):
            in_table = True
            table_years = [c.strip() for c in line.strip("|").split("|")][1:]
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
            elif set(line.replace("|", "").strip()) <= set("-: "):
                continue
            else:
                tm = TABLE_ROW_RE.match(line)
                cells = [c.strip().strip("*") for c in (tm.group("cells") if tm else "").split("|")]
                if not tm or len(cells) != len(table_years):
                    errors.append(f"line {lineno}: malformed counts-table row: {line}")
                    continue
                table[tm.group("venue") or tm.group("total")] = {
                    y: (0 if c in ("", "·", "–", "-") else int(c)) for y, c in zip(table_years, cells)}
                continue
        # ---- headings -----------------------------------------------------------------------------
        if line.startswith("## "):
            section = line[3:].strip()
            year_venue = None
            if section in CONFERENCES or section == JOURNALS_HEADING:
                venue_order.append(section)
            continue
        if line.startswith("### "):
            head = line[4:].strip()
            hm = re.fullmatch(r"(?P<venue>.+) (?P<year>\d{4})", head)
            if section in CONFERENCES:
                if not hm or hm.group("venue") != section:
                    errors.append(f"line {lineno}: '### {head}' must be '### {section} <year>'")
                    year_venue = None
                    continue
            elif section == JOURNALS_HEADING:
                if not hm or hm.group("venue") not in JOURNALS:
                    errors.append(f"line {lineno}: '### {head}' must be '### <{'|'.join(JOURNALS)}> <year>'")
                    year_venue = None
                    continue
            else:
                year_venue = None
                continue
            venue, year = hm.group("venue"), int(hm.group("year"))
            if year < FIRST_YEAR:
                errors.append(f"line {lineno}: year {year} is before {FIRST_YEAR}")
            years_seen.setdefault(venue, []).append(year)
            year_venue = (venue, year)
            continue
        # ---- entries ------------------------------------------------------------------------------
        if line.startswith("- **"):
            if year_venue is None:
                errors.append(f"line {lineno}: entry outside a '### <Venue> <year>' section")
                continue
            em = ENTRY_RE.match(line)
            if not em:
                errors.append(f"line {lineno}: entry does not match the format: {line[:120]}")
                continue
            venue, year = year_venue
            tags = re.findall(r"`([A-Za-z]+)`", em.group("tags"))
            bad = [t for t in tags if t not in TAGS]
            if bad:
                errors.append(f"line {lineno}: unknown tag(s) {bad}; allowed: {TAGS}")
            if len(set(tags)) != len(tags):
                errors.append(f"line {lineno}: repeated tag")
            if not ({"Image", "Video"} & set(tags)):
                errors.append(f"line {lineno}: an entry needs at least one of `Image` / `Video`")
            authors = em.group("authors").strip()
            if not re.fullmatch(r"[^·]+?( et al\.)?", authors):
                errors.append(f"line {lineno}: authors must be 'First Author' or 'First Author et al.'")
            papers.append({
                "venue": venue, "year": year, "title": em.group("title").strip(), "authors": authors,
                "paper": em.group("paper"), "arxiv": em.group("arxiv"), "code": em.group("code"),
                "tags": tags, "synthscope": bool(em.group("synthscope")), "line": lineno})
            continue
        if line.startswith("- ") and section in CONFERENCES + [JOURNALS_HEADING] and year_venue:
            errors.append(f"line {lineno}: list item that is not a paper entry: {line[:80]}")

    # ---- venue and year order -------------------------------------------------------------------
    expected = [v for v in CONFERENCES + [JOURNALS_HEADING] if v in venue_order]
    if venue_order != expected:
        errors.append(f"venue sections out of order: {venue_order} (expected {expected})")
    for venue, ys in years_seen.items():
        if venue in JOURNALS:
            continue
        if ys != sorted(ys, reverse=True) or len(set(ys)) != len(ys):
            errors.append(f"{venue}: year sections must be unique and newest first, got {ys}")
    jy = [(p["venue"], p["year"]) for p in papers if p["venue"] in JOURNALS]
    for j in JOURNALS:
        ys = [y for v, y in dict.fromkeys(jy) if v == j]
        if ys != sorted(ys, reverse=True):
            errors.append(f"{j}: year sections must be newest first, got {ys}")
    return {"last_updated": last_updated, "table_years": table_years, "table": table, "papers": papers}, errors


def check(data: dict) -> list[str]:
    """Semantic checks on parsed data: domains, duplicates, counts table."""
    errors: list[str] = []
    papers = data["papers"]
    if not papers:
        errors.append("no entries found")
    # official domains
    for p in papers:
        host = urlparse(p["paper"]).hostname or ""
        if host not in ALLOWED_DOMAINS.get(p["venue"], set()):
            errors.append(f"line {p['line']}: {p['venue']} paper link must be on "
                          f"{sorted(ALLOWED_DOMAINS.get(p['venue'], set()))}, got {host}")
        if p["venue"] == "TPAMI" and not re.search(r"/document/\d+", p["paper"]):
            errors.append(f"line {p['line']}: TPAMI link must be an IEEE Xplore /document/<id> page")
        if p["venue"] in ("ACM MM", "KDD") and "/doi/10.1145/" not in p["paper"]:
            errors.append(f"line {p['line']}: ACM link must be a dl.acm.org/doi/10.1145/... page")
        if p["venue"] == "IJCV" and "10.1007/s11263" not in p["paper"]:
            errors.append(f"line {p['line']}: IJCV link must be a Springer s11263 article")
    # duplicate titles
    seen: dict[str, int] = {}
    for p in papers:
        key = norm_title(p["title"])
        if key in seen:
            errors.append(f"line {p['line']}: duplicate title (first at line {seen[key]}): {p['title']}")
        else:
            seen[key] = p["line"]
    # counts table
    table, years = data["table"], data["table_years"]
    if not table:
        errors.append("no counts table ('| Venue | ... |') found")
        return errors
    if not years or years[-1] != "Total":
        errors.append("counts table must end with a 'Total' column")
        return errors
    year_cols = [y for y in years if y != "Total"]
    counts: dict[str, dict[str, int]] = {}
    for p in papers:
        counts.setdefault(p["venue"], {}).setdefault(str(p["year"]), 0)
        counts[p["venue"]][str(p["year"])] += 1
    present = {p["venue"] for p in papers}
    for venue in present - set(table):
        if venue not in table:
            errors.append(f"counts table has no row for {venue}")
    for venue, row in table.items():
        if venue == "Total":
            continue
        got = counts.get(venue, {})
        for y in year_cols:
            if y.isdigit() and row.get(y, 0) != got.get(y, 0):
                errors.append(f"counts table {venue} {y}: says {row.get(y, 0)}, entries have {got.get(y, 0)}")
        outside = {y: n for y, n in got.items() if y not in year_cols}
        if outside:
            errors.append(f"counts table {venue}: entries in years without a column: {outside}")
        if row.get("Total", 0) != sum(got.values()):
            errors.append(f"counts table {venue} Total: says {row.get('Total', 0)}, entries have {sum(got.values())}")
    if "Total" in table:
        tot = table["Total"]
        for y in year_cols:
            n = sum(1 for p in papers if str(p["year"]) == y)
            if tot.get(y, 0) != n:
                errors.append(f"counts table Total {y}: says {tot.get(y, 0)}, entries have {n}")
        if tot.get("Total", 0) != len(papers):
            errors.append(f"counts table grand total: says {tot.get('Total', 0)}, entries have {len(papers)}")
    else:
        errors.append("counts table has no 'Total' row")
    return errors


def export(data: dict, path: Path) -> None:
    venues = [v for v in CONFERENCES + JOURNALS if any(p["venue"] == v for p in data["papers"])]
    out = {
        "source": "docs/CONFERENCE_PAPERS.md",
        "last_updated": data["last_updated"],
        "venues": venues,
        "journals": JOURNALS,
        "tags": TAGS,
        "papers": [{k: v for k, v in p.items() if k != "line"} for p in data["papers"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", default=str(DEFAULT_MD), help="markdown file (default: docs/CONFERENCE_PAPERS.md)")
    ap.add_argument("--export", nargs="?", const=str(DEFAULT_JSON), default=None, metavar="JSON",
                    help=f"also write the parsed list as JSON (default path: {DEFAULT_JSON.relative_to(ROOT)})")
    args = ap.parse_args(argv)
    data, errors = parse(Path(args.path).read_text(encoding="utf-8"))
    errors += check(data)
    if errors:
        print(f"{args.path}: {len(errors)} problem(s)")
        for e in errors:
            print("  - " + e)
        return 1
    n = len(data["papers"])
    print(f"{args.path}: OK ({n} papers, {sum(p['synthscope'] for p in data['papers'])} in SynthScope, "
          f"last updated {data['last_updated']})")
    if args.export:
        export(data, Path(args.export))
        print(f"wrote {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
