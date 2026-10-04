#!/usr/bin/env python3
"""Render the 'What's Trending' pages from interface/backend/trending.json.

Single source of truth = interface/backend/trending.json (also served to the web UI at /api/trending).
This script writes
  * TRENDING.md  - the full curated list (image / video / benchmarks), and
  * the short teaser in README.md between the markers
        <!-- TRENDING:START --> ... <!-- TRENDING:END -->
    (counts, last-updated date, link to TRENDING.md),
so the README, TRENDING.md and the website never drift. The daily refresh task updates the JSON, then runs this.

Usage:  python repro/generate_trending_readme.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRENDING = ROOT / "interface" / "backend" / "trending.json"
README = ROOT / "README.md"
TRENDING_MD = ROOT / "TRENDING.md"
START, END = "<!-- TRENDING:START -->", "<!-- TRENDING:END -->"


def _entry_line(e: dict) -> str:
    name = e.get("title", "?")
    link = e.get("github") or e.get("arxiv") or ""
    head = f"[{name}]({link})" if link else name
    venue = " · ".join(x for x in (e.get("authors", ""), e.get("venue", ""), str(e.get("year", ""))) if x)
    code = f" — [code]({e['github']})" if e.get("github") else (f" — [arXiv]({e['arxiv']})" if e.get("arxiv") else "")
    synthscope = f" · _in SynthScope: `{e['synthscope']}`_" if e.get("synthscope") else ""
    parts = [f"- **{head}**"]
    if venue:
        parts.append(f" ({venue})")
    parts.append(f"{code}{synthscope}")
    body = f"  \n  {e['summary']}" if e.get("summary") else ""
    res = f"  \n  _Results: {e['result']}_" if e.get("result") else ""
    return "".join(parts) + body + res


def render_full(data: dict) -> str:
    """The complete TRENDING.md page."""
    out = ["# 📈 What's Trending — AI-generated content detection",
           "",
           f"*High-quality recent papers for image & video AIGC detection. {data.get('criteria','')}"
           f" Last updated: {data.get('updated','')}.*",
           "",
           "Entries marked _in SynthScope_ are integrated in this repo; the rest are tracked, with the reason an "
           "integration is pending noted in their _Results_ line. See [docs/ADD_DETECTOR.md](docs/ADD_DETECTOR.md) "
           "for how a paper is integrated and reproduced.",
           ""]
    if data.get("image"):
        out.append("## Image detectors")
        out += [_entry_line(e) for e in data["image"]]
        out.append("")
    if data.get("video"):
        out.append("## Video detectors")
        out += [_entry_line(e) for e in data["video"]]
        out.append("")
    if data.get("landscape"):
        out.append("## Benchmarks & curated lists")
        for e in data["landscape"]:
            link = e.get("github", "")
            out.append(f"- [{e.get('title','?')}]({link}) — {e.get('summary','')}")
        out.append("")
    out.append("---")
    out.append("*Auto-generated from `interface/backend/trending.json` by `repro/generate_trending_readme.py` "
               "(also shown in the web UI's What's Trending tab) — edit the JSON, not this file.*")
    return "\n".join(out) + "\n"


def render_teaser(data: dict) -> str:
    """The short README block that links to TRENDING.md."""
    n_img, n_vid = len(data.get("image", [])), len(data.get("video", []))
    n_in = sum(1 for k in ("image", "video") for e in data.get(k, []) if e.get("synthscope"))
    return "\n".join([
        START,
        "## 📈 What's Trending",
        "",
        f"A curated, daily-refreshed list of recent image & video AIGC-detection papers with public code: "
        f"**{n_img} image · {n_vid} video** tracked ({n_in} already integrated in SynthScope), last updated "
        f"**{data.get('updated','')}**. → **[TRENDING.md](TRENDING.md)** (also in the web UI's *What's Trending* tab).",
        END])


def main():
    data = json.loads(TRENDING.read_text())
    TRENDING_MD.write_text(render_full(data))
    block = render_teaser(data)
    text = README.read_text()
    if START in text and END in text:
        pre = text[: text.index(START)]
        post = text[text.index(END) + len(END):]
        README.write_text(pre + block + post)
    else:
        # First run: append the section at the end.
        README.write_text(text.rstrip() + "\n\n" + block + "\n")
    print(f"Rendered {len(data.get('image',[]))} image + {len(data.get('video',[]))} video "
          f"+ {len(data.get('landscape',[]))} landscape entries into TRENDING.md (README keeps a teaser)")


if __name__ == "__main__":
    main()
