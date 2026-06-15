#!/usr/bin/env python3
"""Render the README 'What's Trending' section from interface/backend/trending.json.

Single source of truth = interface/backend/trending.json (also served to the web UI at /api/trending).
This script rewrites the block in README.md between the markers:
    <!-- TRENDING:START --> ... <!-- TRENDING:END -->
so the README and the website never drift. The daily refresh task updates the JSON, then runs this.

Usage:  python repro/generate_trending_readme.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRENDING = ROOT / "interface" / "backend" / "trending.json"
README = ROOT / "README.md"
START, END = "<!-- TRENDING:START -->", "<!-- TRENDING:END -->"


def _entry_line(e: dict) -> str:
    name = e.get("title", "?")
    link = e.get("github") or e.get("arxiv") or ""
    head = f"[{name}]({link})" if link else name
    venue = " · ".join(x for x in (e.get("authors", ""), e.get("venue", ""), str(e.get("year", ""))) if x)
    code = f" — [code]({e['github']})" if e.get("github") else (f" — [arXiv]({e['arxiv']})" if e.get("arxiv") else "")
    agid = f" · _in AGID: `{e['agid']}`_" if e.get("agid") else ""
    parts = [f"- **{head}**"]
    if venue:
        parts.append(f" ({venue})")
    parts.append(f"{code}{agid}")
    body = f"  \n  {e['summary']}" if e.get("summary") else ""
    res = f"  \n  _Results: {e['result']}_" if e.get("result") else ""
    return "".join(parts) + body + res


def render(data: dict) -> str:
    out = [START,
           "## 📈 What's Trending — AI-generated content detection",
           "",
           f"*High-quality recent papers for image & video AIGC detection. {data.get('criteria','')}"
           f" Last updated: {data.get('updated','')}.*",
           ""]
    if data.get("image"):
        out.append("### Image detectors")
        out += [_entry_line(e) for e in data["image"]]
        out.append("")
    if data.get("video"):
        out.append("### Video detectors")
        out += [_entry_line(e) for e in data["video"]]
        out.append("")
    if data.get("landscape"):
        out.append("### Benchmarks & curated lists")
        for e in data["landscape"]:
            link = e.get("github", "")
            out.append(f"- [{e.get('title','?')}]({link}) — {e.get('summary','')}")
        out.append("")
    out.append(END)
    return "\n".join(out)


def main():
    data = json.loads(TRENDING.read_text())
    block = render(data)
    text = README.read_text()
    if START in text and END in text:
        pre = text[: text.index(START)]
        post = text[text.index(END) + len(END):]
        README.write_text(pre + block + post)
    else:
        # First run: append the section at the end.
        README.write_text(text.rstrip() + "\n\n" + block + "\n")
    print(f"Rendered {len(data.get('image',[]))} image + {len(data.get('video',[]))} video "
          f"+ {len(data.get('landscape',[]))} landscape entries into README.md")


if __name__ == "__main__":
    main()
