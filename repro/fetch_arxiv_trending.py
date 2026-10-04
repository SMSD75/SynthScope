#!/usr/bin/env python3
"""Read arXiv daily for AI-generated image/video DETECTION papers and surface candidates.

This is the deterministic "reads arXiv every day" engine behind the What's-Trending agent.
It queries the arXiv API for recent papers in the domain, then applies RULE 1 automatically:
keep only papers that advertise a public code repository (a github.com link in the abstract or
the arxiv comment). RULE 2 ("solid results vs previous methods") needs judgment and is left to
the daily agent, which reads the abstracts of the candidates this script prints.

Usage:
  python repro/fetch_arxiv_trending.py [--days N] [--max M] [--out FILE]
Writes candidates to repro/trending_arxiv_candidates.json (and prints a summary).
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"

# Domain queries (title/abstract). Detection-focused so we don't pull generation-only papers.
TERMS = [
    'AI-generated image detection', 'synthetic image detection', 'deepfake detection',
    'AI-generated video detection', 'diffusion-generated image detection',
    'fake image detection', 'generated image detector', 'forgery detection',
]
GITHUB_RE = re.compile(r'https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+', re.I)


def _query(term: str, max_results: int) -> list[dict]:
    q = f'(ti:"{term}" OR abs:"{term}") AND cat:cs.CV'
    url = f"{API}?{urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': max_results, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "SynthScope-trending/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall(f"{ATOM}entry"):
        title = " ".join((e.findtext(f"{ATOM}title") or "").split())
        summ = " ".join((e.findtext(f"{ATOM}summary") or "").split())
        comment = e.findtext(f"{ARX}comment") or ""
        published = e.findtext(f"{ATOM}published") or ""
        aid = (e.findtext(f"{ATOM}id") or "").split("/abs/")[-1]
        authors = [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")]
        gh = GITHUB_RE.search(summ + " " + comment)
        out.append({
            "arxiv_id": aid, "title": title,
            "authors": ", ".join(filter(None, authors[:4])) + (" et al." if len(authors) > 4 else ""),
            "published": published[:10], "arxiv": f"https://arxiv.org/abs/{aid}",
            "github": gh.group(0).rstrip(".") if gh else "",
            "abstract": summ,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="keep papers submitted within the last N days")
    ap.add_argument("--max", type=int, default=40, help="max results per query term")
    ap.add_argument("--out", default="repro/trending_arxiv_candidates.json")
    ap.add_argument("--all", action="store_true", help="keep candidates even without a github link")
    a = ap.parse_args()

    cutoff = (dt.date.today() - dt.timedelta(days=a.days)).isoformat()
    seen, merged = set(), []
    for term in TERMS:
        try:
            for p in _query(term, a.max):
                if p["arxiv_id"] in seen:
                    continue
                seen.add(p["arxiv_id"])
                merged.append(p)
        except Exception as ex:  # one term failing shouldn't kill the run
            print(f"[warn] query '{term}' failed: {ex}", file=sys.stderr)

    recent = [p for p in merged if p["published"] >= cutoff]
    # RULE 1 (automatic): must have a public code repo, unless --all
    candidates = recent if a.all else [p for p in recent if p["github"]]
    candidates.sort(key=lambda p: p["published"], reverse=True)

    with open(a.out, "w") as f:
        json.dump({"generated": dt.datetime.utcnow().isoformat()[:19] + "Z",
                   "cutoff": cutoff, "n_scanned": len(merged), "n_recent": len(recent),
                   "candidates": candidates}, f, indent=2)
    print(f"scanned {len(merged)} | recent(<= {a.days}d) {len(recent)} | "
          f"with-code {len([p for p in recent if p['github']])} | written {len(candidates)} -> {a.out}")
    for p in candidates[:25]:
        print(f"  [{p['published']}] {p['title'][:70]}  | code: {p['github'] or '—'}")


if __name__ == "__main__":
    main()
