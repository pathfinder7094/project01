from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from app.pipeline.markdown import load_knowledge_documents


def local_search(wiki_root: Path, query: str, limit: int) -> list[tuple[float, str, str]]:
    terms = [t for t in re.findall(r"[0-9A-Za-z가-힣_-]{2,}", query.casefold())]
    scored: list[tuple[float, str, str]] = []
    for document, relative in load_knowledge_documents(wiki_root):
        text = " ".join([document.title, document.summary, document.definition, " ".join(document.aliases)]).casefold()
        score = sum(text.count(term) for term in terms)
        if score > 0:
            scored.append((float(score), document.title, relative))
    scored.sort(key=lambda x: (-x[0], x[1].casefold()))
    return scored[:limit]


def qmd_search(wiki_root: Path, query: str, limit: int) -> int:
    qmd = shutil.which("qmd")
    if not qmd:
        return 127
    # QMD is an external search engine over the same Markdown vault.
    proc = subprocess.run([qmd, "query", query, "-n", str(limit)], cwd=str(wiki_root), text=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Markdown-first WikiLLM")
    parser.add_argument("query")
    parser.add_argument("--wiki", default="./llmwiki/wiki")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--engine", choices=["auto", "qmd", "local"], default="auto")
    args = parser.parse_args()

    wiki_root = Path(args.wiki).resolve()
    if args.engine in {"auto", "qmd"}:
        rc = qmd_search(wiki_root, args.query, args.limit)
        if rc == 0 or args.engine == "qmd":
            raise SystemExit(rc)

    rows = local_search(wiki_root, args.query, args.limit)
    if not rows:
        print("No results.")
        return
    for score, title, relative in rows:
        print(f"[{score:.0f}] {title}")
        print(f"  path: {relative}")


if __name__ == "__main__":
    main()
