from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_LLMWIKI_ROOT = PROJECT_ROOT / "llmwiki"


def resolve_root(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path.expanduser().resolve()


def ensure_layout(llmwiki_root: Path) -> None:
    for relative in (
        "raw",
        "wiki/concepts",
        "wiki/entities",
        "wiki/sources",
        "wiki/comparisons",
        "wiki/benchmarks",
        "raw/articles",
        "raw/papers",
        "raw/repos",
        "raw/data",
        "raw/images",
        "raw/assets",
    ):
        (llmwiki_root / relative).mkdir(parents=True, exist_ok=True)
    index = llmwiki_root / "wiki/index.md"
    log = llmwiki_root / "wiki/log.md"
    if not index.exists():
        index.write_text("# LLM Wiki Index\n\n", encoding="utf-8")
    if not log.exists():
        log.write_text("# Pipeline Log\n\n", encoding="utf-8")
