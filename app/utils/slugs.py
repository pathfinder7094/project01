from __future__ import annotations

import re


def stable_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^0-9a-zA-Z가-힣._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "untitled"
