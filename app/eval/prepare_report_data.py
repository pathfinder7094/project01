from __future__ import annotations

"""
WikiLLM project report-data extractor
=====================================

Purpose
-------
Convert an existing WikiLLM benchmark run into a compact, shareable evidence
package without rerunning Ollama. The script intentionally keeps raw logs,
checkpoints, chunks, and generated Wiki files local and emits only aggregated
statistics plus a small set of representative evidence rows.

Designed for the project layout used by:
  app/outputs/benchmark/batch-runs/<RUN_ID>/
  <llmwiki>/benchmark-runs/<RUN_ID>/<MODEL_SLUG>/wiki/
  config/ontology.yaml
  config/ollama-models.txt (or models.txt)
  config/loop.env

Outputs
-------
  report-data.json       Full compact report dataset (primary share file)
  model-comparison.csv   Flat model comparison table
  evidence.jsonl         Small representative sample/outlier/error evidence
  report-summary.md      Human-readable summary and requirement coverage
  share-package.zip      Optional compact archive of the four files above

The script uses only the Python standard library. It does not call Ollama for
inference. Optional system/model metadata commands (`ollama --version`,
`ollama show`, `nvidia-smi`) can be disabled with --no-system-snapshot.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "model"


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\[\[|\]\]", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None or float(den) == 0:
        return None
    return float(num) / float(den)


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def pct(value: Any) -> str:
    n = finite_number(value)
    return "-" if n is None else f"{n * 100:.2f}%"


def num(value: Any, digits: int = 2) -> str:
    n = finite_number(value)
    return "-" if n is None else f"{n:.{digits}f}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    xs = [v for item in values if (v := finite_number(item)) is not None]
    xs.sort()
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "cv": None,
        }
    mean = statistics.fmean(xs)
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "count": len(xs),
        "mean": mean,
        "median": statistics.median(xs),
        "std": std,
        "min": xs[0],
        "max": xs[-1],
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "cv": safe_div(std, mean) if mean else None,
    }


def pearson(xs: Iterable[Any], ys: Iterable[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx, fy = finite_number(x), finite_number(y)
        if fx is not None and fy is not None:
            pairs.append((fx, fy))
    if len(pairs) < 3:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    dx = [x - mx for x in xvals]
    dy = [y - my for y in yvals]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        yield value
                    elif isinstance(value, list):
                        yield from (x for x in value if isinstance(x, dict))
        else:
            value = read_json(path)
            if isinstance(value, dict):
                yield value
            elif isinstance(value, list):
                yield from (x for x in value if isinstance(x, dict))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return


def walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_dicts(nested)


def first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def deep_first(record: dict[str, Any], *keys: str) -> Any:
    value = first_value(record, *keys)
    if value not in (None, ""):
        return value
    for d in walk_dicts(record):
        value = first_value(d, *keys)
        if value not in (None, ""):
            return value
    return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def extract_chunk_id(record: dict[str, Any]) -> str | None:
    value = deep_first(record, "chunk_id", "source_chunk_id")
    if value is None:
        # Do not use arbitrary `id` unless it looks UUID-ish/chunk-ish.
        value = record.get("id")
        if isinstance(value, str) and ("-" in value or ":" in value):
            return value
        return None
    return str(value)


def extract_payload(record: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        record.get("output"),
        record.get("parsed"),
        record.get("data"),
        record.get("document"),
        record.get("result"),
        record.get("response"),
    ]
    signal_keys = {
        "concepts", "entities", "relations", "citations", "summary",
        "evidence", "sources", "source",
    }
    for candidate in candidates:
        if isinstance(candidate, dict) and signal_keys.intersection(candidate):
            return candidate
    if signal_keys.intersection(record):
        return record
    for d in walk_dicts(record):
        if {"concepts", "entities", "relations"}.intersection(d):
            return d
    return record


def extract_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "raw_text", "source_text", "markdown", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for d in walk_dicts(record):
        for key in ("text", "content", "raw_text", "source_text"):
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def unwrap_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("name", "title", "label", "concept", "entity", "id", "value"):
            v = value.get(key)
            if isinstance(v, (str, int, float)) and str(v).strip():
                return str(v).strip()
    return None


def extract_concepts(record: dict[str, Any]) -> list[str]:
    payload = extract_payload(record)
    result: list[str] = []
    for key in ("concepts", "entities"):
        for item in as_list(payload.get(key)):
            name = unwrap_name(item)
            if name:
                result.append(name)
    return result


def relation_triplet(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        src, rel, tgt = value[0], value[1], value[2]
    elif isinstance(value, dict):
        src = first_value(value, "source", "src", "from", "subject", "head")
        rel = first_value(value, "relation", "type", "predicate", "rel", "edge")
        tgt = first_value(value, "target", "dst", "to", "object", "tail")
    else:
        return None
    src_name, rel_name, tgt_name = unwrap_name(src), unwrap_name(rel), unwrap_name(tgt)
    if not (src_name and rel_name and tgt_name):
        return None
    return src_name, rel_name, tgt_name


def extract_relations(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    payload = extract_payload(record)
    result: list[tuple[str, str, str]] = []
    for item in as_list(payload.get("relations")):
        triplet = relation_triplet(item)
        if triplet:
            result.append(triplet)
    return result


def extract_bool(record: dict[str, Any], *keys: str) -> bool | None:
    value = deep_first(record, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on", "pass", "passed", "valid"}:
            return True
        if low in {"0", "false", "no", "n", "off", "fail", "failed", "invalid"}:
            return False
    return None


def extract_ontology_errors(record: dict[str, Any]) -> list[str]:
    value = deep_first(record, "ontology_errors", "validation_errors", "errors")
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    count = deep_first(record, "ontology_error_count")
    n = finite_number(count)
    if n and n > 0:
        return [f"ontology_error_count={int(n)}"]
    return []


# ---------------------------------------------------------------------------
# Chunk dataset
# ---------------------------------------------------------------------------


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_json_records(path))


def classify_chunk(record: dict[str, Any], text: str) -> str:
    explicit = deep_first(record, "type", "kind", "label", "chunk_type", "doc_item_type")
    if explicit:
        return str(explicit)
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) < 30 and "\n" not in stripped:
        return "short/heading-like"
    if "|" in stripped and stripped.count("|") >= 4:
        return "table-like"
    if re.search(r"(^|\n)\s*[-*•]\s+", stripped):
        return "list-like"
    return "paragraph-like"


def summarize_chunks(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    lengths: list[int] = []
    types = Counter()
    pages: list[int] = []
    duplicate_ids = 0

    for index, rec in enumerate(chunks, start=1):
        cid = extract_chunk_id(rec) or f"__row_{index}"
        if cid in by_id:
            duplicate_ids += 1
        by_id[cid] = rec
        text = extract_text(rec)
        lengths.append(len(text))
        types[classify_chunk(rec, text)] += 1
        page = deep_first(rec, "page", "page_no", "page_number")
        p = finite_number(page)
        if p is not None:
            pages.append(int(p))

    return {
        "count": len(chunks),
        "unique_chunk_ids": len(by_id),
        "duplicate_chunk_ids": duplicate_ids,
        "text_chars": describe(lengths),
        "empty_chunks": sum(1 for x in lengths if x == 0),
        "under_20_chars": sum(1 for x in lengths if x < 20),
        "under_50_chars": sum(1 for x in lengths if x < 50),
        "under_100_chars": sum(1 for x in lengths if x < 100),
        "chunk_type_distribution": dict(types.most_common()),
        "page_min": min(pages) if pages else None,
        "page_max": max(pages) if pages else None,
        "page_count_observed": len(set(pages)) if pages else None,
    }, by_id


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


SUCCESS_RE = re.compile(
    r"CHUNK LLM SUCCESS .*?chunk=(?P<idx>\d+)/(?:\d+)\s+chunk_id=(?P<chunk_id>\S+)\s+"
    r"attempts=(?P<attempts>\d+)\s+prompt_chars=(?P<prompt_chars>[0-9.]+)\s+"
    r"raw_chars=(?P<raw_chars>[0-9.]+)\s+ttft_ms=(?P<ttft_ms>[0-9.]+)\s+"
    r"wall_ms=(?P<wall_ms>[0-9.]+)\s+total_ms=(?P<total_ms>[0-9.]+)\s+"
    r"load_ms=(?P<load_ms>[0-9.]+)\s+prompt_tokens=(?P<prompt_tokens>[0-9.]+)\s+"
    r"prompt_eval_ms=(?P<prompt_eval_ms>[0-9.]+)\s+output_tokens=(?P<output_tokens>[0-9.]+)\s+"
    r"eval_ms=(?P<eval_ms>[0-9.]+)\s+tokens_per_sec=(?P<tokens_per_sec>[0-9.]+)"
)

CALL_SUCCESS_RE = re.compile(
    r"Ollama call success .*?request_id=(?P<request_id>\S+)\s+attempt=(?P<attempt>\d+)\s+"
    r"schema_valid=(?P<schema>true|false).*?recovered_json=(?P<recovered>True|False|true|false)"
)

END_RE = re.compile(
    r"CHUNK END .*?chunk=(?P<idx>\d+)/(?:\d+)\s+chunk_id=(?P<chunk_id>\S+)\s+"
    r"ontology_errors=(?P<ontology_errors>\d+)\s+citation_preserved=(?P<citation>True|False|true|false)"
)

PROGRESS_RE = re.compile(
    r"PROGRESS .*?completed=(?P<completed>\d+)/(?:\d+).*?success=(?P<success>\d+)\s+"
    r"failed=(?P<failed>\d+)\s+rate_chunks_min=(?P<rate>[0-9.]+)\s+elapsed_s=(?P<elapsed>[0-9.]+)"
)

START_RE = re.compile(r"CHUNK START .*?chunk=(?P<idx>\d+)/(?:\d+)\s+chunk_id=(?P<chunk_id>\S+)\s+text_chars=(?P<text_chars>\d+)")
VRAM_BYTES_RE = re.compile(r"(?:size_vram|vram_bytes)\s*[=:]\s*(?P<bytes>\d+)", re.IGNORECASE)
VRAM_MB_RE = re.compile(r"(?:VRAM|GPU memory).*?(?:used|usage)?\s*[=:]?\s*(?P<mb>[0-9.]+)\s*MB", re.IGNORECASE)

ONTOLOGY_WARN_RE = re.compile(r"ONTOLOGY WARN .*?chunk=(?P<chunk_id>\S+)\s+error_count=(?P<count>\d+)\s+errors=(?P<errors>\[.*\])")


def parse_error_list(text: str) -> list[str]:
    # Usually a Python repr list. Avoid ast dependency concerns? ast is stdlib and safe for literals.
    import ast
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass
    return [text]


def classify_error_message(message: str) -> str:
    low = message.lower()
    if "unresolved relation endpoint" in low:
        return "unresolved_relation_endpoint"
    if "identity collision" in low:
        return "identity_collision"
    if "schema" in low and ("fail" in low or "invalid" in low or "error" in low):
        return "schema_error"
    if "json" in low and ("decode" in low or "parse" in low or "invalid" in low):
        return "json_parse_error"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "connection" in low and ("error" in low or "refused" in low):
        return "connection_error"
    if "out of memory" in low or "cuda oom" in low:
        return "out_of_memory"
    if "citation" in low:
        return "citation_error"
    if "ontology" in low:
        return "ontology_error"
    return "other"


def parse_log(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "chunk_metrics": {},
        "chunk_ends": {},
        "schema_records": [],
        "ontology_messages": defaultdict(list),
        "progress": None,
        "fatal": False,
        "traceback_tail": [],
        "error_counts": Counter(),
        "warning_counts": Counter(),
        "source_path": None,
        "llmwiki_path": None,
        "ontology_path": None,
        "report_path": None,
        "checkpoint_dir": None,
        "model": None,
        "num_ctx": None,
        "max_retries": None,
        "concurrency": None,
        "min_llm_chars": None,
        "vram_bytes_observed": [],
        "vram_mb_observed": [],
    }
    if not path.exists():
        return data

    traceback_buffer: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "CLI resolved source=" in line:
                    m = re.search(
                        r"source=(?P<source>\S+)\s+llmwiki=(?P<llmwiki>\S+)\s+ontology=(?P<ontology>\S+)\s+"
                        r"report=(?P<report>\S+)\s+checkpoint_dir=(?P<checkpoint>\S+)\s+"
                        r"concurrency=(?P<concurrency>\d+)\s+min_llm_chars=(?P<minchars>\d+)",
                        line,
                    )
                    if m:
                        data.update({
                            "source_path": m.group("source"),
                            "llmwiki_path": m.group("llmwiki"),
                            "ontology_path": m.group("ontology"),
                            "report_path": m.group("report"),
                            "checkpoint_dir": m.group("checkpoint"),
                            "concurrency": int(m.group("concurrency")),
                            "min_llm_chars": int(m.group("minchars")),
                        })
                if "Benchmark start" in line:
                    mm = re.search(r"models=\['(?P<model>[^']+)'\].*?num_ctx=(?P<numctx>\d+).*?max_retries=(?P<retries>\d+)", line)
                    if mm:
                        data["model"] = mm.group("model")
                        data["num_ctx"] = int(mm.group("numctx"))
                        data["max_retries"] = int(mm.group("retries"))

                m = SUCCESS_RE.search(line)
                if m:
                    row = {k: (int(v) if k in {"idx", "attempts"} else float(v)) for k, v in m.groupdict().items() if k != "chunk_id"}
                    row["chunk_id"] = m.group("chunk_id")
                    data["chunk_metrics"][row["chunk_id"]] = row
                    continue

                m = CALL_SUCCESS_RE.search(line)
                if m:
                    data["schema_records"].append({
                        "request_id": m.group("request_id"),
                        "attempt": int(m.group("attempt")),
                        "schema_valid": m.group("schema").lower() == "true",
                        "recovered_json": m.group("recovered").lower() == "true",
                    })
                    continue

                m = END_RE.search(line)
                if m:
                    data["chunk_ends"][m.group("chunk_id")] = {
                        "ontology_errors": int(m.group("ontology_errors")),
                        "citation_preserved": m.group("citation").lower() == "true",
                    }
                    continue

                m = ONTOLOGY_WARN_RE.search(line)
                if m:
                    msgs = parse_error_list(m.group("errors"))
                    data["ontology_messages"][m.group("chunk_id")].extend(msgs)
                    for msg in msgs:
                        data["error_counts"][classify_error_message(msg)] += 1
                    continue

                m = PROGRESS_RE.search(line)
                if m:
                    data["progress"] = {
                        "completed": int(m.group("completed")),
                        "success": int(m.group("success")),
                        "failed": int(m.group("failed")),
                        "chunks_per_minute": float(m.group("rate")),
                        "elapsed_seconds": float(m.group("elapsed")),
                    }

                for vm in VRAM_BYTES_RE.finditer(line):
                    data["vram_bytes_observed"].append(int(vm.group("bytes")))
                for vm in VRAM_MB_RE.finditer(line):
                    data["vram_mb_observed"].append(float(vm.group("mb")))

                if "| WARNING" in line or " WARNING  |" in line:
                    if "RapidOCR returned empty result" in line:
                        data["warning_counts"]["ocr_empty"] += 1
                    elif "Orphan pdf_cell" in line:
                        data["warning_counts"]["orphan_pdf_cell"] += 1
                    elif "ONTOLOGY WARN" not in line:
                        data["warning_counts"]["other"] += 1

                if "| ERROR" in line or " ERROR    |" in line:
                    category = classify_error_message(line)
                    data["error_counts"][category] += 1

                if "FATAL benchmark failure" in line:
                    data["fatal"] = True

                if data["fatal"] or traceback_buffer:
                    if "Traceback (most recent call last):" in line or traceback_buffer:
                        traceback_buffer.append(line.rstrip())
                        if len(traceback_buffer) > 80:
                            traceback_buffer.pop(0)
    except OSError:
        pass

    data["ontology_messages"] = dict(data["ontology_messages"])
    data["error_counts"] = dict(data["error_counts"])
    data["warning_counts"] = dict(data["warning_counts"])
    data["traceback_tail"] = traceback_buffer[-40:]
    return data


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def load_checkpoint_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    parse_failures = 0
    if not path.exists():
        return records, parse_failures
    candidates = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )
    for p in candidates:
        before = len(records)
        try:
            for rec in iter_json_records(p):
                records.append(rec)
        except Exception:
            parse_failures += 1
        if len(records) == before:
            # If a JSON-looking file yielded nothing, count it as suspicious.
            try:
                if p.stat().st_size > 0:
                    parse_failures += 1
            except OSError:
                pass
    deduped: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for rec in records:
        cid = extract_chunk_id(rec)
        if cid:
            deduped[cid] = rec
        else:
            anonymous.append(rec)
    return list(deduped.values()) + anonymous, parse_failures


def summarize_checkpoints(records: list[dict[str, Any]], parse_failures: int) -> dict[str, Any]:
    schema_vals: list[bool] = []
    citation_vals: list[bool] = []
    recovered_vals: list[bool] = []
    ontology_error_counts: list[int] = []
    relation_count = 0
    concept_mentions = 0
    unique_concepts: set[str] = set()
    relation_types = Counter()

    for rec in records:
        schema = extract_bool(rec, "schema_valid", "schema_passed", "valid_schema")
        if schema is not None:
            schema_vals.append(schema)
        citation = extract_bool(rec, "citation_preserved", "source_citation_preserved", "citation_valid")
        if citation is not None:
            citation_vals.append(citation)
        recovered = extract_bool(rec, "recovered_json", "json_recovered")
        if recovered is not None:
            recovered_vals.append(recovered)

        errors = extract_ontology_errors(rec)
        ontology_error_counts.append(len(errors))

        concepts = extract_concepts(rec)
        concept_mentions += len(concepts)
        unique_concepts.update(normalize_text(x) for x in concepts if normalize_text(x))
        relations = extract_relations(rec)
        relation_count += len(relations)
        relation_types.update(rel for _, rel, _ in relations)

    return {
        "records": len(records),
        "parse_failures": parse_failures,
        "schema_pass_rate": safe_div(sum(schema_vals), len(schema_vals)) if schema_vals else None,
        "schema_observations": len(schema_vals),
        "citation_preservation_rate": safe_div(sum(citation_vals), len(citation_vals)) if citation_vals else None,
        "citation_observations": len(citation_vals),
        "json_recovery_rate": safe_div(sum(recovered_vals), len(recovered_vals)) if recovered_vals else None,
        "json_recovery_observations": len(recovered_vals),
        "chunks_with_ontology_errors": sum(1 for n in ontology_error_counts if n > 0),
        "ontology_error_count": sum(ontology_error_counts),
        "ontology_compliance_rate": (
            1.0 - safe_div(sum(1 for n in ontology_error_counts if n > 0), len(ontology_error_counts))
            if ontology_error_counts else None
        ),
        "concept_mentions": concept_mentions,
        "unique_normalized_concepts": len(unique_concepts),
        "concept_reuse_ratio": (
            1.0 - safe_div(len(unique_concepts), concept_mentions)
            if concept_mentions else None
        ),
        "relations": relation_count,
        "relation_type_distribution": dict(relation_types.most_common()),
    }


# ---------------------------------------------------------------------------
# Wiki output analysis
# ---------------------------------------------------------------------------


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
RELATION_LINE_RE = re.compile(r"`(?P<relation>[A-Za-z0-9_.:-]+)`\s*(?:→|->)\s*\[\[(?P<target>[^\]|#]+)")
FRONTMATTER_KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$")


def markdown_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def parse_simple_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = FRONTMATTER_KEY_RE.match(line)
        if m and not line.startswith((" ", "\t")):
            result[m.group("key")] = m.group("value").strip().strip('"\'')
    return result


def analyze_wiki(wiki_root: Path, allowed_relations: set[str] | None = None) -> dict[str, Any]:
    if not wiki_root.exists():
        return {
            "path": str(wiki_root),
            "exists": False,
            "documents": 0,
        }

    files = sorted(p for p in wiki_root.rglob("*.md") if p.is_file())
    stems = Counter(normalize_text(p.stem) for p in files)
    stem_set = set(stems)
    title_to_path: dict[str, Path] = {}
    outbound: dict[Path, set[str]] = {}
    inbound_counts = Counter()
    folder_counts = Counter()
    relation_types = Counter()
    frontmatter_missing = Counter()
    untitled = 0
    wikilink_count = 0
    broken_links: list[dict[str, str]] = []

    texts: dict[Path, str] = {}
    titles: dict[Path, str] = {}
    metas: dict[Path, dict[str, str]] = {}

    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texts[p] = text
        title = markdown_title(p, text)
        titles[p] = title
        title_to_path[normalize_text(title)] = p
        title_to_path[normalize_text(p.stem)] = p
        try:
            rel_parent = p.parent.relative_to(wiki_root).as_posix() or "."
        except ValueError:
            rel_parent = p.parent.as_posix()
        folder_counts[rel_parent] += 1
        if normalize_text(p.stem) in {"untitled", "unknown", "none"} or normalize_text(title) in {"untitled", "unknown", "none"}:
            untitled += 1
        meta = parse_simple_frontmatter(text)
        metas[p] = meta
        for key in ("id", "type", "title"):
            if not meta.get(key):
                frontmatter_missing[key] += 1

    for p, text in texts.items():
        targets = {normalize_text(x) for x in WIKILINK_RE.findall(text) if normalize_text(x)}
        outbound[p] = targets
        wikilink_count += len(targets)
        for t in targets:
            target_path = title_to_path.get(t)
            if target_path:
                inbound_counts[target_path] += 1
            else:
                broken_links.append({"source": str(p.relative_to(wiki_root)), "target": t})
        relation_types.update(m.group("relation") for m in RELATION_LINE_RE.finditer(text))

    orphan_files: list[str] = []
    for p in files:
        if not outbound.get(p) and inbound_counts[p] == 0:
            orphan_files.append(str(p.relative_to(wiki_root)))

    unknown_relation_count = None
    unknown_relation_types: dict[str, int] = {}
    if allowed_relations is not None:
        unknown = Counter({k: v for k, v in relation_types.items() if k not in allowed_relations})
        unknown_relation_count = sum(unknown.values())
        unknown_relation_types = dict(unknown.most_common())

    document_count = len(files)
    return {
        "path": str(wiki_root),
        "exists": True,
        "documents": document_count,
        "folder_distribution": dict(folder_counts.most_common()),
        "untitled_documents": untitled,
        "untitled_rate": safe_div(untitled, document_count),
        "duplicate_normalized_stems": sum(1 for _, c in stems.items() if c > 1),
        "duplicate_stem_instances": sum(c - 1 for c in stems.values() if c > 1),
        "frontmatter_missing": dict(frontmatter_missing),
        "wikilinks": wikilink_count,
        "broken_wikilinks": len(broken_links),
        "valid_wikilink_rate": (
            1.0 - safe_div(len(broken_links), wikilink_count)
            if wikilink_count else None
        ),
        "orphan_documents": len(orphan_files),
        "orphan_rate": safe_div(len(orphan_files), document_count),
        "relation_lines": sum(relation_types.values()),
        "relation_type_distribution": dict(relation_types.most_common()),
        "unknown_relation_count": unknown_relation_count,
        "unknown_relation_types": unknown_relation_types,
        "representative_broken_links": broken_links[:10],
        "representative_orphans": orphan_files[:10],
    }


# ---------------------------------------------------------------------------
# Ontology/config/system metadata
# ---------------------------------------------------------------------------


def parse_ontology_relation_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    relations: set[str] = set()
    in_relations = False
    base_indent: int | None = None
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            stripped = raw.strip()
            if stripped == "relations:":
                in_relations = True
                base_indent = indent
                continue
            if in_relations:
                if indent <= (base_indent or 0):
                    break
                # direct child: exactly one mapping key with trailing colon
                m = re.match(r"^([A-Za-z0-9_.:-]+):\s*$", stripped)
                if m and indent > (base_indent or 0):
                    # nested source:/target: are deeper; accept only smallest child indent.
                    if indent <= (base_indent or 0) + 2:
                        relations.add(m.group(1))
    except OSError:
        return set()
    return relations


def parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            value = v.strip()
            # Never export secret-like values.
            if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                result[key] = "<redacted>"
            else:
                result[key] = value
    except OSError:
        pass
    return result


def read_model_list(config_dir: Path) -> tuple[str | None, list[str]]:
    for name in ("ollama-models.txt", "models.txt"):
        path = config_dir / name
        if path.exists():
            values = []
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    values.append(line)
            return str(path), values
    return None, []


def run_command(args: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        cp = subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout)
        return {
            "available": True,
            "return_code": cp.returncode,
            "stdout": cp.stdout.strip()[:12000],
            "stderr": cp.stderr.strip()[:4000],
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def system_snapshot(models: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": now_iso(),
        "note": "Current machine snapshot; not guaranteed to equal benchmark-time utilization.",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or None,
        "ollama_version": run_command(["ollama", "--version"]),
        "nvidia_smi": run_command([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "models": {},
    }
    # /proc/meminfo is cheap and works under Linux/WSL.
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            kv = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    kv[k] = v.strip()
            snapshot["memory"] = {
                "MemTotal": kv.get("MemTotal"),
                "MemAvailable": kv.get("MemAvailable"),
            }
        except OSError:
            pass
    for model in models:
        snapshot["models"][model] = run_command(["ollama", "show", model], timeout=15)
    return snapshot


# ---------------------------------------------------------------------------
# Golden evaluation (optional)
# ---------------------------------------------------------------------------


def load_golden_records(golden_dir: Path | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if golden_dir is None or not golden_dir.exists():
        return result
    files = sorted(
        p for p in golden_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )
    for p in files:
        for rec in iter_json_records(p):
            cid = extract_chunk_id(rec)
            if not cid:
                continue
            reviewed = rec.get("reviewed")
            if reviewed is False:
                continue
            if not (rec.get("expected_concepts") is not None or rec.get("expected_relations") is not None):
                continue
            if "split" not in rec:
                stem = p.stem.lower()
                if "test" in stem:
                    rec["split"] = "test"
                elif "dev" in stem or "valid" in stem:
                    rec["split"] = "dev"
                else:
                    rec["split"] = "unspecified"
            result[cid] = rec
    return result


def normalized_concept_set(values: Any) -> set[str]:
    out: set[str] = set()
    for item in as_list(values):
        name = unwrap_name(item)
        n = normalize_text(name)
        if n:
            out.add(n)
    return out


def normalized_relation_set(values: Any) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for item in as_list(values):
        t = relation_triplet(item)
        if t:
            a, b, c = t
            out.add((normalize_text(a), normalize_text(b), normalize_text(c)))
    return out


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_golden(records: list[dict[str, Any]], golden: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_chunk = {cid: rec for rec in records if (cid := extract_chunk_id(rec))}
    split_stats: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for cid, gold in golden.items():
        pred = by_chunk.get(cid)
        if pred is None:
            continue
        split = str(gold.get("split") or "unspecified")
        expected_c = normalized_concept_set(gold.get("expected_concepts"))
        pred_c = {normalize_text(x) for x in extract_concepts(pred) if normalize_text(x)}
        expected_r = normalized_relation_set(gold.get("expected_relations"))
        pred_r = {(normalize_text(a), normalize_text(b), normalize_text(c)) for a, b, c in extract_relations(pred)}

        ctp, cfp, cfn = len(pred_c & expected_c), len(pred_c - expected_c), len(expected_c - pred_c)
        rtp, rfp, rfn = len(pred_r & expected_r), len(pred_r - expected_r), len(expected_r - pred_r)
        samples.append({
            "chunk_id": cid,
            "split": split,
            "concept": prf(ctp, cfp, cfn),
            "relation": prf(rtp, rfp, rfn),
        })

    for split in sorted(set(x["split"] for x in samples) | {"all"}):
        subset = samples if split == "all" else [x for x in samples if x["split"] == split]
        ctp = sum(x["concept"]["tp"] for x in subset)
        cfp = sum(x["concept"]["fp"] for x in subset)
        cfn = sum(x["concept"]["fn"] for x in subset)
        rtp = sum(x["relation"]["tp"] for x in subset)
        rfp = sum(x["relation"]["fp"] for x in subset)
        rfn = sum(x["relation"]["fn"] for x in subset)
        cmacro = [x["concept"]["f1"] for x in subset if x["concept"]["f1"] is not None]
        rmacro = [x["relation"]["f1"] for x in subset if x["relation"]["f1"] is not None]
        split_stats[split] = {
            "matched_samples": len(subset),
            "concept_micro": prf(ctp, cfp, cfn),
            "relation_micro": prf(rtp, rfp, rfn),
            "concept_macro_f1": statistics.fmean(cmacro) if cmacro else None,
            "relation_macro_f1": statistics.fmean(rmacro) if rmacro else None,
        }

    return {
        "available": bool(golden),
        "golden_samples": len(golden),
        "matched_samples": len(samples),
        "splits": split_stats,
    }


# ---------------------------------------------------------------------------
# Per-model aggregation
# ---------------------------------------------------------------------------


def infer_checkpoint_dir(model_json_path: Path, model_json: dict[str, Any], log_data: dict[str, Any]) -> Path:
    if log_data.get("checkpoint_dir"):
        return Path(str(log_data["checkpoint_dir"]))
    report_path = model_json.get("report_path")
    if report_path:
        return Path(str(report_path)).expanduser().with_suffix(".checkpoints")
    return model_json_path.with_suffix(".checkpoints")


def infer_wiki_dir(llmwiki: Path | None, run_id: str, model: str, log_data: dict[str, Any]) -> Path | None:
    root = llmwiki
    if root is None and log_data.get("llmwiki_path"):
        root = Path(str(log_data["llmwiki_path"]))
    if root is None:
        return None
    return root / "benchmark-runs" / run_id / slugify(model) / "wiki"


def find_model_jsons(run_dir: Path) -> list[Path]:
    result: list[Path] = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in {"quality-summary.json", "report-data.json"}:
            continue
        try:
            value = read_json(p)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("model"):
            result.append(p)
    return result


def fallback_model_entries_from_logs(run_dir: Path) -> list[tuple[str, Path]]:
    entries = []
    for p in sorted(run_dir.glob("*.log")):
        if p.name == "batch.log":
            continue
        data = parse_log(p)
        if data.get("model"):
            entries.append((str(data["model"]), p))
    return entries


def calc_log_performance(log_data: dict[str, Any]) -> dict[str, Any]:
    rows = list(log_data["chunk_metrics"].values())
    metrics = {}
    for key in (
        "attempts", "prompt_chars", "raw_chars", "ttft_ms", "wall_ms", "total_ms",
        "load_ms", "prompt_tokens", "prompt_eval_ms", "output_tokens", "eval_ms",
        "tokens_per_sec",
    ):
        metrics[key] = describe(row.get(key) for row in rows)
    total_output_tokens = sum(float(row.get("output_tokens", 0)) for row in rows)
    total_prompt_tokens = sum(float(row.get("prompt_tokens", 0)) for row in rows)
    total_eval_ms = sum(float(row.get("eval_ms", 0)) for row in rows)
    weighted_tps = safe_div(total_output_tokens, total_eval_ms / 1000.0) if total_eval_ms else None

    attempts = [int(row.get("attempts", 1)) for row in rows]
    ends = list(log_data["chunk_ends"].values())
    schema_rows = log_data["schema_records"]
    ontology_error_total = sum(int(x.get("ontology_errors", 0)) for x in ends)
    unresolved = int(log_data["error_counts"].get("unresolved_relation_endpoint", 0))

    return {
        "observed_success_chunks": len(rows),
        "metrics": metrics,
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_output_tokens": int(total_output_tokens),
        "weighted_tokens_per_sec": weighted_tps,
        "retry_chunks": sum(1 for a in attempts if a > 1),
        "retry_rate": safe_div(sum(1 for a in attempts if a > 1), len(attempts)),
        "schema_pass_rate": (
            safe_div(sum(1 for x in schema_rows if x["schema_valid"]), len(schema_rows))
            if schema_rows else None
        ),
        "json_recovery_rate": (
            safe_div(sum(1 for x in schema_rows if x["recovered_json"]), len(schema_rows))
            if schema_rows else None
        ),
        "citation_preservation_rate": (
            safe_div(sum(1 for x in ends if x["citation_preserved"]), len(ends))
            if ends else None
        ),
        "chunks_with_ontology_errors": sum(1 for x in ends if x["ontology_errors"] > 0),
        "ontology_error_count": ontology_error_total,
        "ontology_compliance_rate": (
            1.0 - safe_div(sum(1 for x in ends if x["ontology_errors"] > 0), len(ends))
            if ends else None
        ),
        "unresolved_relation_endpoint_errors": unresolved,
        "progress": log_data.get("progress"),
    }


def correlate_chunk_and_performance(chunk_map: dict[str, dict[str, Any]], log_data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cid, perf in log_data["chunk_metrics"].items():
        rec = chunk_map.get(cid)
        if not rec:
            continue
        text_chars = len(extract_text(rec))
        rows.append({
            "text_chars": text_chars,
            "ttft_ms": perf.get("ttft_ms"),
            "total_ms": perf.get("total_ms"),
            "output_tokens": perf.get("output_tokens"),
            "tokens_per_sec": perf.get("tokens_per_sec"),
        })
    return {
        "matched_chunks": len(rows),
        "pearson_text_chars_vs_ttft": pearson((r["text_chars"] for r in rows), (r["ttft_ms"] for r in rows)),
        "pearson_text_chars_vs_total_ms": pearson((r["text_chars"] for r in rows), (r["total_ms"] for r in rows)),
        "pearson_text_chars_vs_output_tokens": pearson((r["text_chars"] for r in rows), (r["output_tokens"] for r in rows)),
        "pearson_output_tokens_vs_total_ms": pearson((r["output_tokens"] for r in rows), (r["total_ms"] for r in rows)),
    }


def warmup_comparison(log_data: dict[str, Any], n: int = 20) -> dict[str, Any]:
    rows = sorted(log_data["chunk_metrics"].values(), key=lambda r: int(r.get("idx", 0)))
    if len(rows) <= n:
        return {"available": False, "reason": f"need > {n} successful chunks"}
    first = rows[:n]
    rest = rows[n:]
    first_ttft = describe(r.get("ttft_ms") for r in first)["mean"]
    rest_ttft = describe(r.get("ttft_ms") for r in rest)["mean"]
    first_tps = describe(r.get("tokens_per_sec") for r in first)["mean"]
    rest_tps = describe(r.get("tokens_per_sec") for r in rest)["mean"]
    return {
        "available": True,
        "first_n": n,
        "first_ttft_mean_ms": first_ttft,
        "rest_ttft_mean_ms": rest_ttft,
        "ttft_first_vs_rest_ratio": safe_div(first_ttft, rest_ttft),
        "first_tokens_per_sec_mean": first_tps,
        "rest_tokens_per_sec_mean": rest_tps,
        "tokens_per_sec_first_vs_rest_ratio": safe_div(first_tps, rest_tps),
    }


def length_bucket_analysis(chunk_map: dict[str, dict[str, Any]], log_data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cid, perf in log_data["chunk_metrics"].items():
        rec = chunk_map.get(cid)
        if rec:
            rows.append((len(extract_text(rec)), perf))
    if len(rows) < 8:
        return {"available": False}
    lengths = sorted(x for x, _ in rows)
    q25 = percentile([float(x) for x in lengths], 0.25) or 0
    q50 = percentile([float(x) for x in lengths], 0.50) or 0
    q75 = percentile([float(x) for x in lengths], 0.75) or 0

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for length, perf in rows:
        if length <= q25:
            label = "Q1_shortest"
        elif length <= q50:
            label = "Q2"
        elif length <= q75:
            label = "Q3"
        else:
            label = "Q4_longest"
        buckets[label].append(perf)
    return {
        "available": True,
        "boundaries_chars": {"q25": q25, "q50": q50, "q75": q75},
        "buckets": {
            label: {
                "count": len(items),
                "ttft_ms": describe(x.get("ttft_ms") for x in items),
                "total_ms": describe(x.get("total_ms") for x in items),
                "tokens_per_sec": describe(x.get("tokens_per_sec") for x in items),
                "output_tokens": describe(x.get("output_tokens") for x in items),
            }
            for label, items in buckets.items()
        },
    }


def classify_pipeline_failure(log_data: dict[str, Any], model_json: dict[str, Any]) -> dict[str, Any]:
    tail = "\n".join(log_data.get("traceback_tail") or [])
    low = tail.lower()
    inference_completed = False
    progress = log_data.get("progress")
    if progress and progress.get("completed") and progress.get("success") == progress.get("completed"):
        inference_completed = True

    if "_materialize_model_wiki" in tail or "upsert_document" in tail or "identity collision" in low:
        stage = "wiki_materialization"
    elif "docling" in low or "chunk preparation" in low:
        stage = "ingestion_or_chunking"
    elif "ollama" in low:
        stage = "model_inference"
    elif log_data.get("fatal"):
        stage = "unknown_fatal"
    else:
        stage = None

    failure_type = None
    tail_category = classify_error_message(tail) if tail else None
    if tail_category and tail_category != "other":
        failure_type = tail_category
    else:
        for candidate in (
            "identity_collision", "out_of_memory", "timeout", "connection_error",
            "schema_error", "json_parse_error", "ontology_error",
        ):
            if log_data.get("error_counts", {}).get(candidate):
                failure_type = candidate
                break
    if failure_type is None and log_data.get("fatal"):
        failure_type = "fatal_other"

    status = model_json.get("status")
    pipeline_completed = status == "success" and not log_data.get("fatal")
    return {
        "inference_completed": inference_completed,
        "pipeline_completed": pipeline_completed,
        "fatal": bool(log_data.get("fatal")),
        "failure_stage": stage,
        "failure_type": failure_type,
        "traceback_tail": log_data.get("traceback_tail")[-12:] if log_data.get("traceback_tail") else [],
    }


def select_evidence(
    model: str,
    log_data: dict[str, Any],
    checkpoint_records: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    cp_by_id = {cid: rec for rec in checkpoint_records if (cid := extract_chunk_id(rec))}
    metrics = list(log_data["chunk_metrics"].values())
    if metrics:
        slowest = sorted(metrics, key=lambda r: float(r.get("total_ms", 0)), reverse=True)[: max(2, max_rows // 4)]
        for row in slowest:
            cid = row["chunk_id"]
            source_text = extract_text(chunk_map.get(cid, {}))[:1200]
            payload = extract_payload(cp_by_id.get(cid, {}))
            evidence.append({
                "model": model,
                "reason": "slow_latency_outlier",
                "chunk_id": cid,
                "source_text_excerpt": source_text,
                "metrics": {k: row.get(k) for k in ("ttft_ms", "total_ms", "tokens_per_sec", "output_tokens")},
                "concepts": extract_concepts(payload)[:20],
                "relations": [list(x) for x in extract_relations(payload)[:20]],
            })

    ontology_ranked = sorted(
        ((cid, end.get("ontology_errors", 0)) for cid, end in log_data["chunk_ends"].items()),
        key=lambda x: x[1], reverse=True,
    )
    for cid, count in ontology_ranked[: max(2, max_rows // 4)]:
        if count <= 0:
            continue
        source_text = extract_text(chunk_map.get(cid, {}))[:1200]
        cp = cp_by_id.get(cid, {})
        evidence.append({
            "model": model,
            "reason": "ontology_error_hotspot",
            "chunk_id": cid,
            "ontology_error_count": count,
            "ontology_errors": log_data.get("ontology_messages", {}).get(cid, [])[:10],
            "source_text_excerpt": source_text,
            "concepts": extract_concepts(cp)[:20],
            "relations": [list(x) for x in extract_relations(cp)[:20]],
        })

    retry_ids = [r for r in metrics if int(r.get("attempts", 1)) > 1]
    for row in retry_ids[: max(1, max_rows // 6)]:
        cid = row["chunk_id"]
        evidence.append({
            "model": model,
            "reason": "retry",
            "chunk_id": cid,
            "attempts": row.get("attempts"),
            "source_text_excerpt": extract_text(chunk_map.get(cid, {}))[:1200],
        })

    # Deduplicate and cap.
    seen: set[tuple[str, str]] = set()
    deduped = []
    for item in evidence:
        key = (item["reason"], item.get("chunk_id", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(item)
        if len(deduped) >= max_rows:
            break
    return deduped


# ---------------------------------------------------------------------------
# Model tag / protocol metadata
# ---------------------------------------------------------------------------


def infer_model_tag_metadata(model: str) -> dict[str, Any]:
    low = model.lower()
    parameter_match = re.search(r"(?::|[-_])(?P<size>\d+(?:\.\d+)?)b(?:[-_:]|$)", low)
    quant_match = re.search(r"(?P<quant>q\d+(?:_[a-z0-9]+)+|q\d+)", low)
    family = model.split(":", 1)[0] if ":" in model else re.split(r"[-_]", model, maxsplit=1)[0]
    return {
        "family_from_tag": family or None,
        "parameter_size_from_tag": (parameter_match.group("size") + "B") if parameter_match else None,
        "quantization_from_tag": quant_match.group("quant").upper() if quant_match else None,
        "note": "Values inferred from the model tag are hints, not authoritative ModelCard metadata.",
    }


def detect_task_protocol(records: list[dict[str, Any]], dataset_count: int | None) -> dict[str, Any]:
    question_ids: list[str] = []
    chunk_ids: list[str] = []
    for rec in records:
        qid = deep_first(rec, "question_id", "qid", "test_id")
        if qid is not None:
            question_ids.append(str(qid))
        cid = extract_chunk_id(rec)
        if cid:
            chunk_ids.append(cid)
    if question_ids:
        counts = Counter(question_ids)
        return {
            "mode": "question_answer_benchmark",
            "unique_questions": len(counts),
            "responses": len(question_ids),
            "repeat_count_distribution": dict(Counter(counts.values())),
            "minimum_repeats": min(counts.values()) if counts else None,
            "maximum_repeats": max(counts.values()) if counts else None,
        }
    return {
        "mode": "chunk_extraction_benchmark",
        "unique_chunks_observed": len(set(chunk_ids)),
        "dataset_chunks": dataset_count,
        "note": "This run compares models on shared document chunks rather than the original fixed-question repeated-QA protocol.",
    }

# ---------------------------------------------------------------------------
# Cross-model comparisons and requirement coverage
# ---------------------------------------------------------------------------


def relation_set_by_chunk(records: list[dict[str, Any]]) -> dict[str, set[tuple[str, str, str]]]:
    result = {}
    for rec in records:
        cid = extract_chunk_id(rec)
        if cid:
            result[cid] = {(normalize_text(a), normalize_text(b), normalize_text(c)) for a, b, c in extract_relations(rec)}
    return result


def concept_set_by_chunk(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    result = {}
    for rec in records:
        cid = extract_chunk_id(rec)
        if cid:
            result[cid] = {normalize_text(x) for x in extract_concepts(rec) if normalize_text(x)}
    return result


def jaccard(a: set[Any], b: set[Any]) -> float | None:
    if not a and not b:
        return 1.0
    union = a | b
    return safe_div(len(a & b), len(union))


def cross_model_disagreement(model_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    models = sorted(model_records)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            ac, bc = concept_set_by_chunk(model_records[a]), concept_set_by_chunk(model_records[b])
            ar, br = relation_set_by_chunk(model_records[a]), relation_set_by_chunk(model_records[b])
            common = sorted(set(ac) & set(bc) | (set(ar) & set(br)))
            concept_js, relation_js = [], []
            exact_c, exact_r = 0, 0
            cobs, robs = 0, 0
            for cid in common:
                if cid in ac and cid in bc:
                    val = jaccard(ac[cid], bc[cid])
                    if val is not None:
                        concept_js.append(val)
                    exact_c += ac[cid] == bc[cid]
                    cobs += 1
                if cid in ar and cid in br:
                    val = jaccard(ar[cid], br[cid])
                    if val is not None:
                        relation_js.append(val)
                    exact_r += ar[cid] == br[cid]
                    robs += 1
            pairs.append({
                "model_a": a,
                "model_b": b,
                "shared_chunks": len(common),
                "concept_jaccard_mean": statistics.fmean(concept_js) if concept_js else None,
                "concept_exact_agreement_rate": safe_div(exact_c, cobs),
                "relation_jaccard_mean": statistics.fmean(relation_js) if relation_js else None,
                "relation_exact_agreement_rate": safe_div(exact_r, robs),
            })
    return {"pairs": pairs}


def requirement_coverage(
    project_root: Path,
    run_dir: Path,
    models: list[dict[str, Any]],
    golden_available: bool,
    cloud_dir: Path | None,
    system_data: dict[str, Any] | None,
) -> dict[str, Any]:
    def all_metric(path: list[str]) -> bool:
        for model in models:
            cur: Any = model
            for key in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(key)
            if cur is None:
                return False
        return bool(models)

    items = [
        {
            "requirement": "Model candidates and execution configuration",
            "status": "available" if models else "missing",
            "evidence": "model result JSON/logs + config snapshot",
        },
        {
            "requirement": "Fixed/common input dataset for fair comparison",
            "status": "available" if (run_dir / "chunks.jsonl").exists() else "missing",
            "evidence": "chunks.jsonl",
        },
        {
            "requirement": "Original fixed-question repeated protocol (10 questions × 2 repeats per local model)",
            "status": "not_demonstrated_by_chunk_run",
            "evidence": "Current WikiLLM benchmark is chunk-based unless question_id/repeat fields are present in checkpoints.",
        },
        {
            "requirement": "Latency / TTFT / generation speed",
            "status": "available" if all_metric(["performance", "ttft_ms", "mean"]) else "partial_or_missing",
            "evidence": "CHUNK LLM SUCCESS logs",
        },
        {
            "requirement": "Success/failure/retry and schema/JSON stability",
            "status": "available" if all_metric(["reliability", "success_rate"]) else "partial_or_missing",
            "evidence": "logs/checkpoints",
        },
        {
            "requirement": "Citation / ontology / relation structural quality",
            "status": "available" if all_metric(["quality", "ontology_compliance_rate"]) else "partial_or_missing",
            "evidence": "chunk-end logs/checkpoints/wiki",
        },
        {
            "requirement": "Golden knowledge accuracy (concept/relation P/R/F1)",
            "status": "available" if golden_available else "missing_optional_but_recommended",
            "evidence": "golden dataset + checkpoints",
        },
        {
            "requirement": "Human-rated answer quality / rubric evidence",
            "status": "not_captured_unless_supplied_separately",
            "evidence": "Project brief calls for answer-quality judgment; Golden concept/relation F1 is complementary, not identical.",
        },
        {
            "requirement": "Generated Wiki artifact quality",
            "status": "available" if any(m.get("wiki", {}).get("exists") for m in models) else "missing_or_materialization_failed",
            "evidence": "llmwiki/benchmark-runs/<run>/<model>/wiki",
        },
        {
            "requirement": "GPU/VRAM/RAM/software environment",
            "status": "current_snapshot_only" if system_data else "missing",
            "evidence": "nvidia-smi/ollama/python current snapshot; benchmark-time VRAM requires run-time capture",
        },
        {
            "requirement": "Local vs Cloud API quality/latency/cost/security comparison",
            "status": "available" if cloud_dir and cloud_dir.exists() else "missing",
            "evidence": str(cloud_dir) if cloud_dir else "no cloud result directory supplied",
        },
        {
            "requirement": "Model card/license/architecture/context metadata",
            "status": "partial" if system_data else "missing",
            "evidence": "ollama show output is captured when available; authoritative ModelCard/license may still need separate source",
        },
    ]
    return {
        "items": items,
        "available_count": sum(1 for x in items if x["status"] == "available"),
        "missing_or_partial_count": sum(1 for x in items if x["status"] != "available"),
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def flatten_model_row(model: dict[str, Any]) -> dict[str, Any]:
    perf = model.get("performance", {})
    rel = model.get("reliability", {})
    qual = model.get("quality", {})
    wiki = model.get("wiki", {})
    golden = model.get("golden", {}).get("splits", {}).get("all", {})
    c = golden.get("concept_micro", {}) if isinstance(golden, dict) else {}
    r = golden.get("relation_micro", {}) if isinstance(golden, dict) else {}
    return {
        "model": model.get("model"),
        "batch_status": model.get("batch_status"),
        "expected_chunks": rel.get("expected_chunks"),
        "successful_chunks": rel.get("successful_chunks"),
        "success_rate": rel.get("success_rate"),
        "retry_rate": rel.get("retry_rate"),
        "schema_pass_rate": rel.get("schema_pass_rate"),
        "json_recovery_rate": rel.get("json_recovery_rate"),
        "citation_preservation_rate": qual.get("citation_preservation_rate"),
        "ontology_compliance_rate": qual.get("ontology_compliance_rate"),
        "relation_endpoint_validity_rate": qual.get("relation_endpoint_validity_rate"),
        "mean_ttft_ms": perf.get("ttft_ms", {}).get("mean"),
        "p95_ttft_ms": perf.get("ttft_ms", {}).get("p95"),
        "mean_tokens_per_sec": perf.get("tokens_per_sec", {}).get("mean"),
        "median_tokens_per_sec": perf.get("tokens_per_sec", {}).get("median"),
        "mean_total_ms": perf.get("total_ms", {}).get("mean"),
        "p95_total_ms": perf.get("total_ms", {}).get("p95"),
        "total_output_tokens": perf.get("total_output_tokens"),
        "wiki_documents": wiki.get("documents"),
        "wiki_valid_link_rate": wiki.get("valid_wikilink_rate"),
        "wiki_orphan_rate": wiki.get("orphan_rate"),
        "wiki_untitled_rate": wiki.get("untitled_rate"),
        "golden_samples": model.get("golden", {}).get("matched_samples"),
        "concept_precision": c.get("precision"),
        "concept_recall": c.get("recall"),
        "concept_f1": c.get("f1"),
        "relation_precision": r.get("precision"),
        "relation_recall": r.get("recall"),
        "relation_f1": r.get("f1"),
        "pipeline_completed": model.get("failure", {}).get("pipeline_completed"),
        "failure_stage": model.get("failure", {}).get("failure_stage"),
        "failure_type": model.get("failure", {}).get("failure_type"),
    }


def render_summary(report: dict[str, Any]) -> str:
    run = report["run"]
    lines = [
        "# WikiLLM benchmark report data summary",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Source: `{run.get('source_path') or '-'}`",
        f"- Models: `{len(report.get('models', []))}`",
        f"- Shared chunks: `{report.get('dataset', {}).get('count', 0)}`",
        "",
        "## Model comparison",
        "",
        "| Model | Success | TTFT mean ms | TTFT p95 ms | tok/s mean | Total p95 ms | Schema | Citation | Ontology | Wiki links | Concept F1 | Relation F1 | Pipeline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in report.get("models", []):
        row = flatten_model_row(model)
        lines.append(
            f"| {row['model']} | {pct(row['success_rate'])} | {num(row['mean_ttft_ms'])} | {num(row['p95_ttft_ms'])} | "
            f"{num(row['mean_tokens_per_sec'])} | {num(row['p95_total_ms'])} | {pct(row['schema_pass_rate'])} | "
            f"{pct(row['citation_preservation_rate'])} | {pct(row['ontology_compliance_rate'])} | "
            f"{pct(row['wiki_valid_link_rate'])} | {pct(row['concept_f1'])} | {pct(row['relation_f1'])} | "
            f"{'OK' if row['pipeline_completed'] else (row['failure_stage'] or 'INCOMPLETE')} |"
        )

    lines.extend(["", "## Dataset", ""])
    ds = report.get("dataset", {})
    lines.extend([
        f"- Total chunks: **{ds.get('count', 0)}**",
        f"- Character length mean / median / p95: **{num(ds.get('text_chars', {}).get('mean'))} / {num(ds.get('text_chars', {}).get('median'))} / {num(ds.get('text_chars', {}).get('p95'))}**",
        f"- Under 20 chars: **{ds.get('under_20_chars', 0)}**",
        f"- Under 100 chars: **{ds.get('under_100_chars', 0)}**",
        "",
        "## Cross-model agreement",
        "",
    ])
    pairs = report.get("cross_model", {}).get("pairs", [])
    if pairs:
        lines.extend([
            "| Pair | Concept Jaccard | Concept exact | Relation Jaccard | Relation exact | Shared chunks |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for p in pairs:
            lines.append(
                f"| {p['model_a']} ↔ {p['model_b']} | {pct(p.get('concept_jaccard_mean'))} | "
                f"{pct(p.get('concept_exact_agreement_rate'))} | {pct(p.get('relation_jaccard_mean'))} | "
                f"{pct(p.get('relation_exact_agreement_rate'))} | {p.get('shared_chunks', 0)} |"
            )
    else:
        lines.append("- Cross-model checkpoint comparison was not available.")

    lines.extend(["", "## Requirement coverage", ""])
    for item in report.get("requirement_coverage", {}).get("items", []):
        lines.append(f"- **{item['requirement']}** — `{item['status']}` — {item['evidence']}")

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- `success_rate` is chunk inference completion; `pipeline_completed` is end-to-end completion. Keep them separate.",
        "- Current system snapshot values are not benchmark-time VRAM utilization unless the benchmark logged them during execution.",
        "- Ontology compliance is chunk-level when derived from `CHUNK END`; relation-endpoint validity is only emitted when a relation denominator is available.",
        "- Golden Concept/Relation F1 is emitted only when reviewed golden samples match checkpoint chunk IDs.",
        "- Model disagreement metrics are not accuracy metrics; they identify chunks worth human review.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_report(
    *,
    run_dir: Path,
    project_root: Path,
    llmwiki: Path | None,
    config_dir: Path,
    golden_dir: Path | None,
    cloud_dir: Path | None,
    evidence_per_model: int,
    capture_system: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    config_dir = config_dir.expanduser().resolve()
    llmwiki = llmwiki.expanduser().resolve() if llmwiki else None
    golden_dir = golden_dir.expanduser().resolve() if golden_dir else None
    cloud_dir = cloud_dir.expanduser().resolve() if cloud_dir else None
    run_id = run_dir.name

    chunks_path = run_dir / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    dataset_summary, chunk_map = summarize_chunks(chunks)

    ontology_path = config_dir / "ontology.yaml"
    allowed_relations = parse_ontology_relation_names(ontology_path)
    model_list_path, configured_models = read_model_list(config_dir)
    loop_env_path = config_dir / "loop.env"
    loop_config = parse_env_file(loop_env_path)
    golden = load_golden_records(golden_dir)

    model_json_paths = find_model_jsons(run_dir)
    model_inputs: list[tuple[str, Path | None, dict[str, Any], Path]] = []

    for json_path in model_json_paths:
        value = read_json(json_path)
        model = str(value.get("model"))
        log_path = Path(str(value.get("log_path"))) if value.get("log_path") else json_path.with_suffix(".log")
        if not log_path.is_absolute():
            log_path = (run_dir / log_path).resolve()
        model_inputs.append((model, json_path, value, log_path))

    if not model_inputs:
        for model, log_path in fallback_model_entries_from_logs(run_dir):
            model_inputs.append((model, None, {"model": model, "status": "unknown"}, log_path))

    models: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    model_checkpoint_records: dict[str, list[dict[str, Any]]] = {}
    source_path: str | None = None

    for model, json_path, model_json, log_path in model_inputs:
        log_data = parse_log(log_path)
        source_path = source_path or log_data.get("source_path")
        checkpoint_dir = infer_checkpoint_dir(json_path or log_path.with_suffix(".json"), model_json, log_data)
        checkpoint_records, checkpoint_parse_failures = load_checkpoint_records(checkpoint_dir)
        model_checkpoint_records[model] = checkpoint_records
        checkpoint_summary = summarize_checkpoints(checkpoint_records, checkpoint_parse_failures)
        log_perf = calc_log_performance(log_data)

        summary = model_json.get("result") if isinstance(model_json.get("result"), dict) else {}
        expected_chunks = dataset_summary.get("count") or summary.get("chunks") or (log_data.get("progress") or {}).get("completed")
        successful_chunks = log_perf.get("observed_success_chunks") or (log_data.get("progress") or {}).get("success") or 0
        failed_chunks = max(0, int(expected_chunks or 0) - int(successful_chunks or 0)) if expected_chunks is not None else None

        schema_rate = log_perf.get("schema_pass_rate")
        if schema_rate is None:
            schema_rate = checkpoint_summary.get("schema_pass_rate")
        if schema_rate is None:
            schema_rate = summary.get("schema_pass_rate")

        json_recovery = log_perf.get("json_recovery_rate")
        if json_recovery is None:
            json_recovery = checkpoint_summary.get("json_recovery_rate")
        if json_recovery is None:
            json_recovery = summary.get("json_recovery_rate")

        citation_rate = log_perf.get("citation_preservation_rate")
        if citation_rate is None:
            citation_rate = checkpoint_summary.get("citation_preservation_rate")
        if citation_rate is None:
            citation_rate = summary.get("source_citation_rate")

        ontology_rate = log_perf.get("ontology_compliance_rate")
        if ontology_rate is None:
            ontology_rate = checkpoint_summary.get("ontology_compliance_rate")
        if ontology_rate is None:
            ontology_rate = summary.get("ontology_compliance_rate")

        relation_count = checkpoint_summary.get("relations") or 0
        unresolved = log_perf.get("unresolved_relation_endpoint_errors") or 0
        relation_endpoint_validity = (
            max(0.0, 1.0 - safe_div(unresolved, relation_count))
            if relation_count else None
        )

        wiki_dir = infer_wiki_dir(llmwiki, run_id, model, log_data)
        wiki_summary = analyze_wiki(wiki_dir, allowed_relations or None) if wiki_dir else {"exists": False, "documents": 0, "path": None}
        golden_eval = evaluate_golden(checkpoint_records, golden)
        failure = classify_pipeline_failure(log_data, model_json)

        performance = log_perf["metrics"]
        # Preserve old benchmark summary values when detailed logs are unavailable.
        fallback_means = {
            "ttft_ms": summary.get("mean_ttft_ms"),
            "tokens_per_sec": summary.get("mean_tokens_per_sec"),
            "total_ms": summary.get("mean_total_duration_ms"),
            "load_ms": summary.get("mean_load_duration_ms"),
        }
        for metric_name, fallback_value in fallback_means.items():
            if performance.get(metric_name, {}).get("count", 0) == 0 and fallback_value is not None:
                performance[metric_name] = {
                    "count": None, "mean": fallback_value, "median": None, "std": None,
                    "min": None, "max": None, "p90": None,
                    "p95": summary.get("p95_total_duration_ms") if metric_name == "total_ms" else None,
                    "p99": None, "cv": None, "source": "benchmark_summary_fallback",
                }
        performance.update({
            "total_prompt_tokens": log_perf.get("total_prompt_tokens"),
            "total_output_tokens": log_perf.get("total_output_tokens"),
            "weighted_tokens_per_sec": log_perf.get("weighted_tokens_per_sec"),
            "chunks_per_minute": (log_perf.get("progress") or {}).get("chunks_per_minute"),
            "elapsed_seconds": (log_perf.get("progress") or {}).get("elapsed_seconds"),
            "warmup": warmup_comparison(log_data),
            "chunk_length_correlation": correlate_chunk_and_performance(chunk_map, log_data),
            "chunk_length_buckets": length_bucket_analysis(chunk_map, log_data),
        })

        model_result = {
            "model": model,
            "model_slug": slugify(model),
            "batch_status": model_json.get("status"),
            "paths": {
                "model_json": str(json_path) if json_path else None,
                "log": str(log_path),
                "checkpoint_dir": str(checkpoint_dir),
                "wiki_dir": str(wiki_dir) if wiki_dir else None,
            },
            "model_tag_metadata": infer_model_tag_metadata(model),
            "config_observed": {
                "num_ctx": log_data.get("num_ctx"),
                "max_retries": log_data.get("max_retries"),
                "concurrency": log_data.get("concurrency"),
                "min_llm_chars": log_data.get("min_llm_chars"),
            },
            "task_protocol": detect_task_protocol(checkpoint_records, dataset_summary.get("count")),
            "benchmark_time_vram": {
                "bytes": describe(log_data.get("vram_bytes_observed", [])),
                "mb": describe(log_data.get("vram_mb_observed", [])),
                "note": "Only populated if benchmark logs recorded VRAM/size_vram values during the run.",
            },
            "performance": performance,
            "reliability": {
                "expected_chunks": expected_chunks,
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "success_rate": safe_div(successful_chunks, expected_chunks),
                "retry_chunks": log_perf.get("retry_chunks"),
                "retry_rate": log_perf.get("retry_rate"),
                "schema_pass_rate": schema_rate,
                "json_recovery_rate": json_recovery,
                "checkpoint_records": checkpoint_summary.get("records"),
                "checkpoint_parse_failures": checkpoint_summary.get("parse_failures"),
            },
            "quality": {
                "citation_preservation_rate": citation_rate,
                "ontology_compliance_rate": ontology_rate,
                "chunks_with_ontology_errors": log_perf.get("chunks_with_ontology_errors"),
                "ontology_error_count": log_perf.get("ontology_error_count"),
                "unresolved_relation_endpoint_errors": unresolved,
                "relation_endpoint_validity_rate": relation_endpoint_validity,
                "concept_mentions": checkpoint_summary.get("concept_mentions"),
                "unique_normalized_concepts": checkpoint_summary.get("unique_normalized_concepts"),
                "concept_reuse_ratio": checkpoint_summary.get("concept_reuse_ratio"),
                "relations": relation_count,
                "relation_type_distribution": checkpoint_summary.get("relation_type_distribution"),
            },
            "wiki": wiki_summary,
            "golden": golden_eval,
            "errors": {
                "categories": log_data.get("error_counts"),
                "warnings": log_data.get("warning_counts"),
                "ontology_message_examples": [
                    {"chunk_id": cid, "errors": msgs[:5]}
                    for cid, msgs in list(log_data.get("ontology_messages", {}).items())[:10]
                ],
            },
            "failure": failure,
        }
        models.append(model_result)
        evidence_rows.extend(
            select_evidence(model, log_data, checkpoint_records, chunk_map, max_rows=evidence_per_model)
        )

    current_system = system_snapshot([m["model"] for m in models]) if capture_system else None

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "purpose": "Compact benchmark/report reference data generated from existing WikiLLM outputs without rerunning inference.",
        "run": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "source_path": source_path,
            "source_sha256": sha256_file(Path(source_path)) if source_path and Path(source_path).exists() else None,
            "models_observed": [m["model"] for m in models],
        },
        "dataset": dataset_summary,
        "config": {
            "config_dir": str(config_dir),
            "ontology_path": str(ontology_path) if ontology_path.exists() else None,
            "ontology_relation_names": sorted(allowed_relations),
            "model_list_path": model_list_path,
            "configured_models": configured_models,
            "loop_env_path": str(loop_env_path) if loop_env_path.exists() else None,
            "loop_env_redacted": loop_config,
        },
        "models": models,
        "cross_model": cross_model_disagreement(model_checkpoint_records),
        "golden": {
            "directory": str(golden_dir) if golden_dir else None,
            "reviewed_samples_loaded": len(golden),
            "available": bool(golden),
        },
        "cloud": {
            "directory": str(cloud_dir) if cloud_dir else None,
            "available": bool(cloud_dir and cloud_dir.exists()),
            "note": "Cloud quality/latency/cost is not fabricated; supply a cloud result directory to include it in coverage.",
        },
        "system_snapshot": current_system,
    }
    report["requirement_coverage"] = requirement_coverage(
        project_root, run_dir, models, bool(golden), cloud_dir, current_system
    )
    return report, evidence_rows


def write_outputs(report: dict[str, Any], evidence_rows: list[dict[str, Any]], output_dir: Path, make_zip: bool) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report-data.json"
    comparison_csv = output_dir / "model-comparison.csv"
    evidence_jsonl = output_dir / "evidence.jsonl"
    summary_md = output_dir / "report-summary.md"

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [flatten_model_row(x) for x in report.get("models", [])]
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with evidence_jsonl.open("w", encoding="utf-8") as f:
        for row in evidence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_md.write_text(render_summary(report), encoding="utf-8")

    outputs = {
        "report_data": str(report_json),
        "model_comparison": str(comparison_csv),
        "evidence": str(evidence_jsonl),
        "summary": str(summary_md),
    }
    if make_zip:
        zip_path = output_dir / "share-package.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in (report_json, comparison_csv, evidence_jsonl, summary_md):
                zf.write(p, arcname=p.name)
        outputs["zip"] = str(zip_path)
    return outputs


def infer_project_root(run_dir: Path) -> Path:
    # Expected: <project>/app/outputs/benchmark/batch-runs/<run_id>
    parts = list(run_dir.resolve().parts)
    try:
        idx = parts.index("app")
        if idx > 0:
            return Path(*parts[:idx])
    except ValueError:
        pass
    return Path.cwd().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate an existing WikiLLM benchmark run into compact report/reference data."
    )
    parser.add_argument("--run-dir", required=True, help="Existing app/outputs/benchmark/batch-runs/<RUN_ID>")
    parser.add_argument("--project-root", default=None, help="Project root; auto-detected from run-dir when omitted")
    parser.add_argument("--llmwiki", default=None, help="Independent llmwiki Vault root; optional but needed for generated-Wiki analysis")
    parser.add_argument("--config-dir", default=None, help="Config directory; defaults to <project-root>/config")
    parser.add_argument("--golden-dir", default=None, help="Optional reviewed Golden JSON/JSONL directory")
    parser.add_argument("--cloud-dir", default=None, help="Optional cloud benchmark result directory")
    parser.add_argument("--output-dir", default=None, help="Defaults to <project-root>/app/outputs/report-data/<RUN_ID>")
    parser.add_argument("--evidence-per-model", type=int, default=12, help="Small number of representative evidence rows per model")
    parser.add_argument("--no-system-snapshot", action="store_true", help="Do not run nvidia-smi/ollama metadata commands")
    parser.add_argument("--no-zip", action="store_true", help="Do not build share-package.zip")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else infer_project_root(run_dir)
    config_dir = Path(args.config_dir).expanduser().resolve() if args.config_dir else project_root / "config"
    llmwiki = Path(args.llmwiki).expanduser().resolve() if args.llmwiki else None
    golden_dir = Path(args.golden_dir).expanduser().resolve() if args.golden_dir else None
    cloud_dir = Path(args.cloud_dir).expanduser().resolve() if args.cloud_dir else None
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root / "app" / "outputs" / "report-data" / run_dir.name
    )

    report, evidence = build_report(
        run_dir=run_dir,
        project_root=project_root,
        llmwiki=llmwiki,
        config_dir=config_dir,
        golden_dir=golden_dir,
        cloud_dir=cloud_dir,
        evidence_per_model=max(0, args.evidence_per_model),
        capture_system=not args.no_system_snapshot,
    )
    outputs = write_outputs(report, evidence, output_dir, make_zip=not args.no_zip)

    print(json.dumps({
        "run_id": run_dir.name,
        "models": [x.get("model") for x in report.get("models", [])],
        "chunks": report.get("dataset", {}).get("count"),
        "golden_samples": report.get("golden", {}).get("reviewed_samples_loaded"),
        "outputs": outputs,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
