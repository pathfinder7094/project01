from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.eval.quality_eval import evaluate_records, load_checkpoint_records, load_golden_dir
from app.utils.logging import configure_logging, get_logger


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "model"


def _load_model_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("model") else None


def _checkpoint_dir(result_json: Path, result: dict[str, Any]) -> Path:
    report_path = result.get("report_path")
    if report_path:
        report = Path(str(report_path)).expanduser()
        return report.with_suffix(".checkpoints")
    return result_json.with_suffix(".checkpoints")


def _merge_fallback_summary(quality: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Use benchmark summary when old checkpoints do not expose a metric.

    This lets the evaluator work with both old and new benchmark runs.
    """
    summary = result.get("result") or {}
    mapping = {
        "schema_pass_rate": "schema_pass_rate",
        "citation_presence_rate": "source_citation_rate",
        "ontology_compliance_rate": "ontology_compliance_rate",
    }
    for target, source in mapping.items():
        if quality.get(target) is None and summary.get(source) is not None:
            quality[target] = summary[source]
    return quality


def evaluate_model_result(
    result_json: Path,
    *,
    golden_dir: Path | None,
) -> dict[str, Any] | None:
    result = _load_model_result(result_json)
    if result is None:
        return None

    checkpoint_dir = _checkpoint_dir(result_json, result)
    records, parse_failures = load_checkpoint_records(checkpoint_dir)
    golden = load_golden_dir(golden_dir)
    quality = evaluate_records(records, parse_failures=parse_failures, golden=golden).to_dict()
    quality = _merge_fallback_summary(quality, result)

    # If checkpoint-based structural_score was impossible, compute it again from
    # fallback summary values without pretending missing metrics are zero.
    if quality.get("structural_score") is None:
        parts = [
            (quality.get("schema_pass_rate"), 0.30),
            (quality.get("citation_presence_rate"), 0.25),
            (quality.get("ontology_compliance_rate"), 0.25),
            (quality.get("relation_endpoint_validity_rate"), 0.20),
        ]
        available = [(float(v), w) for v, w in parts if v is not None]
        if available:
            quality["structural_score"] = sum(v * w for v, w in available) / sum(w for _, w in available)

    summary = result.get("result") or {}
    return {
        "model": result["model"],
        "status": result.get("status"),
        "result_json": str(result_json),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_dir_exists": checkpoint_dir.exists(),
        "quality": quality,
        "performance": {
            "chunks": summary.get("chunks"),
            "mean_ttft_ms": summary.get("mean_ttft_ms"),
            "mean_tokens_per_sec": summary.get("mean_tokens_per_sec"),
            "mean_total_duration_ms": summary.get("mean_total_duration_ms"),
            "p95_total_duration_ms": summary.get("p95_total_duration_ms"),
            "json_recovery_rate": summary.get("json_recovery_rate"),
        },
    }


def render_markdown(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# WikiLLM Quality Benchmark",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Quality",
        "",
        "| Model | Structural | Knowledge | Schema | Citation | Ontology | Relation endpoint | Concept F1 | Relation F1 | Golden |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        q = row["quality"]
        lines.append(
            f"| {row['model']} | {_pct(q.get('structural_score'))} | {_pct(q.get('knowledge_score'))} | "
            f"{_pct(q.get('schema_pass_rate'))} | {_pct(q.get('citation_presence_rate'))} | "
            f"{_pct(q.get('ontology_compliance_rate'))} | {_pct(q.get('relation_endpoint_validity_rate'))} | "
            f"{_pct(q.get('concept_f1'))} | {_pct(q.get('relation_f1'))} | {q.get('golden_samples_matched', 0)} |"
        )

    lines.extend([
        "",
        "## Performance",
        "",
        "| Model | Chunks | TTFT ms | tok/s | Mean total ms | P95 total ms | JSON recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        p = row["performance"]
        lines.append(
            f"| {row['model']} | {p.get('chunks', '-')} | {_num(p.get('mean_ttft_ms'))} | "
            f"{_num(p.get('mean_tokens_per_sec'))} | {_num(p.get('mean_total_duration_ms'))} | "
            f"{_num(p.get('p95_total_duration_ms'))} | {_pct(p.get('json_recovery_rate'))} |"
        )

    lines.extend([
        "",
        "## Score definition",
        "",
        "`Structural score = schema 30% + citation 25% + ontology 25% + relation-endpoint validity 20%`.",
        "Missing metrics are excluded from the denominator instead of being scored as zero.",
        "",
        "`Knowledge score = concept F1 45% + relation F1 45% + citation 10%`.",
        "Knowledge score is only meaningful when matching golden samples exist.",
        "",
    ])
    return "\n".join(lines)


def evaluate_batch_run(
    *,
    run_dir: Path,
    golden_dir: Path | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> list[dict[str, Any]]:
    logger = get_logger(__name__)
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    candidates = sorted(
        p for p in run_dir.glob("*.json")
        if p.name not in {"quality-summary.json", "chunks.json"}
    )
    rows: list[dict[str, Any]] = []
    for path in candidates:
        row = evaluate_model_result(path, golden_dir=golden_dir)
        if row is None:
            continue
        rows.append(row)
        logger.info(
            "QUALITY model=%s structural=%s knowledge=%s checkpoints=%s",
            row["model"],
            row["quality"].get("structural_score"),
            row["quality"].get("knowledge_score"),
            row["quality"].get("checkpoint_records"),
        )

    output_json = output_json or run_dir / "quality-summary.json"
    output_md = output_md or run_dir / "quality-summary.md"
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(run_dir, rows), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an existing WikiLLM batch benchmark run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--golden-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    configure_logging(component=__name__)
    rows = evaluate_batch_run(
        run_dir=Path(args.run_dir),
        golden_dir=Path(args.golden_dir).expanduser().resolve() if args.golden_dir else None,
        output_json=Path(args.output_json).expanduser().resolve() if args.output_json else None,
        output_md=Path(args.output_md).expanduser().resolve() if args.output_md else None,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
