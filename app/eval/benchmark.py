from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from app.models.schema import BenchmarkMetrics, BenchmarkResult, KnowledgeExtraction
from app.llm.base import StructuredLLMClient
from app.ollama.client import OllamaClient
from app.openai_client.client import OpenAIClient
from app.pipeline.ingest import (
    SYSTEM_PROMPT,
    build_prompt,
    chunk_document,
    load_ontology,
    retrieve_existing_context,
    citation_preserved,
    source_id,
    source_ref_defaults,
    materialize_relations,
    build_knowledge_registry,
    upsert_document,
    append_source_note,
    upsert_index,
    append_log,
    _title_lookup,
    _name_key,
)
from app.utils.paths import APP_ROOT, DEFAULT_LLMWIKI_ROOT, ensure_layout
from app.utils.logging import configure_logging, get_logger, log_environment
from app.pipeline.chunk_cache import load_chunk_cache
from app.pipeline.markdown import document_from_markdown, write_index
from app.eval.wiki_quality import evaluate_wiki


DEFAULT_MODELS = [
    "qwen3:4b-q4_K_M",
    "gemma3:4b-it-q4_K_M",
]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def aggregate(provider: str, model: str, document: str, source_uuid: UUID, metrics: list[BenchmarkMetrics]) -> BenchmarkResult:
    success = [m for m in metrics if m.success]
    schema_ok = [m for m in metrics if m.schema_valid]
    recovered = [m for m in metrics if m.recovered_json]
    citations = [m for m in metrics if m.source_citation_preserved]
    ttft = [m.ttft_ms for m in success if m.ttft_ms is not None]
    tps = [m.tokens_per_sec for m in success if m.tokens_per_sec is not None]
    total = [m.total_duration_ms for m in success if m.total_duration_ms is not None]
    prompt_eval_ms = [m.prompt_eval_duration_ms for m in success if m.prompt_eval_duration_ms is not None]
    output_tokens = [m.eval_count for m in success if m.eval_count is not None]
    ontology = [m.ontology_compliance_rate for m in success]

    count = max(len(metrics), 1)
    return BenchmarkResult(
        provider=provider,
        model=model,
        document=document,
        source_id=source_uuid,
        chunks=len(metrics),
        successful_chunks=len(success),
        schema_pass_rate=len(schema_ok) / count,
        json_recovery_rate=len(recovered) / count,
        source_citation_rate=len(citations) / count,
        ontology_compliance_rate=statistics.mean(ontology) if ontology else 0.0,
        mean_ttft_ms=statistics.mean(ttft) if ttft else None,
        mean_tokens_per_sec=statistics.mean(tps) if tps else None,
        mean_prompt_eval_ms=statistics.mean(prompt_eval_ms) if prompt_eval_ms else None,
        mean_output_tokens=statistics.mean(output_tokens) if output_tokens else None,
        p95_output_tokens=percentile([float(value) for value in output_tokens], 0.95) if output_tokens else None,
        max_output_tokens=max(output_tokens) if output_tokens else None,
        mean_total_duration_ms=statistics.mean(total) if total else None,
        p95_total_duration_ms=percentile(total, 0.95),
        details=metrics,
    )


def write_report(*, output_path: Path, source: Path, results: list[BenchmarkResult]) -> None:
    lines = [
        "# WikiLLM Provider Benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"Source: `{source}`",
        "",
        "> Same Docling chunks and the same extraction schema/prompt are used for every model.",
        "> Performance values are model/runtime measurements from this run; they are not model quality rankings.",
        "",
        "## Summary",
        "",
        "| Provider | Model | Chunks | Schema pass | Citation | Ontology | TTFT ms | tok/s | Prompt ms | Mean out tok | P95 out tok | Max out tok | Chunks/min | Mean total ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    def num(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    for result in results:
        lines.append(
            "| {provider} | {model} | {chunks} | {schema} | {citation} | {ontology} | {ttft} | {tps} | {prompt} | {out_mean} | {out_p95} | {out_max} | {cpm} | {total} |".format(
                provider=result.provider,
                model=result.model,
                chunks=result.chunks,
                schema=pct(result.schema_pass_rate),
                citation=pct(result.source_citation_rate),
                ontology=pct(result.ontology_compliance_rate),
                ttft=num(result.mean_ttft_ms),
                tps=num(result.mean_tokens_per_sec),
                prompt=num(result.mean_prompt_eval_ms),
                out_mean=num(result.mean_output_tokens),
                out_p95=num(result.p95_output_tokens),
                out_max=result.max_output_tokens if result.max_output_tokens is not None else "-",
                cpm=num(result.effective_chunks_per_min),
                total=num(result.mean_total_duration_ms),
            )
        )

    lines += ["", "## Wiki outputs", ""]
    for result in results:
        quality = result.wiki_quality
        lines += [f"### {result.provider} / {result.model}", "", f"- Wiki sandbox: `{result.wiki_root or '-'}`"]
        lines += [
            f"- Relations emitted/retained/dropped-unresolved: {result.raw_relation_count}/{result.retained_relation_count}/{result.dropped_unresolved_relation_count}",
        ]
        if quality is not None:
            lines += [
                f"- Documents: {quality.documents} (concepts={quality.concepts}, entities={quality.entities}, sources={quality.sources})",
                f"- Relations: {quality.relations}; relation integrity: {quality.relation_integrity_rate:.1%}",
                f"- Chunk reference coverage: {quality.chunk_reference_coverage:.1%}",
                f"- Sourced document rate: {quality.sourced_document_rate:.1%}",
                f"- Orphan document rate: {quality.orphan_document_rate:.1%}",
            ]
        lines.append("")

    lines += ["", "## Per-chunk details", ""]
    for result in results:
        lines += [
            f"### {result.provider} / {result.model}",
            "",
            "| Chunk | Valid | Citation | Ontology | TTFT ms | Load ms | Prompt tok | Prompt ms | Output tok | Eval ms | tok/s | Total ms | Error |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for detail in result.details:
            lines.append(
                "| {chunk} | {valid} | {citation} | {ontology:.1%} | {ttft} | {load} | {ptok} | {pms} | {otok} | {ems} | {tps} | {total} | {error} |".format(
                    chunk=detail.chunk_id,
                    valid="yes" if detail.schema_valid else "no",
                    citation="yes" if detail.source_citation_preserved else "no",
                    ontology=detail.ontology_compliance_rate,
                    ttft=num(detail.ttft_ms),
                    load=num(detail.load_duration_ms),
                    ptok=detail.prompt_eval_count if detail.prompt_eval_count is not None else "-",
                    pms=num(detail.prompt_eval_duration_ms),
                    otok=detail.eval_count if detail.eval_count is not None else "-",
                    ems=num(detail.eval_duration_ms),
                    tps=num(detail.tokens_per_sec),
                    total=num(detail.total_duration_ms),
                    error=(detail.error or "").replace("\n", " ")[:180],
                )
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return value or "model"


def _checkpoint_path(checkpoint_dir: Path, model: str) -> Path:
    return checkpoint_dir / f"{_slug(model)}.jsonl"


def _read_checkpoint_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if item.get("_type") == "meta":
            return item
    return None


def _checkpoint_runtime_matches(
    meta: dict[str, Any],
    *,
    model: str,
    source_uuid: UUID,
    num_ctx: int,
    concurrency: int,
    num_batch: int | None,
    num_predict: int | None,
) -> bool:
    if meta.get("model") != model or meta.get("source_id") != str(source_uuid):
        return False
    try:
        if int(meta.get("num_ctx", -1)) != num_ctx:
            return False
        # Legacy checkpoints were always concurrency=1 with Ollama defaults.
        meta_concurrency = int(meta.get("concurrency", 1))
        meta_num_batch = meta.get("num_batch")
        meta_num_predict = meta.get("num_predict")
        if meta_concurrency != concurrency:
            return False
        if (int(meta_num_batch) if meta_num_batch is not None else None) != num_batch:
            return False
        if (int(meta_num_predict) if meta_num_predict is not None else None) != num_predict:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _archive_incompatible_checkpoint(
    checkpoint_path: Path,
    extraction_path: Path,
    *,
    model: str,
    source_uuid: UUID,
    num_ctx: int,
    concurrency: int,
    num_batch: int | None,
    num_predict: int | None,
) -> bool:
    """Archive partial metrics/extractions when runtime tuning changed.

    Mixing concurrency or Ollama runtime knobs in one benchmark would make the
    performance summary misleading.  Completed legacy runs can still be read
    when their runtime matches the default (concurrency=1, no explicit knobs).
    """
    meta = _read_checkpoint_meta(checkpoint_path)
    if meta is None or _checkpoint_runtime_matches(
        meta,
        model=model,
        source_uuid=source_uuid,
        num_ctx=num_ctx,
        concurrency=concurrency,
        num_batch=num_batch,
        num_predict=num_predict,
    ):
        return False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = get_logger(__name__)
    for path in (checkpoint_path, extraction_path):
        if not path.exists():
            continue
        archived = path.with_name(f"{path.name}.incompatible-{stamp}.bak")
        path.replace(archived)
        logger.warning(
            "CHECKPOINT runtime changed; archived old checkpoint model=%s old=%s archived=%s",
            model,
            path,
            archived,
        )
    return True

def _load_checkpoint(
    path: Path,
    *,
    model: str,
    source_uuid: UUID,
    num_ctx: int,
    concurrency: int = 1,
    num_batch: int | None = None,
    num_predict: int | None = None,
) -> dict[UUID, BenchmarkMetrics]:
    logger = get_logger(__name__)
    if not path.exists():
        return {}

    loaded: dict[UUID, BenchmarkMetrics] = {}
    meta_seen = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("CHECKPOINT ignoring invalid JSON path=%s line=%d", path, line_no)
            continue
        if item.get("_type") == "meta":
            meta_seen = True
            if not _checkpoint_runtime_matches(
                item,
                model=model,
                source_uuid=source_uuid,
                num_ctx=num_ctx,
                concurrency=concurrency,
                num_batch=num_batch,
                num_predict=num_predict,
            ):
                logger.warning("CHECKPOINT incompatible runtime; ignoring path=%s", path)
                return {}
            continue
        if item.get("_type") != "metric":
            continue
        try:
            metric = BenchmarkMetrics.model_validate(item["data"])
        except Exception as exc:
            logger.warning("CHECKPOINT invalid metric path=%s line=%d error=%s", path, line_no, exc)
            continue
        loaded[metric.chunk_id] = metric

    if not meta_seen:
        logger.warning("CHECKPOINT missing metadata; ignoring path=%s", path)
        return {}
    logger.info("CHECKPOINT loaded model=%s completed=%d path=%s", model, len(loaded), path)
    return loaded


def _ensure_checkpoint_meta(
    path: Path,
    *,
    model: str,
    source_uuid: UUID,
    num_ctx: int,
    concurrency: int = 1,
    num_batch: int | None = None,
    num_predict: int | None = None,
) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_type": "meta",
        "model": model,
        "source_id": str(source_uuid),
        "num_ctx": num_ctx,
        "concurrency": concurrency,
        "num_batch": num_batch,
        "num_predict": num_predict,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_checkpoint(path: Path, metric: BenchmarkMetrics) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"_type": "metric", "data": metric.model_dump(mode="json")}, ensure_ascii=False) + "\n")
        handle.flush()


def _extraction_path(checkpoint_dir: Path, model: str) -> Path:
    return checkpoint_dir / f"{_slug(model)}.extractions.jsonl"


def _load_extractions(path: Path) -> dict[UUID, KnowledgeExtraction]:
    loaded: dict[UUID, KnowledgeExtraction] = {}
    if not path.exists():
        return loaded
    logger = get_logger(__name__)
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            chunk_id = UUID(str(item["chunk_id"]))
            loaded[chunk_id] = KnowledgeExtraction.model_validate(item["extraction"])
        except Exception as exc:
            logger.warning("EXTRACTION CHECKPOINT invalid path=%s line=%d error=%s", path, line_no, exc)
    return loaded


def _append_extraction(path: Path, chunk_id: UUID, extraction: KnowledgeExtraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "chunk_id": str(chunk_id),
            "extraction": extraction.model_dump(mode="json"),
        }, ensure_ascii=False) + "\n")
        handle.flush()



def _normalize_model_relations(
    extractions: dict[UUID, KnowledgeExtraction],
    ontology: Any,
) -> tuple[dict[UUID, KnowledgeExtraction], dict[UUID, list[str]], int, int]:
    """Validate relation endpoints against the model's complete extracted node set.

    The prompt contract requires relation endpoints to name extracted concept/entity
    nodes.  Some models emit outline labels or arbitrary prose as relation endpoints;
    those relations cannot be materialized into a stable Wiki graph.  Build the full
    model node registry first so legitimate cross-chunk relations survive, then drop
    only endpoints that never became a concept/entity anywhere in this model run.
    """
    known_documents: dict[UUID, Any] = {}
    known_names: dict[str, tuple[UUID, Any]] = {}

    for extraction in extractions.values():
        node_only = extraction.model_copy(update={"relations": []})
        documents, _ = materialize_relations(node_only, ontology)
        for document in documents:
            known_documents[document.id] = document
            known_names[_name_key(document.title)] = (document.id, document.type)
            for alias in document.aliases:
                known_names.setdefault(_name_key(alias), (document.id, document.type))

    cleaned: dict[UUID, KnowledgeExtraction] = {}
    errors_by_chunk: dict[UUID, list[str]] = {}
    raw_relations = 0
    retained_relations = 0
    for chunk_id, extraction in extractions.items():
        raw_relations += len(extraction.relations)
        kept = [
            relation
            for relation in extraction.relations
            if _name_key(relation.source) in known_names and _name_key(relation.target) in known_names
        ]
        retained_relations += len(kept)
        normalized = extraction.model_copy(update={"relations": kept})
        cleaned[chunk_id] = normalized
        _, ontology_errors = materialize_relations(
            normalized,
            ontology,
            known_documents=known_documents,
            known_names=known_names,
        )
        errors_by_chunk[chunk_id] = ontology_errors

    return cleaned, errors_by_chunk, raw_relations, retained_relations

def _materialize_model_wiki(
    *,
    model: str,
    model_root: Path,
    source: Path,
    source_rel: str,
    source_uuid: UUID,
    chunks: list[Any],
    extractions: dict[UUID, KnowledgeExtraction],
    ontology: Any,
) -> tuple[str, Any, list[str]]:
    logger = get_logger(__name__)
    # The benchmark Wiki is reproducible output, so rebuild it from the extraction journal.
    if (model_root / "wiki").exists():
        shutil.rmtree(model_root / "wiki")
    ensure_layout(model_root)
    title_lookup = _title_lookup(model_root)
    known_documents, known_names = build_knowledge_registry(model_root)
    index_entries: list[tuple[str, str, UUID]] = []
    ontology_errors: list[str] = []

    for chunk in chunks:
        extraction = extractions.get(chunk.id)
        if extraction is None:
            continue
        documents, errors = materialize_relations(
            extraction,
            ontology,
            known_documents=known_documents,
            known_names=known_names,
        )
        ontology_errors.extend(errors)
        for document in documents:
            rel_path = upsert_document(model_root, document, title_lookup=title_lookup)
            index_entries.append((document.title, rel_path, document.id))
            merged_path = model_root / "wiki" / rel_path
            try:
                merged = document_from_markdown(merged_path)
                known_documents[merged.id] = merged
                known_names[_name_key(merged.title)] = (merged.id, merged.type)
                for alias in merged.aliases:
                    known_names.setdefault(_name_key(alias), (merged.id, merged.type))
            except Exception:
                logger.exception("BENCHMARK WIKI registry refresh failed path=%s", merged_path)

    source_note = append_source_note(model_root, source, source_uuid, chunks)
    upsert_index(model_root, index_entries + [(source.stem, source_note, source_uuid)])
    append_log(
        model_root,
        f"benchmark sandbox generated with `{model}` from `{source_rel}`; extracted_chunks={len(extractions)}/{len(chunks)}",
    )
    write_index(model_root / "wiki")
    readme = model_root / "README.md"
    readme.write_text(
        "# Benchmark Wiki Sandbox\n\n"
        f"- Model: `{model}`\n"
        f"- Source: `{source}`\n"
        "- This Wiki is isolated benchmark output and is not canonical knowledge.\n"
        "- Promote/merge into the canonical `llmwiki/wiki/` only after review.\n",
        encoding="utf-8",
    )
    quality = evaluate_wiki(model_root / "wiki", {chunk.id for chunk in chunks})
    return str(model_root), quality, ontology_errors


def _progress_log(
    *,
    model: str,
    completed: int,
    total: int,
    succeeded: int,
    failed: int,
    started_monotonic: float,
    last_chunk: str,
) -> None:
    logger = get_logger(__name__)
    elapsed = max(time.monotonic() - started_monotonic, 0.001)
    rate = completed / elapsed if completed else 0.0
    remaining = max(total - completed, 0)
    eta_s = remaining / rate if rate > 0 else None
    logger.info(
        "PROGRESS model=%s completed=%d/%d progress=%.2f%% success=%d failed=%d rate_chunks_min=%.2f elapsed_s=%.1f eta_s=%s last_chunk=%s",
        model,
        completed,
        total,
        (completed / total * 100.0) if total else 100.0,
        succeeded,
        failed,
        rate * 60.0,
        elapsed,
        f"{eta_s:.1f}" if eta_s is not None else "None",
        last_chunk,
    )


def _build_metric_for_chunk(
    *,
    client: StructuredLLMClient,
    model: str,
    chunk_index: int,
    total_chunks: int,
    chunk: Any,
    source: Path,
    source_rel: str,
    existing_context: str,
    ontology: Any,
    max_retries: int,
    num_ctx: int,
    num_batch: int | None = None,
    num_predict: int | None = None,
) -> tuple[BenchmarkMetrics, KnowledgeExtraction | None]:
    logger = get_logger(__name__)
    request_id = f"{chunk_index}/{total_chunks}:{chunk.id}"
    logger.info(
        "CHUNK START model=%s chunk=%d/%d chunk_id=%s text_chars=%d",
        model,
        chunk_index,
        total_chunks,
        chunk.id,
        len(chunk.text),
    )
    prompt = build_prompt(
        source_path=source,
        source_rel=source_rel,
        chunk=chunk,
        existing_context=existing_context,
        ontology=ontology,
    )
    try:
        runtime_options: dict[str, Any] = {"temperature": 0.0, "num_ctx": num_ctx, "seed": 42}
        if num_batch is not None:
            runtime_options["num_batch"] = num_batch
        if num_predict is not None:
            runtime_options["num_predict"] = num_predict
        call = client.generate_structured(
            model=model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            response_model=KnowledgeExtraction,
            options=runtime_options,
            max_retries=max_retries,
            think=False,
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("CHUNK EXCEPTION model=%s chunk=%s error=%s", model, chunk.id, exc)
        return BenchmarkMetrics(
            model=model,
            chunk_id=chunk.id,
            success=False,
            schema_valid=False,
            error=f"{type(exc).__name__}: {exc}",
        ), None

    if call.parsed is None:
        logger.error(
            "CHUNK FAILED model=%s chunk=%s attempts=%d error=%s raw_chars=%d",
            model,
            chunk.id,
            call.attempts,
            call.error,
            len(call.raw_text or ""),
        )
        return BenchmarkMetrics(
            model=model,
            chunk_id=chunk.id,
            success=False,
            schema_valid=False,
            recovered_json=call.recovered_json,
            attempts=call.attempts,
            ttft_ms=call.ttft_ms,
            wall_time_ms=call.wall_time_ms,
            total_duration_ms=call.total_duration_ms,
            load_duration_ms=call.load_duration_ms,
            prompt_eval_count=call.prompt_eval_count,
            prompt_eval_duration_ms=call.prompt_eval_duration_ms,
            eval_count=call.eval_count,
            eval_duration_ms=call.eval_duration_ms,
            tokens_per_sec=call.tokens_per_sec,
            error=call.error,
        ), None

    warn_raw = os.getenv("LLMWIKI_OUTPUT_TOKEN_WARN_THRESHOLD", "1024").strip()
    try:
        output_warn_threshold = max(0, int(warn_raw))
    except ValueError:
        output_warn_threshold = 1024
    if output_warn_threshold and call.eval_count is not None and call.eval_count >= output_warn_threshold:
        logger.warning(
            "OUTPUT TOKEN WARN model=%s chunk=%d/%d chunk_id=%s output_tokens=%d threshold=%d wall_ms=%.2f",
            model,
            chunk_index,
            total_chunks,
            chunk.id,
            call.eval_count,
            output_warn_threshold,
            call.wall_time_ms,
        )

    extraction = source_ref_defaults(call.parsed, chunk, source_rel)
    _, ontology_errors = materialize_relations(extraction, ontology)
    citation_ok = citation_preserved(extraction, source_rel, chunk.source_id, chunk.id)
    relation_count = len(extraction.relations)
    ontology_rate = 1.0 if relation_count == 0 else max(0.0, (relation_count - len(ontology_errors)) / relation_count)

    logger.info(
        "CHUNK LLM SUCCESS model=%s chunk=%d/%d chunk_id=%s attempts=%d prompt_chars=%d raw_chars=%d ttft_ms=%s wall_ms=%.2f total_ms=%s load_ms=%s prompt_tokens=%s prompt_eval_ms=%s output_tokens=%s eval_ms=%s tokens_per_sec=%s",
        model,
        chunk_index,
        total_chunks,
        chunk.id,
        call.attempts,
        len(prompt),
        len(call.raw_text or ""),
        f"{call.ttft_ms:.2f}" if call.ttft_ms is not None else "None",
        call.wall_time_ms,
        f"{call.total_duration_ms:.2f}" if call.total_duration_ms is not None else "None",
        f"{call.load_duration_ms:.2f}" if call.load_duration_ms is not None else "None",
        call.prompt_eval_count,
        f"{call.prompt_eval_duration_ms:.2f}" if call.prompt_eval_duration_ms is not None else "None",
        call.eval_count,
        f"{call.eval_duration_ms:.2f}" if call.eval_duration_ms is not None else "None",
        f"{call.tokens_per_sec:.2f}" if call.tokens_per_sec is not None else "None",
    )
    if ontology_errors:
        logger.warning(
            "ONTOLOGY WARN model=%s chunk=%s error_count=%d errors=%s",
            model,
            chunk.id,
            len(ontology_errors),
            ontology_errors[:10],
        )
    logger.info(
        "CHUNK END model=%s chunk=%d/%d chunk_id=%s ontology_errors=%d citation_preserved=%s",
        model,
        chunk_index,
        total_chunks,
        chunk.id,
        len(ontology_errors),
        citation_ok,
    )

    return BenchmarkMetrics(
        model=model,
        chunk_id=chunk.id,
        success=True,
        schema_valid=call.schema_valid,
        recovered_json=call.recovered_json,
        attempts=call.attempts,
        ttft_ms=call.ttft_ms,
        wall_time_ms=call.wall_time_ms,
        total_duration_ms=call.total_duration_ms,
        load_duration_ms=call.load_duration_ms,
        prompt_eval_count=call.prompt_eval_count,
        prompt_eval_duration_ms=call.prompt_eval_duration_ms,
        eval_count=call.eval_count,
        eval_duration_ms=call.eval_duration_ms,
        tokens_per_sec=call.tokens_per_sec,
        source_citation_preserved=citation_ok,
        ontology_compliance_rate=ontology_rate,
    ), extraction


def run_benchmark(
    *,
    source: Path,
    llmwiki_root: Path,
    ontology_path: Path,
    models: list[str],
    provider: str,
    ollama_host: str,
    api_key_env: str | None = None,
    openai_base_url: str | None = None,
    max_retries: int,
    num_ctx: int,
    num_batch: int | None = None,
    num_predict: int | None = None,
    concurrency: int = 1,
    min_llm_chars: int = 0,
    progress_every_chunks: int = 10,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    chunk_cache: Path | None = None,
    benchmark_wiki_base: Path | None = None,
) -> list[BenchmarkResult]:
    logger = get_logger(__name__)
    concurrency = max(1, concurrency)
    progress_every_chunks = max(1, progress_every_chunks)
    min_llm_chars = max(0, min_llm_chars)
    logger.info(
        "Benchmark start provider=%s source=%s models=%s host=%s num_ctx=%d num_batch=%s num_predict=%s max_retries=%d concurrency=%d min_llm_chars=%d resume=%s",
        provider,
        source,
        models,
        ollama_host if provider == "ollama" else (openai_base_url or "<sdk-default>"),
        num_ctx,
        num_batch if num_batch is not None else "default",
        num_predict if num_predict is not None else "default",
        max_retries,
        concurrency,
        min_llm_chars,
        resume,
    )

    ensure_layout(llmwiki_root)
    if not source.exists():
        raise FileNotFoundError(f"Benchmark source not found: {source}")
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    ontology = load_ontology(ontology_path)
    logger.info("Ontology loaded relation_count=%d", len(ontology.relations))
    if chunk_cache is not None:
        logger.info("Loading shared chunk cache path=%s", chunk_cache)
        all_chunks = load_chunk_cache(chunk_cache, source)
    else:
        logger.info("Starting Docling chunking source=%s", source)
        all_chunks = chunk_document(source)
    if min_llm_chars > 0:
        chunks = [chunk for chunk in all_chunks if len(chunk.text.strip()) >= min_llm_chars]
    else:
        chunks = all_chunks
    logger.info(
        "Chunk preparation complete total_chunks=%d selected_chunks=%d skipped_short=%d",
        len(all_chunks),
        len(chunks),
        len(all_chunks) - len(chunks),
    )
    if not chunks:
        raise RuntimeError("No chunks selected for benchmark")

    source_uuid = source_id(source)
    source_resolved = source.resolve()
    root_resolved = llmwiki_root.resolve()
    source_rel = str(source_resolved.relative_to(root_resolved)) if source_resolved.is_relative_to(root_resolved) else str(source_resolved)
    existing_context = retrieve_existing_context(llmwiki_root, source.stem.replace("_", " "), top_k=5)
    logger.debug("Existing wiki context chars=%d", len(existing_context))

    results: list[BenchmarkResult] = []
    for model in models:
        logger.info("MODEL START model=%s chunks=%d concurrency=%d", model, len(chunks), concurrency)
        model_started = time.monotonic()
        checkpoint_path: Path | None = None
        extraction_checkpoint_path: Path | None = None
        restored: dict[UUID, BenchmarkMetrics] = {}
        extraction_by_chunk: dict[UUID, KnowledgeExtraction] = {}
        if checkpoint_dir is not None:
            checkpoint_path = _checkpoint_path(checkpoint_dir, model)
            extraction_checkpoint_path = _extraction_path(checkpoint_dir, model)
            if resume:
                _archive_incompatible_checkpoint(
                    checkpoint_path,
                    extraction_checkpoint_path,
                    model=model,
                    source_uuid=source_uuid,
                    num_ctx=num_ctx,
                    concurrency=concurrency,
                    num_batch=num_batch,
                    num_predict=num_predict,
                )
                restored = _load_checkpoint(
                    checkpoint_path,
                    model=model,
                    source_uuid=source_uuid,
                    num_ctx=num_ctx,
                    concurrency=concurrency,
                    num_batch=num_batch,
                    num_predict=num_predict,
                )
                loaded_extractions = _load_extractions(extraction_checkpoint_path)
                chunk_by_id = {chunk.id: chunk for chunk in chunks}
                extraction_by_chunk = {
                    chunk_id: source_ref_defaults(extraction, chunk_by_id[chunk_id], source_rel)
                    for chunk_id, extraction in loaded_extractions.items()
                    if chunk_id in chunk_by_id
                }
                if loaded_extractions:
                    logger.info(
                        "EXTRACTION CHECKPOINT restored provider=%s model=%s loaded=%d normalized=%d",
                        provider,
                        model,
                        len(loaded_extractions),
                        len(extraction_by_chunk),
                    )
            _ensure_checkpoint_meta(
                checkpoint_path,
                model=model,
                source_uuid=source_uuid,
                num_ctx=num_ctx,
                concurrency=concurrency,
                num_batch=num_batch,
                num_predict=num_predict,
            )

        selected_chunk_ids = {chunk.id for chunk in chunks}
        # A successful metric without its structured extraction cannot rebuild the model Wiki.
        if benchmark_wiki_base is not None:
            restored = {
                chunk_id: metric for chunk_id, metric in restored.items()
                if (not metric.success) or chunk_id in extraction_by_chunk
            }
        metric_by_chunk: dict[UUID, BenchmarkMetrics] = {
            chunk_id: metric
            for chunk_id, metric in restored.items()
            if chunk_id in selected_chunk_ids
        }
        completed = len(metric_by_chunk)
        succeeded = sum(1 for metric in metric_by_chunk.values() if metric.success)
        failed = completed - succeeded
        if completed:
            _progress_log(
                model=model,
                completed=completed,
                total=len(chunks),
                succeeded=succeeded,
                failed=failed,
                started_monotonic=model_started,
                last_chunk="checkpoint",
            )

        pending = [(idx, chunk) for idx, chunk in enumerate(chunks, start=1) if chunk.id not in metric_by_chunk]
        thread_local = threading.local()

        def get_client() -> StructuredLLMClient:
            client = getattr(thread_local, "client", None)
            if client is None:
                if provider == "ollama":
                    client = OllamaClient(host=ollama_host)
                elif provider == "openai":
                    if not api_key_env:
                        raise RuntimeError("OpenAI provider requires api_key_env")
                    client = OpenAIClient(api_key_env=api_key_env, base_url=openai_base_url)
                else:
                    raise RuntimeError(f"Unsupported provider: {provider}")
                thread_local.client = client
            return client

        def work(item: tuple[int, Any]) -> tuple[int, Any, BenchmarkMetrics, KnowledgeExtraction | None]:
            idx, chunk = item
            metric, extraction = _build_metric_for_chunk(
                client=get_client(),
                model=model,
                chunk_index=idx,
                total_chunks=len(chunks),
                chunk=chunk,
                source=source,
                source_rel=source_rel,
                existing_context=existing_context,
                ontology=ontology,
                max_retries=max_retries,
                num_ctx=num_ctx,
                num_batch=num_batch,
                num_predict=num_predict,
            )
            return idx, chunk, metric, extraction

        def record(idx: int, chunk: Any, metric: BenchmarkMetrics, extraction: KnowledgeExtraction | None) -> None:
            nonlocal completed, succeeded, failed
            metric_by_chunk[chunk.id] = metric
            if checkpoint_path is not None:
                _append_checkpoint(checkpoint_path, metric)
            if extraction is not None:
                extraction_by_chunk[chunk.id] = extraction
                if extraction_checkpoint_path is not None:
                    _append_extraction(extraction_checkpoint_path, chunk.id, extraction)
            completed += 1
            if metric.success:
                succeeded += 1
            else:
                failed += 1
            if completed == len(chunks) or completed % progress_every_chunks == 0 or not metric.success:
                _progress_log(
                    model=model,
                    completed=completed,
                    total=len(chunks),
                    succeeded=succeeded,
                    failed=failed,
                    started_monotonic=model_started,
                    last_chunk=f"{idx}:{chunk.id}",
                )

        if concurrency == 1:
            for item in pending:
                idx, chunk, metric, extraction = work(item)
                record(idx, chunk, metric, extraction)
        else:
            logger.warning(
                "EXPERIMENTAL CONCURRENCY enabled model=%s workers=%d; monitor VRAM, Ollama queueing, and TTFT",
                model,
                concurrency,
            )
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="benchmark-worker") as executor:
                futures: list[Future[tuple[int, Any, BenchmarkMetrics, KnowledgeExtraction | None]]] = [executor.submit(work, item) for item in pending]
                for future in as_completed(futures):
                    idx, chunk, metric, extraction = future.result()
                    record(idx, chunk, metric, extraction)

        normalized_extractions, ontology_errors_by_chunk, raw_relation_count, retained_relation_count = _normalize_model_relations(
            extraction_by_chunk,
            ontology,
        )
        dropped_relation_count = raw_relation_count - retained_relation_count
        if dropped_relation_count:
            logger.warning(
                "RELATION ENDPOINT FILTER model=%s raw=%d retained=%d dropped_unresolved=%d",
                model,
                raw_relation_count,
                retained_relation_count,
                dropped_relation_count,
            )

        # Recompute relation/ontology compliance against the complete model-level
        # node registry.  Dropped unresolved endpoints still count as failures;
        # otherwise a model could hallucinate many relations, have them filtered
        # before Wiki materialization, and incorrectly receive a near-perfect
        # ontology score. Legitimate cross-chunk endpoints are resolved globally.
        for chunk in chunks:
            metric = metric_by_chunk.get(chunk.id)
            normalized = normalized_extractions.get(chunk.id)
            original = extraction_by_chunk.get(chunk.id)
            if metric is None or normalized is None or original is None or not metric.success:
                continue
            raw_count = len(original.relations)
            retained_count = len(normalized.relations)
            dropped_count = max(0, raw_count - retained_count)
            ontology_errors = ontology_errors_by_chunk.get(chunk.id, [])
            failures = dropped_count + len(ontology_errors)
            ontology_rate = 1.0 if raw_count == 0 else max(0.0, (raw_count - failures) / raw_count)
            metric_by_chunk[chunk.id] = metric.model_copy(update={"ontology_compliance_rate": ontology_rate})

        details = [metric_by_chunk[chunk.id] for chunk in chunks if chunk.id in metric_by_chunk]
        aggregate_result = aggregate(provider, model, str(source), source_uuid, details)
        model_elapsed_s = max(time.monotonic() - model_started, 0.001)
        aggregate_result = aggregate_result.model_copy(update={
            "effective_chunks_per_min": len(details) / (model_elapsed_s / 60.0),
            "raw_relation_count": raw_relation_count,
            "retained_relation_count": retained_relation_count,
            "dropped_unresolved_relation_count": dropped_relation_count,
        })
        if benchmark_wiki_base is not None:
            model_wiki_root = benchmark_wiki_base / _slug(f"{provider}-{model}")
            wiki_root, wiki_quality, wiki_ontology_errors = _materialize_model_wiki(
                model=model,
                model_root=model_wiki_root,
                source=source,
                source_rel=source_rel,
                source_uuid=source_uuid,
                chunks=chunks,
                extractions=normalized_extractions,
                ontology=ontology,
            )
            aggregate_result = aggregate_result.model_copy(update={
                "wiki_root": wiki_root,
                "wiki_quality": wiki_quality,
            })
            logger.info(
                "MODEL WIKI model=%s root=%s documents=%d relations=%d chunk_coverage=%.3f relation_integrity=%.3f materialization_errors=%d",
                model,
                wiki_root,
                wiki_quality.documents,
                wiki_quality.relations,
                wiki_quality.chunk_reference_coverage,
                wiki_quality.relation_integrity_rate,
                len(wiki_ontology_errors),
            )
        slowest = sorted(
            (m for m in details if m.total_duration_ms is not None),
            key=lambda m: m.total_duration_ms or 0.0,
            reverse=True,
        )[:5]
        if slowest:
            logger.info(
                "MODEL SLOWEST model=%s top=%s",
                model,
                [(str(m.chunk_id), round(m.total_duration_ms or 0.0, 2), m.eval_count, m.prompt_eval_count) for m in slowest],
            )
        logger.info(
            "MODEL END model=%s success=%d/%d schema_pass=%.3f citation=%.3f ontology=%.3f mean_tps=%s chunks_min=%s relations=%d/%d dropped=%d elapsed_s=%.1f",
            model,
            aggregate_result.successful_chunks,
            aggregate_result.chunks,
            aggregate_result.schema_pass_rate,
            aggregate_result.source_citation_rate,
            aggregate_result.ontology_compliance_rate,
            f"{aggregate_result.mean_tokens_per_sec:.2f}" if aggregate_result.mean_tokens_per_sec is not None else "None",
            f"{aggregate_result.effective_chunks_per_min:.2f}" if aggregate_result.effective_chunks_per_min is not None else "None",
            aggregate_result.retained_relation_count,
            aggregate_result.raw_relation_count,
            aggregate_result.dropped_unresolved_relation_count,
            time.monotonic() - model_started,
        )
        results.append(aggregate_result)
    return results


def main() -> None:
    log_path = os.getenv("LLMWIKI_LOG_FILE")
    configure_logging(
        component=__name__,
        log_file=Path(log_path) if log_path else APP_ROOT / "outputs" / "logs" / "benchmark.log",
    )
    logger = get_logger(__name__)
    log_environment(logger)

    parser = argparse.ArgumentParser(description="Benchmark Ollama/OpenAI models on identical Docling chunks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--llmwiki", default=str(DEFAULT_LLMWIKI_ROOT))
    parser.add_argument("--ontology", default=str(APP_ROOT.parent / "config" / "ontology.yaml"))
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:3003/api/proxy"),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name containing the provider API key")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL; omit for official OpenAI API")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-batch", type=int, default=None, help="Optional Ollama prompt batch size")
    parser.add_argument("--num-predict", type=int, default=None, help="Optional Ollama output-token safety cap")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("LLMWIKI_BENCHMARK_CONCURRENCY", "1")))
    parser.add_argument("--min-llm-chars", type=int, default=int(os.getenv("LLMWIKI_MIN_LLM_CHARS", "0")))
    parser.add_argument("--progress-every-chunks", type=int, default=int(os.getenv("LLMWIKI_PROGRESS_EVERY_CHUNKS", "10")))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--chunk-cache", default=None, help="Reuse a precomputed Docling chunk JSONL cache")
    parser.add_argument("--benchmark-wiki-base", default=None, help="Base directory for isolated per-model benchmark Wikis")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    llmwiki_root = Path(args.llmwiki).expanduser().resolve()
    ontology_path = Path(args.ontology).expanduser().resolve()
    output = Path(args.report).expanduser().resolve() if args.report else APP_ROOT / "outputs" / "benchmark" / f"benchmark-{source.stem}.md"
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else output.parent / f"{output.stem}.checkpoints"
    chunk_cache = Path(args.chunk_cache).expanduser().resolve() if args.chunk_cache else None
    benchmark_wiki_base = Path(args.benchmark_wiki_base).expanduser().resolve() if args.benchmark_wiki_base else None

    logger.info(
        "CLI resolved source=%s llmwiki=%s ontology=%s report=%s checkpoint_dir=%s concurrency=%d num_ctx=%d num_batch=%s num_predict=%s min_llm_chars=%d",
        source,
        llmwiki_root,
        ontology_path,
        output,
        checkpoint_dir,
        args.concurrency,
        args.num_ctx,
        args.num_batch if args.num_batch is not None else "default",
        args.num_predict if args.num_predict is not None else "default",
        args.min_llm_chars,
    )
    try:
        results = run_benchmark(
            source=source,
            llmwiki_root=llmwiki_root,
            ontology_path=ontology_path,
            models=list(args.models),
            provider=args.provider,
            ollama_host=args.ollama_host,
            api_key_env=args.api_key_env,
            openai_base_url=args.openai_base_url,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
            num_batch=args.num_batch,
            num_predict=args.num_predict,
            concurrency=args.concurrency,
            min_llm_chars=args.min_llm_chars,
            progress_every_chunks=args.progress_every_chunks,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
            chunk_cache=chunk_cache,
            benchmark_wiki_base=benchmark_wiki_base,
        )
        write_report(output_path=output, source=source, results=results)
        logger.info("Benchmark report written to %s", output)
    except Exception:
        logger.exception("FATAL benchmark failure")
        raise

    print(json.dumps([result.model_dump(mode="json", exclude={"details"}) for result in results], ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
