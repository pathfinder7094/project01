from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from queue import Queue, Empty
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from app.utils.paths import DEFAULT_LLMWIKI_ROOT, PROJECT_ROOT, ensure_layout
from app.utils.logging import configure_logging, get_logger, log_environment

DEFAULT_OPENAI_MODEL_LIST = PROJECT_ROOT / "config" / "models.txt"
DEFAULT_OLLAMA_MODEL_LIST = PROJECT_ROOT / "config" / "ollama-models.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "outputs" / "benchmark" / "batch-runs"
DEFAULT_OBSIDIAN_DIR = DEFAULT_LLMWIKI_ROOT / "wiki" / "benchmarks"
DEFAULT_LOOP_CONFIG = PROJECT_ROOT / "config" / "loop.env"

UUID_NAMESPACE_SOURCE = UUID("9d4c1d2b-0c37-5c65-9b9a-111111111111")


def _source_uuid(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(uuid5(UUID_NAMESPACE_SOURCE, digest))


def _read_chunk_cache_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                item = json.loads(raw)
                return item if item.get("_type") == "meta" else None
    except Exception:
        return None
    return None


def _env_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_loop_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid loop.env line {line_no}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid loop.env line {line_no}: empty key")
        values[key] = value
    return values


def env_int(values: dict[str, str], key: str, default: int) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "model"


def read_model_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Model list not found: {path}")

    models: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        model = raw.strip()
        if not model or model.startswith("#"):
            continue
        if model in seen:
            raise ValueError(f"Duplicate model in {path}:{line_no}: {model}")
        seen.add(model)
        models.append(model)

    if not models:
        raise ValueError(f"No models found in {path}")
    return models



@dataclass(frozen=True, slots=True)
class ModelTarget:
    provider: str
    model: str
    api_key_env: str | None = None
    # Optional per-model runtime tuning.  None means use the batch default.
    concurrency: int | None = None
    num_ctx: int | None = None
    num_batch: int | None = None
    num_predict: int | None = None


def read_openai_model_list(path: Path) -> list[ModelTarget]:
    """Read `model:API_KEY_ENV` entries. Blank/comment-only files are allowed."""
    if not path.exists():
        return []
    targets: list[ModelTarget] = []
    seen: set[tuple[str, str]] = set()
    env_name_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Invalid OpenAI model mapping {path}:{line_no}: expected MODEL:API_KEY_ENV"
            )
        model, api_key_env = line.rsplit(":", 1)
        model = model.strip()
        api_key_env = api_key_env.strip()
        if not model or not env_name_re.fullmatch(api_key_env):
            raise ValueError(
                f"Invalid OpenAI model mapping {path}:{line_no}: {line!r}"
            )
        key = (model, api_key_env)
        if key in seen:
            raise ValueError(f"Duplicate OpenAI model mapping in {path}:{line_no}: {line}")
        seen.add(key)
        targets.append(ModelTarget(provider="openai", model=model, api_key_env=api_key_env))
    return targets


def read_ollama_targets(path: Path) -> list[ModelTarget]:
    """Read Ollama model tags with optional per-model tuning.

    Backward compatible format::

        qwen3:4b-instruct-2507-q4_K_M

    Optional tuning uses ``|key=value`` fields so Ollama's ``model:tag`` syntax
    remains unambiguous::

        gemma3:4b-it-q4_K_M | concurrency=2 | num_ctx=8192

    Supported keys: concurrency, num_ctx, num_batch, num_predict.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model list not found: {path}")

    allowed = {"concurrency", "num_ctx", "num_batch", "num_predict"}
    targets: list[ModelTarget] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        model = parts[0]
        if not model:
            raise ValueError(f"Invalid Ollama model entry {path}:{line_no}: empty model")
        if model in seen:
            raise ValueError(f"Duplicate model in {path}:{line_no}: {model}")
        seen.add(model)

        tuning: dict[str, int] = {}
        for field in parts[1:]:
            if not field:
                continue
            if "=" not in field:
                raise ValueError(
                    f"Invalid Ollama tuning {path}:{line_no}: expected key=value, got {field!r}"
                )
            key, raw_value = (value.strip() for value in field.split("=", 1))
            if key not in allowed:
                raise ValueError(
                    f"Invalid Ollama tuning key {path}:{line_no}: {key!r}; allowed={sorted(allowed)}"
                )
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Ollama tuning value {path}:{line_no}: {key}={raw_value!r}"
                ) from exc
            if value < 1:
                raise ValueError(f"{key} must be >= 1 in {path}:{line_no}")
            tuning[key] = value

        targets.append(ModelTarget(provider="ollama", model=model, **tuning))

    if not targets:
        raise ValueError(f"No models found in {path}")
    return targets


def _as_target(target: ModelTarget | str) -> ModelTarget:
    if isinstance(target, ModelTarget):
        return target
    return ModelTarget(provider="ollama", model=str(target))


def target_slug(target: ModelTarget | str) -> str:
    target = _as_target(target)
    return slugify(f"{target.provider}-{target.model}")


def _checkpoint_dir_candidates(run_dir: Path, target: ModelTarget | str) -> list[Path]:
    target = _as_target(target)
    patterns = [f"*-{target_slug(target)}.checkpoints"]
    if target.provider == "ollama":
        # Backward compatibility with pre-provider runs such as 01-qwen3-....checkpoints.
        patterns.append(f"*-{slugify(target.model)}.checkpoints")
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def resolve_checkpoint_dir(run_dir: Path, sequence: int, target: ModelTarget | str) -> Path:
    target = _as_target(target)
    for candidate in _checkpoint_dir_candidates(run_dir, target):
        metric = candidate / f"{slugify(target.model)}.jsonl"
        extraction = candidate / f"{slugify(target.model)}.extractions.jsonl"
        if metric.exists() or extraction.exists():
            return candidate
    return run_dir / f"{sequence:02d}-{target_slug(target)}.checkpoints"


def _target_success_result(run_dir: Path, target: ModelTarget | str) -> Path | None:
    target = _as_target(target)
    patterns = [f"*-{target_slug(target)}.json"]
    if target.provider == "ollama":
        patterns.append(f"*-{slugify(target.model)}.json")
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern), reverse=True):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            provider = item.get("provider", "ollama")
            if item.get("model") == target.model and provider == target.provider and item.get("status") == "success":
                if path.with_suffix(".md").exists():
                    return path
    return None


def _run_is_complete(run_dir: Path, targets: list[ModelTarget | str]) -> bool:
    return bool(targets) and all(_target_success_result(run_dir, target) is not None for target in targets)


def _run_has_checkpoint(run_dir: Path, targets: list[ModelTarget | str]) -> bool:
    for target_value in targets:
        target = _as_target(target_value)
        for checkpoint_dir in _checkpoint_dir_candidates(run_dir, target):
            metric = checkpoint_dir / f"{slugify(target.model)}.jsonl"
            extraction = checkpoint_dir / f"{slugify(target.model)}.extractions.jsonl"
            if metric.exists() and metric.stat().st_size > 0 and extraction.exists() and extraction.stat().st_size > 0:
                return True
    return False


def find_latest_resumable_run(output_dir: Path, source: Path, targets: list[ModelTarget | str]) -> Path | None:
    if not output_dir.exists():
        return None
    expected_source_id = _source_uuid(source)
    for run_dir in sorted((p for p in output_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        cache_meta = _read_chunk_cache_meta(run_dir / "chunks.jsonl")
        if not cache_meta or str(cache_meta.get("source_id")) != expected_source_id:
            continue
        if _run_is_complete(run_dir, targets):
            continue
        if _run_has_checkpoint(run_dir, targets):
            return run_dir
    return None


def _write_run_manifest(
    path: Path,
    *,
    run_id: str,
    source: Path,
    targets: list[ModelTarget],
    num_ctx: int,
    status: str,
    resumed: bool,
) -> None:
    payload = {
        "run_id": run_id,
        "source": str(source),
        "source_id": _source_uuid(source),
        "targets": [
            {
                "provider": target.provider,
                "model": target.model,
                "api_key_env": target.api_key_env,
                "concurrency": target.concurrency,
                "num_ctx": target.num_ctx,
                "num_batch": target.num_batch,
                "num_predict": target.num_predict,
            }
            for target in targets
        ],
        "num_ctx": num_ctx,
        "status": status,
        "resumed": resumed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def parse_summary_json(stdout: str) -> dict[str, Any] | None:
    lines = stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "["), None)
    if start is None:
        return None

    candidate = "\n".join(lines[start:])
    if "\nReport:" in candidate:
        candidate = candidate.split("\nReport:", 1)[0]

    try:
        items = json.loads(candidate.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return None


def run_single_model(
    *,
    project_root: Path,
    source: Path,
    llmwiki: Path,
    ontology: Path,
    model: str,
    provider: str,
    ollama_host: str,
    api_key_env: str | None = None,
    openai_base_url: str | None = None,
    max_retries: int,
    num_ctx: int,
    num_batch: int | None = None,
    num_predict: int | None = None,
    report_path: Path,
    loop_values: dict[str, str],
    force_keep_alive_zero: bool = False,
    concurrency: int = 1,
    min_llm_chars: int = 0,
    progress_every_chunks: int = 10,
    subprocess_heartbeat_seconds: int = 15,
    chunk_cache: Path | None = None,
    benchmark_wiki_base: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[int, str, str, dict[str, Any] | None]:
    command = [
        sys.executable,
        "-m",
        "app.eval.benchmark",
        "--input",
        str(source),
        "--llmwiki",
        str(llmwiki),
        "--ontology",
        str(ontology),
        "--ollama-host",
        ollama_host,
        "--provider",
        provider,
        "--models",
        model,
        "--max-retries",
        str(max_retries),
        "--num-ctx",
        str(num_ctx),
        "--concurrency",
        str(max(1, concurrency)),
        "--min-llm-chars",
        str(max(0, min_llm_chars)),
        "--progress-every-chunks",
        str(max(1, progress_every_chunks)),
        "--report",
        str(report_path),
    ]
    if provider == "ollama" and num_batch is not None:
        command.extend(["--num-batch", str(num_batch)])
    if provider == "ollama" and num_predict is not None:
        command.extend(["--num-predict", str(num_predict)])
    if chunk_cache is not None:
        command.extend(["--chunk-cache", str(chunk_cache)])
    if benchmark_wiki_base is not None:
        command.extend(["--benchmark-wiki-base", str(benchmark_wiki_base)])
    if api_key_env is not None:
        command.extend(["--api-key-env", api_key_env])
    if openai_base_url:
        command.extend(["--openai-base-url", openai_base_url])
    if checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", str(checkpoint_dir), "--resume"])

    logger = get_logger(__name__)
    env = dict(os.environ)

    for key, value in loop_values.items():
       if value is not None:
          env[key] = str(value)

    env["PYTHONUNBUFFERED"] = "1"
    child_log_file = report_path.with_suffix(".log")
    env["LLMWIKI_LOG_FILE"] = str(child_log_file)

    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LLMWIKI_FORCE_KEEP_ALIVE_ZERO"] = "1" if force_keep_alive_zero else "0"
    env["LLMWIKI_STREAM_HEARTBEAT_SECONDS"] = loop_values.get("STREAM_HEARTBEAT_SECONDS", "10")
    env["LLMWIKI_STALL_WARN_SECONDS"] = loop_values.get("STALL_WARN_SECONDS", "30")

    logger.info("Launching benchmark subprocess provider=%s model=%s", provider, model)
    logger.debug("Command=%s", command)
    logger.debug(
        "Child environment provider=%s OLLAMA_HOST=%s OLLAMA_API_KEY=%s OPENAI_KEY_ENV=%s OPENAI_KEY_PRESENT=%s LLMWIKI_LOG_FILE=%s",
        provider,
        env.get("OLLAMA_HOST"),
        "SET" if env.get("OLLAMA_API_KEY") else "NOT_SET",
        api_key_env or "-",
        "SET" if (api_key_env and env.get(api_key_env)) else "N/A",
        child_log_file or "<not set>",
    )

    completed = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    queue: Queue[tuple[str, str | None]] = Queue()

    def _reader(stream, label: str) -> None:
        try:
            assert stream is not None
            for line in iter(stream.readline, ""):
                queue.put((label, line))
        finally:
            queue.put((label, None))

    stdout_thread = threading.Thread(target=_reader, args=(completed.stdout, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=_reader, args=(completed.stderr, "stderr"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    done_streams = 0
    last_output_at = time.monotonic()
    last_heartbeat_at = last_output_at
    while done_streams < 2:
        try:
            label, line = queue.get(timeout=0.25)
        except Empty:
            now = time.monotonic()
            if completed.poll() is not None and done_streams < 2:
                # Give reader threads a chance to flush their final lines.
                continue
            if subprocess_heartbeat_seconds > 0 and now - last_heartbeat_at >= subprocess_heartbeat_seconds:
                silent_for = now - last_output_at
                logger.info(
                    "CHILD HEARTBEAT model=%s pid=%s alive=%s silent_for_s=%.1f",
                    model,
                    completed.pid,
                    completed.poll() is None,
                    silent_for,
                )
                print(
                    f"[{model}] HEARTBEAT pid={completed.pid} alive={completed.poll() is None} silent_for={silent_for:.1f}s",
                    flush=True,
                )
                last_heartbeat_at = now
            continue

        if line is None:
            done_streams += 1
            continue

        last_output_at = time.monotonic()
        last_heartbeat_at = last_output_at
        if label == "stdout":
            stdout_lines.append(line)
            print(f"[{model}] {line}", end="", flush=True)
            logger.debug("CHILD STDOUT model=%s %s", model, line.rstrip())
        else:
            stderr_lines.append(line)
            print(f"[{model} STDERR] {line}", end="", file=sys.stderr, flush=True)
            logger.warning("CHILD STDERR model=%s %s", model, line.rstrip())

    return_code = completed.wait()
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    summary = parse_summary_json(stdout)
    if return_code != 0:
        logger.error("E_BENCHMARK_SUBPROCESS: model=%s return_code=%d", model, return_code)
    logger.info(
        "Benchmark subprocess finished model=%s return_code=%d stdout_chars=%d stderr_chars=%d summary=%s",
        model,
        return_code,
        len(stdout),
        len(stderr),
        summary is not None,
    )
    return return_code, stdout, stderr, summary


def prepare_shared_chunk_cache(*, project_root: Path, source: Path, cache_path: Path) -> None:
    logger = get_logger(__name__)
    command = [
        sys.executable, "-m", "app.pipeline.chunk_cache",
        "--input", str(source),
        "--output", str(cache_path),
    ]
    logger.info("Preparing shared Docling chunk cache source=%s cache=%s", source, cache_path)
    completed = subprocess.run(command, cwd=project_root, text=True, capture_output=True, check=False)
    if completed.stdout.strip():
        logger.info("Chunk cache output: %s", completed.stdout.strip()[-2000:])
    if completed.returncode != 0:
        logger.error("E_CHUNK_CACHE: %s", completed.stderr.strip())
        raise RuntimeError(f"Chunk cache preparation failed return_code={completed.returncode}: {completed.stderr[-2000:]}")
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        raise RuntimeError(f"Chunk cache was not created: {cache_path}")


def stop_model(model: str) -> tuple[bool, str]:
    logger = get_logger(__name__)
    logger.info("Executing ollama stop model=%s", model)
    completed = subprocess.run(["ollama", "stop", model], text=True, capture_output=True, check=False)
    message = (completed.stdout or completed.stderr).strip()
    if completed.returncode == 0:
        logger.info("ollama stop succeeded model=%s message=%s", model, message)
    else:
        logger.error("ollama stop failed model=%s return_code=%d message=%s", model, completed.returncode, message)
    return completed.returncode == 0, message


def ollama_loaded_models() -> list[str] | None:
    logger = get_logger(__name__)
    completed = subprocess.run(["ollama", "ps"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        logger.error("ollama ps failed return_code=%d stderr=%s", completed.returncode, (completed.stderr or "").strip())
        return None
    lines = completed.stdout.splitlines()
    if len(lines) <= 1:
        return []
    models: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts:
            models.append(parts[0])
    logger.debug("ollama ps loaded_models=%s", models)
    return models


def wait_until_model_unloaded(model: str, timeout_seconds: int, poll_seconds: int) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        loaded = ollama_loaded_models()
        if loaded is not None and not any(name == model or name.startswith(model + "@") for name in loaded):
            return True, "model unloaded"
        time.sleep(poll_seconds)
    return False, f"model still present after {timeout_seconds}s"


def gpu_memory_used_mb() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(float(line))
            except ValueError:
                pass
    return max(values) if values else None


def wait_for_gpu_free(threshold_mb: int, timeout_seconds: int, poll_seconds: int) -> tuple[bool, str]:
    if threshold_mb < 0:
        return False, "invalid threshold"
    deadline = time.monotonic() + timeout_seconds
    last: float | None = None
    while time.monotonic() < deadline:
        last = gpu_memory_used_mb()
        if last is None:
            return True, "nvidia-smi unavailable; skipped GPU memory verification"
        if last <= threshold_mb:
            return True, f"GPU memory used {last:.0f} MB <= {threshold_mb} MB"
        time.sleep(poll_seconds)
    return False, (
        f"E_GPU_CLEANUP_TIMEOUT: GPU memory still {last:.0f} MB > {threshold_mb} MB after {timeout_seconds}s"
        if last is not None
        else "E_GPU_MEMORY_UNAVAILABLE: GPU memory unavailable"
    )


def verify_model_output(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"E_OUTPUT_MISSING: output missing: {path}"
    if path.stat().st_size == 0:
        return False, f"E_OUTPUT_EMPTY: output empty: {path}"
    return True, "output verified"


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def _num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def write_obsidian_model_note(
    *,
    output_path: Path,
    result: dict[str, Any],
    source: Path,
    run_id: str,
    sequence: int,
) -> None:
    provider = result.get("provider", "ollama")
    model = result["model"]
    summary = result.get("result") or {}
    status = result.get("status", "failed")
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "---",
        f"id: {slugify(model)}-{run_id}",
        "type: benchmark",
        f"title: {model} benchmark - {source.stem}",
        f"provider: {provider}",
        f"model: {model}",
        f"run_id: {run_id}",
        f"sequence: {sequence}",
        f"status: {status}",
        f"source: {source}",
        f"generated_at: {generated}",
        "---",
        "",
        f"# {provider} / {model}",
        "",
        f"- Batch: [[benchmarks/{run_id}|{run_id}]]",
        f"- Source: `{source}`",
        f"- Provider: `{provider}`",
        f"- Status: `{status}`",
        f"- Wiki sandbox: `{summary.get('wiki_root', '-') if summary else '-'}`",
        f"- Open generated Wiki: [[benchmark-runs/{run_id}/{slugify(f'{provider}-{model}')}/wiki/index|{provider}/{model} Wiki]]",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Chunks | {summary.get('chunks', '-') if summary else '-'} |",
        f"| Schema pass rate | {_pct(summary.get('schema_pass_rate') if summary else None)} |",
        f"| JSON recovery rate | {_pct(summary.get('json_recovery_rate') if summary else None)} |",
        f"| Source citation rate | {_pct(summary.get('source_citation_rate') if summary else None)} |",
        f"| Ontology compliance | {_pct(summary.get('ontology_compliance_rate') if summary else None)} |",
        f"| Wiki documents | {(summary.get('wiki_quality') or {}).get('documents', '-') if summary else '-'} |",
        f"| Wiki relations | {(summary.get('wiki_quality') or {}).get('relations', '-') if summary else '-'} |",
        f"| Wiki chunk coverage | {_pct((summary.get('wiki_quality') or {}).get('chunk_reference_coverage') if summary else None)} |",
        f"| Wiki relation integrity | {_pct((summary.get('wiki_quality') or {}).get('relation_integrity_rate') if summary else None)} |",
        f"| Mean TTFT (ms) | {_num(summary.get('mean_ttft_ms') if summary else None)} |",
        f"| Mean tokens/sec | {_num(summary.get('mean_tokens_per_sec') if summary else None)} |",
        f"| Mean prompt eval (ms) | {_num(summary.get('mean_prompt_eval_ms') if summary else None)} |",
        f"| Mean output tokens | {_num(summary.get('mean_output_tokens') if summary else None)} |",
        f"| P95 output tokens | {_num(summary.get('p95_output_tokens') if summary else None)} |",
        f"| Max output tokens | {summary.get('max_output_tokens', '-') if summary else '-'} |",
        f"| Effective chunks/min | {_num(summary.get('effective_chunks_per_min') if summary else None)} |",
        f"| Relations emitted | {summary.get('raw_relation_count', '-') if summary else '-'} |",
        f"| Relations retained | {summary.get('retained_relation_count', '-') if summary else '-'} |",
        f"| Relations dropped (unresolved endpoint) | {summary.get('dropped_unresolved_relation_count', '-') if summary else '-'} |",
        f"| Mean total duration (ms) | {_num(summary.get('mean_total_duration_ms') if summary else None)} |",
        f"| P95 total duration (ms) | {_num(summary.get('p95_total_duration_ms') if summary else None)} |",
        "",
    ]
    if result.get("error"):
        lines.extend(["## Error", "", f"```text\n{result['error'][:4000]}\n```", ""])
    lines.extend(["## Raw run", "", f"- JSON: `{result.get('report_path', '-')}`", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_obsidian_batch_note(
    *,
    output_path: Path,
    run_id: str,
    source: Path,
    model_list_path: Path,
    ollama_model_list_path: Path,
    results: list[dict[str, Any]],
) -> None:
    lines = [
        "---",
        f"id: batch-{run_id}",
        "type: benchmark-batch",
        f"title: Batch benchmark - {source.stem} - {run_id}",
        f"run_id: {run_id}",
        f"source: {source}",
        f"openai_model_list: {model_list_path}",
        f"ollama_model_list: {ollama_model_list_path}",
        f"generated_at: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "---",
        "",
        f"# Batch benchmark — {source.stem}",
        "",
        f"OpenAI model list: `{model_list_path}`",
        f"Ollama model list: `{ollama_model_list_path}`",
        f"Source: `{source}`",
        "",
        "## Summary",
        "",
        "| # | Provider | Model | Status | Chunks | Schema | Citation | Ontology | Wiki docs | Wiki rels | Dropped rels | TTFT ms | tok/s | Mean out tok | chunks/min | Total ms |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(results, start=1):
        summary = item.get("result") or {}
        lines.append(
            f"| {idx} | {item.get('provider', 'ollama')} | {item['model']} | {item.get('status', 'failed').upper()} | "
            f"{summary.get('chunks', '-')} | {_pct(summary.get('schema_pass_rate'))} | "
            f"{_pct(summary.get('source_citation_rate'))} | {_pct(summary.get('ontology_compliance_rate'))} | "
            f"{(summary.get('wiki_quality') or {}).get('documents', '-')} | {(summary.get('wiki_quality') or {}).get('relations', '-')} | "
            f"{summary.get('dropped_unresolved_relation_count', '-')} | "
            f"{_num(summary.get('mean_ttft_ms'))} | {_num(summary.get('mean_tokens_per_sec'))} | "
            f"{_num(summary.get('mean_output_tokens'))} | {_num(summary.get('effective_chunks_per_min'))} | "
            f"{_num(summary.get('mean_total_duration_ms'))} |"
        )

    lines.extend(["", "## Model notes", ""])
    for idx, item in enumerate(results, start=1):
        provider = item.get("provider", "ollama")
        model = item["model"]
        model_note = f"{idx:02d}-{slugify(provider + '-' + model)}.md"
        lines.append(f"- [[benchmarks/{run_id}/{model_note[:-3]}|{provider}/{model}]]")

    lines.extend(["", "## Execution policy", "", "OpenAI models run first, followed by Ollama models.", "All models reuse the same Docling chunk cache.", "Only Ollama models are unloaded and checked for VRAM cleanup.", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_overview_note(*, overview_path: Path, run_id: str, source: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Runs",
        "",
        f"Latest run: [[benchmarks/{run_id}|{run_id}]]",
        f"Latest source: `{source}`",
        "",
        "## Models in latest run",
        "",
    ]
    for idx, item in enumerate(results, start=1):
        provider = item.get("provider", "ollama")
        model = item["model"]
        note_slug = slugify(provider + "-" + model)
        lines.append(f"{idx}. [[benchmarks/{run_id}/{idx:02d}-{note_slug}|{provider}/{model}]] — `{item.get('status', 'failed')}`")
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential OpenAI then Ollama WikiLLM benchmark runner")
    parser.add_argument("--input", required=True, help="Source document")
    parser.add_argument("--llmwiki", default=str(DEFAULT_LLMWIKI_ROOT))
    parser.add_argument("--ontology", default=str(PROJECT_ROOT / "config" / "ontology.yaml"))
    parser.add_argument("--model-list", default=str(DEFAULT_OPENAI_MODEL_LIST), help="OpenAI mappings: MODEL:API_KEY_ENV")
    parser.add_argument("--ollama-model-list", default=str(DEFAULT_OLLAMA_MODEL_LIST), help="Ollama model tags, one per line")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL; omit for official API")
    parser.add_argument(
    "--ollama-host",
    default=None,
)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=None, help="Override BENCHMARK_CONCURRENCY")
    parser.add_argument("--min-llm-chars", type=int, default=None, help="Override MIN_LLM_CHARS")
    parser.add_argument("--progress-every-chunks", type=int, default=None, help="Override PROGRESS_EVERY_CHUNKS")
    parser.add_argument("--subprocess-heartbeat-seconds", type=int, default=None, help="Override SUBPROCESS_HEARTBEAT_SECONDS")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--obsidian-dir", default=None, help="Defaults to <llmwiki>/wiki/benchmarks")
    parser.add_argument("--batch-report", default=None, help="Optional non-Obsidian combined report path")
    parser.add_argument("--loop-config", default=str(DEFAULT_LOOP_CONFIG), help="KEY=VALUE loop control file")
    parser.add_argument("--stop-after-each", action=argparse.BooleanOptionalAction, default=None, help="Override STOP_MODEL_AFTER_EACH")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Reuse an existing batch run directory/checkpoints instead of creating a new run",
    )
    parser.add_argument(
        "--auto-resume-failed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically reuse the latest incomplete run for the same source; provider/model checkpoint names are matched independent of sequence",
    )
    args = parser.parse_args()

    loop_config_path = Path(args.loop_config).expanduser().resolve()
    loop_values = load_loop_env(loop_config_path)
    ollama_host = (
    args.ollama_host
    or loop_values.get("OLLAMA_HOST")
    or os.getenv("OLLAMA_HOST")
    or "http://localhost:3003/api/proxy"
    )
    task_wait_seconds = env_int(loop_values, "TASK_LOOP_WAIT_SECONDS", 300)
    cleanup_wait_seconds = env_int(loop_values, "VRAM_CLEANUP_WAIT_SECONDS", 10)
    poll_seconds = env_int(loop_values, "VRAM_POLL_INTERVAL_SECONDS", 2) or 1
    cleanup_timeout_seconds = env_int(loop_values, "VRAM_CLEANUP_TIMEOUT_SECONDS", 60)
    stop_after_each = args.stop_after_each if args.stop_after_each is not None else _env_bool(loop_values.get("STOP_MODEL_AFTER_EACH"), True)
    verify_output = _env_bool(loop_values.get("VERIFY_MODEL_OUTPUT"), True)
    verify_gpu = _env_bool(loop_values.get("VERIFY_GPU_MEMORY"), True)
    gpu_threshold_mb = env_int(loop_values, "GPU_FREE_THRESHOLD_MB", 1200)
    force_keep_alive_zero = _env_bool(loop_values.get("FORCE_KEEP_ALIVE_ZERO"), False)
    benchmark_concurrency = args.concurrency if args.concurrency is not None else (env_int(loop_values, "BENCHMARK_CONCURRENCY", 1) or 1)
    min_llm_chars = args.min_llm_chars if args.min_llm_chars is not None else env_int(loop_values, "MIN_LLM_CHARS", 0)
    progress_every_chunks = args.progress_every_chunks if args.progress_every_chunks is not None else (env_int(loop_values, "PROGRESS_EVERY_CHUNKS", 10) or 1)
    subprocess_heartbeat_seconds = args.subprocess_heartbeat_seconds if args.subprocess_heartbeat_seconds is not None else env_int(loop_values, "SUBPROCESS_HEARTBEAT_SECONDS", 15)
    auto_resume_failed = (
        args.auto_resume_failed
        if args.auto_resume_failed is not None
        else _env_bool(loop_values.get("AUTO_RESUME_FAILED_RUN"), True)
    )
    if benchmark_concurrency < 1:
        raise ValueError("BENCHMARK_CONCURRENCY/--concurrency must be >= 1")
    if min_llm_chars < 0:
        raise ValueError("MIN_LLM_CHARS/--min-llm-chars must be >= 0")
    if progress_every_chunks < 1:
        raise ValueError("PROGRESS_EVERY_CHUNKS/--progress-every-chunks must be >= 1")
    if subprocess_heartbeat_seconds < 0:
        raise ValueError("SUBPROCESS_HEARTBEAT_SECONDS/--subprocess-heartbeat-seconds must be >= 0")

    project_root = Path(__file__).resolve().parents[2]
    source = Path(args.input).expanduser().resolve()
    llmwiki = Path(args.llmwiki).expanduser().resolve()
    ontology = Path(args.ontology).expanduser().resolve()
    model_list = Path(args.model_list).expanduser().resolve()
    ollama_model_list = Path(args.ollama_model_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    obsidian_dir = Path(args.obsidian_dir).expanduser().resolve() if args.obsidian_dir else llmwiki / "wiki" / "benchmarks"

    if not source.exists():
        raise SystemExit(f"Input document not found: {source}")
    if not ontology.exists():
        raise SystemExit(f"Ontology file not found: {ontology}")

    ensure_layout(llmwiki)
    openai_targets = read_openai_model_list(model_list)
    ollama_targets = read_ollama_targets(ollama_model_list)
    targets = [*openai_targets, *ollama_targets]
    if not targets:
        raise RuntimeError("No benchmark models configured in config/models.txt or config/ollama-models.txt")

    # Preflight provider credentials before the one-time Docling/OCR step.
    for target in openai_targets:
        key_value = (loop_values.get(target.api_key_env or "") or os.getenv(target.api_key_env or "") or "").strip()
        if not key_value:
            raise RuntimeError(
                f"E_CONFIG_OPENAI_API_KEY_MISSING: {target.model} requires {target.api_key_env} "
                "in config/loop.env (or process environment). Benchmark stopped before Docling/OCR."
            )
    if ollama_targets:
        ollama_api_key = (loop_values.get("OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY") or "").strip()
        if not ollama_api_key:
            raise RuntimeError(
                "E_CONFIG_API_KEY_MISSING: OLLAMA_API_KEY is missing from config/loop.env (or process environment). "
                "Benchmark stopped before Docling/OCR."
            )

    resumed_run = False
    if args.resume_run_id:
        run_dir = output_dir / args.resume_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Requested resume run does not exist: {run_dir}")
        cache_meta = _read_chunk_cache_meta(run_dir / "chunks.jsonl")
        if cache_meta is not None and str(cache_meta.get("source_id")) != _source_uuid(source):
            raise RuntimeError(
                f"E_RESUME_SOURCE_MISMATCH: run {args.resume_run_id} belongs to a different source document"
            )
        run_id = run_dir.name
        resumed_run = True
    elif auto_resume_failed:
        resumable = find_latest_resumable_run(output_dir, source, targets)
        if resumable is not None:
            run_dir = resumable
            run_id = run_dir.name
            resumed_run = True
        else:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = output_dir / run_id
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "batch.log"
    configure_logging(component=__name__, log_file=log_path)
    logger = get_logger(__name__)

    logger.info("Batch run initialized run_id=%s resumed=%s", run_id, resumed_run)
    if resumed_run:
        logger.warning("RESUME BATCH run_id=%s existing checkpoints/chunk cache will be reused", run_id)
        print(f"Resuming batch run: {run_id}", flush=True)
    log_environment(logger)
    _write_run_manifest(
        run_dir / "run.json",
        run_id=run_id,
        source=source,
        targets=targets,
        num_ctx=args.num_ctx,
        status="running",
        resumed=resumed_run,
    )
    logger.info(
        "Batch configuration source=%s openai_model_list=%s ollama_model_list=%s openai_models=%d ollama_models=%d ollama_host=%s openai_base_url=%s",
        source,
        model_list,
        ollama_model_list,
        len(openai_targets),
        len(ollama_targets),
        ollama_host,
        args.openai_base_url or loop_values.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "<sdk-default>",
    )
    logger.info(
        "Batch loop settings wait=%ss cleanup_wait=%ss cleanup_timeout=%ss stop_after_each=%s verify_output=%s verify_gpu=%s gpu_threshold=%s force_keep_alive_zero=%s concurrency=%d min_llm_chars=%d progress_every=%d child_heartbeat=%ss",
        task_wait_seconds,
        cleanup_wait_seconds,
        cleanup_timeout_seconds,
        stop_after_each,
        verify_output,
        verify_gpu,
        gpu_threshold_mb,
        force_keep_alive_zero,
        benchmark_concurrency,
        min_llm_chars,
        progress_every_chunks,
        subprocess_heartbeat_seconds,
    )
    obsidian_run_dir = obsidian_dir / run_id
    obsidian_run_dir.mkdir(parents=True, exist_ok=True)
    chunk_cache = run_dir / "chunks.jsonl"
    benchmark_wiki_base = llmwiki / "benchmark-runs" / run_id
    benchmark_wiki_base.mkdir(parents=True, exist_ok=True)
    prepare_shared_chunk_cache(project_root=project_root, source=source, cache_path=chunk_cache)
    if resumed_run:
        logger.info("RESUME chunk cache ready path=%s", chunk_cache)

    results: list[dict[str, Any]] = []

    print(f"Batch run: {run_id}")
    print(f"OpenAI model list: {model_list}")
    print(f"Ollama model list: {ollama_model_list}")
    print(f"Models: {len(targets)} (OpenAI={len(openai_targets)}, Ollama={len(ollama_targets)})")
    print()

    openai_base_url = args.openai_base_url or loop_values.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None

    for sequence, target in enumerate(targets, start=1):
        provider = target.provider
        model = target.model
        model_slug = slugify(f"{provider}-{model}")
        model_report = output_dir / run_id / f"{sequence:02d}-{model_slug}.md"
        model_json = output_dir / run_id / f"{sequence:02d}-{model_slug}.json"
        checkpoint_dir = resolve_checkpoint_dir(run_dir, sequence, target)
        print(f"\n[{sequence}/{len(targets)}] START {provider}/{model}", flush=True)
        logger.info("MODEL BATCH START sequence=%d/%d provider=%s model=%s", sequence, len(targets), provider, model)

        model_num_ctx = target.num_ctx or args.num_ctx
        model_concurrency = target.concurrency or benchmark_concurrency
        model_num_batch = target.num_batch
        model_num_predict = target.num_predict
        logger.info(
            "MODEL RUNTIME provider=%s model=%s concurrency=%d num_ctx=%d num_batch=%s num_predict=%s",
            provider,
            model,
            model_concurrency,
            model_num_ctx,
            model_num_batch if model_num_batch is not None else "default",
            model_num_predict if model_num_predict is not None else "default",
        )

        started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        try:
            return_code, stdout, stderr, summary = run_single_model(
                project_root=project_root,
                source=source,
                llmwiki=llmwiki,
                ontology=ontology,
                model=model,
                provider=provider,
                ollama_host=ollama_host,
                api_key_env=target.api_key_env,
                openai_base_url=openai_base_url,
                max_retries=args.max_retries,
                num_ctx=model_num_ctx,
                num_batch=model_num_batch,
                num_predict=model_num_predict,
                report_path=model_report,
                loop_values=loop_values,
                force_keep_alive_zero=force_keep_alive_zero,
                concurrency=model_concurrency,
                min_llm_chars=min_llm_chars,
                progress_every_chunks=progress_every_chunks,
                subprocess_heartbeat_seconds=subprocess_heartbeat_seconds,
                chunk_cache=chunk_cache,
                benchmark_wiki_base=benchmark_wiki_base,
                checkpoint_dir=checkpoint_dir,
            )
        except Exception as exc:
            logger.exception("MODEL BATCH EXCEPTION sequence=%d model=%s", sequence, model)
            return_code, stdout, stderr, summary = 1, "", f"{type(exc).__name__}: {exc}", None

        output_ok, output_message = verify_model_output(model_report) if verify_output else (True, "output verification disabled")
        status = "success" if return_code == 0 and (summary is not None) and output_ok else "failed"
        result_item: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "sequence": sequence,
            "status": status,
            "return_code": return_code,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "result": summary,
            "report_path": str(model_report),
            "log_path": str(model_report.with_suffix(".log")),
            "runtime": {
                "concurrency": model_concurrency,
                "num_ctx": model_num_ctx,
                "num_batch": model_num_batch,
                "num_predict": model_num_predict,
            },
            "output_verified": output_ok,
            "output_message": output_message,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "resumed_run": resumed_run,
            "checkpoint_dir": str(checkpoint_dir),
        }
        results.append(result_item)
        model_json.parent.mkdir(parents=True, exist_ok=True)
        model_json.write_text(json.dumps(result_item, ensure_ascii=False, indent=2), encoding="utf-8")

        write_obsidian_model_note(
            output_path=obsidian_dir / run_id / f"{sequence:02d}-{model_slug}.md",
            result=result_item,
            source=source,
            run_id=run_id,
            sequence=sequence,
        )

        print(f"[{sequence}/{len(targets)}] RESULT {provider}/{model}: {status} ({output_message})", flush=True)
        logger.info(
            "MODEL BATCH RESULT sequence=%d provider=%s model=%s status=%s return_code=%d output=%s summary=%s",
            sequence,
            provider,
            model,
            status,
            return_code,
            output_message,
            summary is not None,
        )

        # Do not advance to the next model until the current model's artifacts are confirmed
        # and its loaded model has been released. This matters on small VRAM GPUs.
        if provider == "ollama" and stop_after_each:
            if cleanup_wait_seconds:
                print(f"[{sequence}/{len(targets)}] cleanup wait {cleanup_wait_seconds}s", flush=True)
                time.sleep(cleanup_wait_seconds)
            stopped, stop_message = stop_model(model)
            print(f"[{sequence}/{len(targets)}] ollama stop: {'OK' if stopped else 'WARN'} {stop_message}", flush=True)
            unloaded, unload_message = wait_until_model_unloaded(model, cleanup_timeout_seconds, poll_seconds)
            print(f"[{sequence}/{len(targets)}] unload check: {'OK' if unloaded else 'WARN'} {unload_message}", flush=True)
            result_item["ollama_stop_ok"] = stopped
            result_item["ollama_stop_message"] = stop_message
            result_item["model_unloaded"] = unloaded
            result_item["model_unload_message"] = unload_message

            if verify_gpu:
                gpu_ok, gpu_message = wait_for_gpu_free(gpu_threshold_mb, cleanup_timeout_seconds, poll_seconds)
                print(f"[{sequence}/{len(targets)}] VRAM check: {'OK' if gpu_ok else 'WARN'} {gpu_message}", flush=True)
                result_item["gpu_memory_ok"] = gpu_ok
                result_item["gpu_memory_message"] = gpu_message
            else:
                result_item["gpu_memory_ok"] = None
                result_item["gpu_memory_message"] = "disabled"

            model_json.write_text(json.dumps(result_item, ensure_ascii=False, indent=2), encoding="utf-8")
            write_obsidian_model_note(
                output_path=obsidian_dir / run_id / f"{sequence:02d}-{model_slug}.md",
                result=result_item,
                source=source,
                run_id=run_id,
                sequence=sequence,
            )
        else:
            result_item["ollama_stop_ok"] = None
            result_item["model_unloaded"] = None
            result_item["gpu_memory_ok"] = None

        if sequence < len(targets):
            print(f"[{sequence}/{len(targets)}] waiting {task_wait_seconds}s before next model", flush=True)
            for remaining in range(task_wait_seconds, 0, -1):
                if remaining <= 10 or remaining % 60 == 0:
                    print(f"  next model in {remaining}s", flush=True)
                time.sleep(1)

        if status != "success" and not args.continue_on_error:
            print(f"Stopping batch because {model} failed.", file=sys.stderr, flush=True)
            break

    obsidian_batch = obsidian_dir / f"{run_id}.md"
    write_obsidian_batch_note(
        output_path=obsidian_batch,
        run_id=run_id,
        source=source,
        model_list_path=model_list,
        ollama_model_list_path=ollama_model_list,
        results=results,
    )
    write_overview_note(overview_path=obsidian_dir / "index.md", run_id=run_id, source=source, results=results)

    if args.batch_report:
        batch_report_path = Path(args.batch_report).expanduser().resolve()
        batch_report_path.parent.mkdir(parents=True, exist_ok=True)
        batch_report_path.write_text(obsidian_batch.read_text(encoding="utf-8"), encoding="utf-8")

    success_count = sum(item["status"] == "success" for item in results)
    _write_run_manifest(
        run_dir / "run.json",
        run_id=run_id,
        source=source,
        targets=targets,
        num_ctx=args.num_ctx,
        status="completed" if success_count == len(results) else "incomplete",
        resumed=resumed_run,
    )
    print(f"Batch complete: {success_count}/{len(results)} models succeeded")
    print(f"Raw run directory: {run_dir}")
    print(f"Obsidian benchmark: {obsidian_batch}")
    print(f"Model Wiki sandboxes: {benchmark_wiki_base}")
    return 0 if success_count == len(results) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Configure a fallback console logger if failure happened before run initialization.
        configure_logging(component=__name__)
        get_logger(__name__).exception("FATAL batch benchmark failure")
        raise
