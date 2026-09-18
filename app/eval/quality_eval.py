from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_RELATION_ENDPOINT_ERROR = "unresolved relation endpoint"


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _mean(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not items:
        return None
    return sum(items) / len(items)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\[\[|\]\]", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _find_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the most likely model-output payload from a checkpoint record.

    The benchmark implementation has changed a few times, so this deliberately
    accepts several wrapper keys instead of depending on one exact checkpoint schema.
    """
    candidates = [
        record.get("output"),
        record.get("parsed"),
        record.get("data"),
        record.get("document"),
        record.get("result"),
        record.get("response"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            if any(k in candidate for k in ("concepts", "entities", "relations", "citations", "summary", "evidence")):
                return candidate

    if any(k in record for k in ("concepts", "entities", "relations", "citations", "summary", "evidence")):
        return record

    for mapping in _walk_dicts(record):
        if any(k in mapping for k in ("concepts", "entities", "relations")):
            return mapping
    return record


def _extract_chunk_id(record: dict[str, Any]) -> str | None:
    value = _first(record, "chunk_id", "id", "source_chunk_id")
    if value:
        return str(value)
    for mapping in _walk_dicts(record):
        value = _first(mapping, "chunk_id", "source_chunk_id")
        if value:
            return str(value)
    return None


def _extract_bool(record: dict[str, Any], *keys: str) -> bool | None:
    for mapping in _walk_dicts(record):
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "1", "yes", "y", "pass", "passed"}:
                        return True
                    if lowered in {"false", "0", "no", "n", "fail", "failed"}:
                        return False
    return None


def _extract_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for mapping in _walk_dicts(record):
        for key in ("ontology_errors", "validation_errors", "errors"):
            value = mapping.get(key)
            if isinstance(value, list):
                errors.extend(str(item) for item in value)
            elif isinstance(value, str) and value.strip():
                errors.append(value)
    return errors


def _name_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, dict):
        return None
    value = _first(item, "name", "title", "label", "concept", "entity", "id")
    return str(value).strip() if value not in (None, "") else None


def extract_concepts(record: dict[str, Any]) -> list[str]:
    payload = _find_payload(record)
    names: list[str] = []
    for key in ("concepts", "entities", "nodes"):
        for item in _as_list(payload.get(key)):
            name = _name_from_item(item)
            if name:
                names.append(name)
    return names


def _relation_triplet(item: Any) -> tuple[str, str, str] | None:
    if not isinstance(item, dict):
        return None
    source = _first(item, "source", "from", "subject", "source_name", "source_title")
    rel_type = _first(item, "type", "relation", "predicate", "relation_type")
    target = _first(item, "target", "to", "object", "target_name", "target_title")

    def unwrap(value: Any) -> Any:
        if isinstance(value, dict):
            return _first(value, "name", "title", "label", "id")
        return value

    source, rel_type, target = unwrap(source), unwrap(rel_type), unwrap(target)
    if source in (None, "") or rel_type in (None, "") or target in (None, ""):
        return None
    return str(source), str(rel_type), str(target)


def extract_relations(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    payload = _find_payload(record)
    output: list[tuple[str, str, str]] = []
    for item in _as_list(payload.get("relations")):
        triplet = _relation_triplet(item)
        if triplet:
            output.append(triplet)
    return output


def _citation_present(record: dict[str, Any]) -> bool | None:
    explicit = _extract_bool(record, "citation_preserved", "source_citation_preserved", "citation_valid")
    if explicit is not None:
        return explicit

    payload = _find_payload(record)
    citations = _as_list(payload.get("citations"))
    if citations:
        return True

    for key in ("source", "sources", "evidence", "provenance"):
        value = payload.get(key)
        if value:
            return True
    return None


def load_checkpoint_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load JSON/JSONL checkpoints recursively.

    Returns (records, parse_failures). JSON arrays are expanded into records.
    """
    records: list[dict[str, Any]] = []
    parse_failures = 0

    if not path.exists():
        return records, parse_failures

    candidates: list[Path]
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        )

    for file_path in candidates:
        try:
            if file_path.suffix.lower() == ".jsonl":
                for line in file_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        parse_failures += 1
                        continue
                    if isinstance(value, dict):
                        records.append(value)
                    elif isinstance(value, list):
                        records.extend(item for item in value if isinstance(item, dict))
            else:
                value = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    records.append(value)
                elif isinstance(value, list):
                    records.extend(item for item in value if isinstance(item, dict))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            parse_failures += 1

    # Deduplicate by chunk_id when possible. Some checkpoint layouts keep both
    # intermediate and final copies of the same chunk.
    deduped: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for record in records:
        chunk_id = _extract_chunk_id(record)
        if chunk_id:
            deduped[chunk_id] = record
        else:
            anonymous.append(record)
    return list(deduped.values()) + anonymous, parse_failures


@dataclass
class GoldenRelation:
    source: str
    relation: str
    target: str


@dataclass
class GoldenSample:
    chunk_id: str
    expected_concepts: list[str] = field(default_factory=list)
    expected_relations: list[GoldenRelation] = field(default_factory=list)


def load_golden_dir(path: Path | None) -> dict[str, GoldenSample]:
    if path is None or not path.exists():
        return {}

    samples: dict[str, GoldenSample] = {}
    for file_path in sorted(path.glob("*.json")):
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = raw if isinstance(raw, list) else [raw]
        for row in rows:
            if not isinstance(row, dict) or not row.get("chunk_id"):
                continue
            relations: list[GoldenRelation] = []
            for item in _as_list(row.get("expected_relations")):
                triplet = _relation_triplet(item)
                if triplet:
                    relations.append(GoldenRelation(triplet[0], triplet[1], triplet[2]))
            samples[str(row["chunk_id"])] = GoldenSample(
                chunk_id=str(row["chunk_id"]),
                expected_concepts=[str(v) for v in _as_list(row.get("expected_concepts")) if str(v).strip()],
                expected_relations=relations,
            )
    return samples


def _prf(predicted: set[Any], expected: set[Any]) -> tuple[float | None, float | None, float | None]:
    if not predicted and not expected:
        return None, None, None
    tp = len(predicted & expected)
    precision = _safe_div(tp, len(predicted)) if predicted else 0.0
    recall = _safe_div(tp, len(expected)) if expected else 0.0
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _normalized_relation(rel: tuple[str, str, str]) -> tuple[str, str, str]:
    return tuple(normalize_text(part) for part in rel)  # type: ignore[return-value]


@dataclass
class QualityResult:
    checkpoint_records: int
    checkpoint_parse_failures: int
    json_parse_rate: float | None
    schema_pass_rate: float | None
    citation_presence_rate: float | None
    ontology_compliance_rate: float | None
    relation_endpoint_validity_rate: float | None
    relation_count: int
    unresolved_relation_endpoints: int
    concept_precision: float | None = None
    concept_recall: float | None = None
    concept_f1: float | None = None
    relation_precision: float | None = None
    relation_recall: float | None = None
    relation_f1: float | None = None
    golden_samples_matched: int = 0
    structural_score: float | None = None
    knowledge_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    parse_failures: int = 0,
    golden: dict[str, GoldenSample] | None = None,
) -> QualityResult:
    golden = golden or {}
    total_units = len(records) + parse_failures
    json_parse_rate = _safe_div(len(records), total_units)

    schema_values = [v for record in records if (v := _extract_bool(record, "schema_valid", "schema_passed")) is not None]
    citation_values = [v for record in records if (v := _citation_present(record)) is not None]

    ontology_values: list[bool] = []
    relation_count = 0
    unresolved_count = 0

    predicted_concepts: set[tuple[str, str]] = set()
    expected_concepts: set[tuple[str, str]] = set()
    predicted_relations: set[tuple[str, str, str, str]] = set()
    expected_relations: set[tuple[str, str, str, str]] = set()
    matched_golden = 0

    for record in records:
        errors = _extract_errors(record)
        explicit_ontology = _extract_bool(record, "ontology_valid", "ontology_compliant")
        if explicit_ontology is not None:
            ontology_values.append(explicit_ontology)
        else:
            ontology_errors_seen = False
            for mapping in _walk_dicts(record):
                if "ontology_errors" in mapping:
                    ontology_errors_seen = True
                    value = mapping.get("ontology_errors")
                    if isinstance(value, list):
                        ontology_values.append(len(value) == 0)
                    elif isinstance(value, str):
                        ontology_values.append(not value.strip())
                    break
            if not ontology_errors_seen:
                count = None
                for mapping in _walk_dicts(record):
                    if "ontology_error_count" in mapping:
                        try:
                            count = int(mapping["ontology_error_count"])
                        except (TypeError, ValueError):
                            count = None
                        break
                if count is not None:
                    ontology_values.append(count == 0)

        relations = extract_relations(record)
        relation_count += len(relations)
        unresolved_count += sum(_RELATION_ENDPOINT_ERROR in error.lower() for error in errors)

        chunk_id = _extract_chunk_id(record)
        if chunk_id and chunk_id in golden:
            matched_golden += 1
            sample = golden[chunk_id]
            predicted_concepts.update(
                (chunk_id, normalize_text(v))
                for v in extract_concepts(record)
                if normalize_text(v)
            )
            expected_concepts.update(
                (chunk_id, normalize_text(v))
                for v in sample.expected_concepts
                if normalize_text(v)
            )
            predicted_relations.update(
                (chunk_id, *_normalized_relation(v)) for v in relations
            )
            expected_relations.update(
                (chunk_id, *_normalized_relation((v.source, v.relation, v.target)))
                for v in sample.expected_relations
            )

    schema_rate = _mean(1.0 if v else 0.0 for v in schema_values)
    citation_rate = _mean(1.0 if v else 0.0 for v in citation_values)
    ontology_rate = _mean(1.0 if v else 0.0 for v in ontology_values)
    relation_validity = None
    if relation_count > 0:
        relation_validity = max(0.0, 1.0 - (unresolved_count / relation_count))

    concept_precision = concept_recall = concept_f1 = None
    relation_precision = relation_recall = relation_f1 = None
    if matched_golden:
        concept_precision, concept_recall, concept_f1 = _prf(predicted_concepts, expected_concepts)
        relation_precision, relation_recall, relation_f1 = _prf(predicted_relations, expected_relations)

    structural_parts = [
        (schema_rate, 0.30),
        (citation_rate, 0.25),
        (ontology_rate, 0.25),
        (relation_validity, 0.20),
    ]
    available = [(score, weight) for score, weight in structural_parts if score is not None]
    structural_score = None
    if available:
        structural_score = sum(score * weight for score, weight in available) / sum(weight for _, weight in available)

    knowledge_score = None
    if matched_golden:
        knowledge_parts = [
            (concept_f1, 0.45),
            (relation_f1, 0.45),
            (citation_rate, 0.10),
        ]
        k_available = [(score, weight) for score, weight in knowledge_parts if score is not None]
        if k_available:
            knowledge_score = sum(score * weight for score, weight in k_available) / sum(weight for _, weight in k_available)

    return QualityResult(
        checkpoint_records=len(records),
        checkpoint_parse_failures=parse_failures,
        json_parse_rate=json_parse_rate,
        schema_pass_rate=schema_rate,
        citation_presence_rate=citation_rate,
        ontology_compliance_rate=ontology_rate,
        relation_endpoint_validity_rate=relation_validity,
        relation_count=relation_count,
        unresolved_relation_endpoints=unresolved_count,
        concept_precision=concept_precision,
        concept_recall=concept_recall,
        concept_f1=concept_f1,
        relation_precision=relation_precision,
        relation_recall=relation_recall,
        relation_f1=relation_f1,
        golden_samples_matched=matched_golden,
        structural_score=structural_score,
        knowledge_score=knowledge_score,
    )
