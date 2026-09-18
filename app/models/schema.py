from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    CONCEPT = "concept"
    ENTITY = "entity"
    SOURCE = "source"
    COMPARISON = "comparison"


class RelationType(str, Enum):
    USES = "uses"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    COMPARES_WITH = "compares_with"


class SourceRefDraft(BaseModel):
    """LLM-facing provenance. No UUID generation is requested from the model."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1, description="Path relative to llmwiki/raw")
    locator: str = Field(default="", description="Page, section, heading, or other source locator")
    chunk_key: str = Field(default="", description="Stable chunk key supplied by the pipeline")
    quote: str = Field(default="", description="Short grounding quote")


class SourceRef(BaseModel):
    """Materialized provenance with pipeline-owned UUIDs."""

    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    chunk_id: UUID
    source_file: str = Field(min_length=1, description="Path relative to llmwiki/raw")
    locator: str = Field(default="", description="Page, section, heading, or other source locator")
    quote: str = Field(default="", description="Short grounding quote")


class RelationDraft(BaseModel):
    """LLM-facing relation. Targets are semantic names before identity resolution."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    relation: RelationType
    target: str = Field(min_length=1)
    evidence: str = Field(default="")
    source_refs: list[SourceRefDraft] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ConceptDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    definition: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    source_refs: list[SourceRefDraft] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EntityDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    description: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    source_refs: list[SourceRefDraft] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class KnowledgeExtraction(BaseModel):
    """LLM-facing structured extraction.

    Identity is resolved after model validation so re-running a model does not
    create a different UUID for the same logical knowledge object.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    language: Literal["ko", "en", "mixed", "unknown"] = "mixed"
    abstract: str = Field(default="")
    concepts: list[ConceptDraft] = Field(default_factory=list)
    entities: list[EntityDraft] = Field(default_factory=list)
    relations: list[RelationDraft] = Field(default_factory=list)
    source_refs: list[SourceRefDraft] = Field(default_factory=list)
    wiki_paths: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def validate_sources(cls, value: list[SourceRefDraft]) -> list[SourceRefDraft]:
        if len(value) > 100:
            raise ValueError("too many source_refs")
        return value


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source: UUID
    relation: RelationType
    target: UUID
    evidence: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    type: DocumentType
    title: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    definition: str = ""
    entity_type: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_id: UUID
    source_file: str
    ordinal: int = Field(ge=0)
    text: str
    meta: dict = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    chunk_id: UUID
    success: bool
    schema_valid: bool
    recovered_json: bool = False
    attempts: int = 1
    ttft_ms: float | None = None
    wall_time_ms: float | None = None
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ms: float | None = None
    tokens_per_sec: float | None = None
    source_citation_preserved: bool = False
    ontology_compliance_rate: float = 0.0
    error: str | None = None




class WikiQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: int = 0
    concepts: int = 0
    entities: int = 0
    sources: int = 0
    comparisons: int = 0
    relations: int = 0
    unresolved_relation_targets: int = 0
    relation_integrity_rate: float = 1.0
    sourced_document_rate: float = 0.0
    referenced_chunks: int = 0
    chunk_reference_coverage: float = 0.0
    orphan_documents: int = 0
    orphan_document_rate: float = 0.0
    duplicate_titles: int = 0
    average_relations_per_document: float = 0.0

class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "ollama"
    model: str
    document: str
    source_id: UUID
    chunks: int
    successful_chunks: int
    schema_pass_rate: float
    json_recovery_rate: float
    source_citation_rate: float
    ontology_compliance_rate: float
    mean_ttft_ms: float | None = None
    mean_tokens_per_sec: float | None = None
    mean_prompt_eval_ms: float | None = None
    mean_output_tokens: float | None = None
    p95_output_tokens: float | None = None
    max_output_tokens: int | None = None
    mean_total_duration_ms: float | None = None
    p95_total_duration_ms: float | None = None
    effective_chunks_per_min: float | None = None
    raw_relation_count: int = 0
    retained_relation_count: int = 0
    dropped_unresolved_relation_count: int = 0
    wiki_root: str | None = None
    wiki_quality: WikiQualityMetrics | None = None
    details: list[BenchmarkMetrics] = Field(default_factory=list)
