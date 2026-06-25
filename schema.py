"""Typed structures for evidence, scores, gates, and the LangGraph state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict


@dataclass
class Evidence:
    criterion_id: str
    claim: str
    tier: int               # 1..4, see config.EVIDENCE_TIERS
    source_title: str
    source_url: str
    snippet: str = ""

    def to_row(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "tier": self.tier,
            "claim": self.claim,
            "source": self.source_title,
            "url": self.source_url,
        }


@dataclass
class CriterionScore:
    criterion_id: str
    category: str
    description: str
    weight: float
    score: Optional[int]        # None == N/A / not evaluable -> BLANK, never 0
    evidence_tier: Optional[int]
    confidence: float           # 0..1
    rationale: str
    sources: list[str] = field(default_factory=list)
    is_gap: bool = False        # no usable evidence found
    capped: bool = False        # score was limited by evidence tier

    @property
    def weighted(self) -> float:
        return 0.0 if self.score is None else round(self.weight * self.score, 4)


@dataclass
class GateResult:
    gate_id: str
    criterion_id: str
    label: str
    passed: Optional[bool]      # None == undetermined (treated as fail-safe flag)
    rationale: str
    evidence_tier: Optional[int] = None


@dataclass
class ConsistencyReport:
    agreement_within_1: float        # share of overlapping criteria within 1 point
    mean_abs_diff: float
    divergences: list[dict] = field(default_factory=list)  # criterion-level deltas
    judge_model: str = ""
    note: str = ""


@dataclass
class RunMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    web_search_requests: int = 0
    llm_calls: int = 0
    node_seconds: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    def add_usage(self, usage: dict) -> None:
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.web_search_requests += usage.get("web_search_requests", 0)
        self.llm_calls += 1


class AgentState(TypedDict, total=False):
    # inputs
    vendor_name: str
    use_case: str
    gates_cfg: list           # list[config.Gate]
    # node 1
    evidence: list            # list[Evidence]
    # node 2
    gate_results: list        # list[GateResult]
    eliminated: bool
    elimination_reasons: list
    scores: list              # list[CriterionScore]
    weighted_total: float
    normalized: float
    # node 3
    consistency: object       # ConsistencyReport | None
    # cross-cutting
    metrics: object           # RunMetrics
    errors: list
