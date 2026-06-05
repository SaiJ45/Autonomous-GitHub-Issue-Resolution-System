from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PipelineStatus(str, Enum):
    """Lifecycle states for the pipeline."""

    INIT = "INIT"
    ISSUES_FETCHED = "ISSUES_FETCHED"
    ISSUE_SELECTED = "ISSUE_SELECTED"
    REPO_READY = "REPO_READY"
    PATCH_GENERATED = "PATCH_GENERATED"
    INTEGRATED = "INTEGRATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    FAILED_TERMINATED = "FAILED_TERMINATED"
    MERGED = "MERGED"
    COMPLETED = "COMPLETED"


class DecisionType(str, Enum):
    """Routing decisions emitted by decision logic."""

    APPROVED = "approved"
    RETRY = "retry"
    FAIL = "fail"
    FAILED = "failed"


class FailureType(str, Enum):
    """Normalized failure categories for analysis and retry strategy."""

    RATE_LIMIT = "RATE_LIMIT"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    GUARDRAIL_FAILURE = "GUARDRAIL_FAILURE"
    LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
    LOGIC_ERROR = "LOGIC_ERROR"
    EDGE_CASE_MISSING = "EDGE_CASE_MISSING"
    MINIMAL_FIX_INSUFFICIENT = "MINIMAL_FIX_INSUFFICIENT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    REQUIREMENT_MISMATCH = "REQUIREMENT_MISMATCH"
    BRANCH_CONFLICT = "BRANCH_CONFLICT"
    PLAN_GENERATION_FAILED = "PLAN_GENERATION_FAILED"
    BEHAVIORAL_TESTS_FAILED = "BEHAVIORAL_TESTS_FAILED"
    UNKNOWN = "UNKNOWN"


class RepoContext(BaseModel):
    """Execution context for repository operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_path: str = ""
    branch: str = ""
    run_id: Optional[str] = None
    base_branch: Optional[str] = "main"
    workspace_path: Optional[str] = None
    commit_hash: Optional[str] = None


class FileChange(BaseModel):
    """A single structured file change payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    diff: str
    change_type: Literal["add", "modify", "delete"]


class QaFeedbackRecord(BaseModel):
    """Canonical QA feedback per attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int
    suggestions: str
    failed_tests: int = 0
    failure_type: Optional[FailureType] = None


class FailureRecord(BaseModel):
    """Structured failure snapshot for diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int
    failure_type: FailureType = FailureType.UNKNOWN
    reason: str = ""
    exception_type: Optional[str] = None
    exception_msg: Optional[str] = None


class AttemptRecord(BaseModel):
    """Retry attempt metadata and selected strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int
    strategy: str = "default"
    based_on_failure: Optional[FailureType] = None
    decision: Optional[DecisionType] = None
    changed_files: List[str] = Field(default_factory=list)
    patch_fingerprint: Optional[str] = None
    failed_tests: Optional[int] = None
    summary: str = ""


class NodeTrace(BaseModel):
    """Per-node observability record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    timestamp: float
    duration_ms: int
    success: bool
    input_summary: Dict[str, str] = Field(default_factory=dict)
    decision: Optional[str] = None
    output_summary: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None


class GeneratedTestCase(BaseModel):
    """Structured LLM-generated test case for simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    input: str
    expected_output: str
    reasoning: str


class SimulatedTestResult(BaseModel):
    """Reasoned simulation outcome for a generated test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_name: str
    status: Literal["pass", "fail", "uncertain"]
    reasoning: str
    confidence: float


class AgentState(BaseModel):
    """Immutable typed workflow state; update using model_copy(update=...)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue_id: str = ""
    issue_title: Optional[str] = None
    issue_body: Optional[str] = None

    repo: RepoContext = Field(default_factory=RepoContext)
    repo_path: Optional[str] = None
    branch: Optional[str] = None
    run_id: Optional[str] = None

    plan: Optional[Dict[str, Any]] = None
    patch: List[FileChange] = Field(default_factory=list)
    previous_patch: List[FileChange] = Field(default_factory=list)
    diff: str = ""
    repo_summary: Optional[str] = None
    patch_summary: Optional[str] = None
    file_summaries: Dict[str, str] = Field(default_factory=dict)
    generated_tests: List[Dict[str, Any]] = Field(default_factory=list)
    simulated_results: List[Dict[str, Any]] = Field(default_factory=list)
    generated_tests_cache_key: Optional[str] = None
    simulated_results_cache_key: Optional[str] = None
    risk_level: Optional[str] = None
    validation_confidence: Optional[float] = None

    qa_output: Optional[Dict[str, Any]] = None
    qa_feedback: Optional[Dict[str, Any]] = None  # structured feedback for planner on retry
    qa_feedback_history: List[QaFeedbackRecord] = Field(default_factory=list)
    test_results: Dict[str, int] = Field(default_factory=lambda: {"passed_tests": 0, "failed_tests": 0})
    tests_run: int = 0  # actual count of tests generated & simulated/executed
    qa_state: Optional[str] = None  # "PASS" | "FAIL" | "WEAK_PASS" — canonical QA state
    failed_tests: Optional[int] = None
    
    # Behavioral validation & alignment results (added for comprehensive validation)
    behavioral_validation: Optional[Dict[str, Any]] = None  # {"status": "PASS|FAIL", "findings": [...]}
    issue_alignment: Optional[Dict[str, Any]] = None  # {"status": "PASS|FAIL", "reason": "..."}
    regression_check: Optional[Dict[str, Any]] = None  # {"status": "PASS|FAIL", "removed_functions": [...]}
    
    edge_cases: List[str] = Field(default_factory=list)
    retry_strategy: str = "normal"
    top_candidate_file: Optional[str] = None
    failure_stage: Optional[str] = None           # "planner" | "coder" | "validation" | "unknown"
    planner_mode: Optional[str] = None            # "full" | "feedback_guided" — controls planner skip
    semantic_signals: Optional[Dict[str, Any]] = None  # patch quality signals from issue_agent_node
    
    # New semantic alignment & behavioral validation fields (enforce Rules 1-10)
    semantic_alignment_pass: bool = False  # target function/class from plan actually modified?
    behavioral_change_detected: bool = False  # logic changed (NOT just imports or formatting)?
    feature_completeness_pass: bool = True  # all required implementation steps present?
    fallback_used: bool = False  # did diff fallback strategy execute?
    execution_path_reachable: bool = False  # is modified logic in executable code path?

    status: PipelineStatus = PipelineStatus.INIT
    decision: Optional[DecisionType] = None
    retry_count: int = 0
    cross_verify_retry_count: int = 0
    failure_type: Optional[FailureType] = None
    failure_reason: Optional[str] = None

    issue_pr_number: Optional[str] = None
    qa_pr_number: Optional[str] = None
    issue_prs: List[str] = Field(default_factory=list)
    qa_prs: List[str] = Field(default_factory=list)
    approved_pr: Optional[str] = None
    pr_url: Optional[str] = None

    error: Optional[str] = None
    pr_status: Optional[str] = None
    exception_type: Optional[str] = None
    exception_msg: Optional[str] = None

    failures: List[FailureRecord] = Field(default_factory=list)
    attempts: List[AttemptRecord] = Field(default_factory=list)
    node_traces: List[NodeTrace] = Field(default_factory=list)

