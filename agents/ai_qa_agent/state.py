from typing import TypedDict, List, Optional


class QAState(TypedDict, total=False):
    # ── Core Pipeline Identity ──
    repo_url: str
    repo_name: str
    pr_number: int

    # ── Issue context (injected by orchestrator) ──
    issue_title: str
    issue_body: str
    issue_id: str
    issue_pr_number: str       # PR number created by issue_pr_node
    changed_files: List[str]   # list of patched file paths
    diff: str                  # pre-computed git diff (passed from diff_node)

    # ── Node 1: Setup Repo ──
    repo_path: str
    repo_status: str  # "cloned" | "updated"

    # ── Node 2: Analyze Diff ──
    files_changed: List[str]
    issues_found: List[str]
    risk_level: str  # "low" | "medium" | "high"
    diff_decision: str
    test_cases: List[str]
    tests_passed: List[str]
    tests_failed: List[str]
    execution_errors: str
    suggestions: List[str]
    final_comment: str
    confidence: str  # "low" | "medium" | "high"
    test_execution_reasoning: str

    # ── Node 3: Run Tests ──
    tests_ran: bool
    exit_code: int
    test_passed: bool
    test_failures: List[str]
    test_logs: str

    # ── Node 4: Decision ──
    decision: str  # "APPROVED" | "REJECTED" | "REVIEW_REQUIRED"
    reason: str
    requires_pr: bool

    # ── Node 5: Executor ──
    action_log: str
    execution_status: str  # "SUCCESS" | "FAILED"

    # ── ReAct Context Engineering ──
    reasoning_history: str  # Accumulated Thought/Action/Observation steps
    repo_signals: str  # Extracted imports, function names, patterns

    # ── Error Handling ──
    error: Optional[str]
    status: str  # "IN_PROGRESS" | "FAILED" | "COMPLETED"
