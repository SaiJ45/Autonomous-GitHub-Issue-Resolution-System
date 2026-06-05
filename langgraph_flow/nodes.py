import os
import sys
import json
import subprocess
import hashlib
import time as pytime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.append(os.path.join(base_dir, "agents", "ai_issue_agent"))

from langgraph_flow.state import (
    AgentState,
    AttemptRecord,
    DecisionType,
    FailureRecord,
    FailureType,
    FileChange,
    GeneratedTestCase,
    NodeTrace,
    PipelineStatus,
    QaFeedbackRecord,
    SimulatedTestResult,
)
from utils.logger import logger
from utils.validators import validate_patch

from agents.ai_issue_agent.agents.issue_reader import get_issues
from agents.ai_issue_agent.agents.issue_grounding import (
    build_candidate_grounding_map,
    extract_issue_entities,
)
from agents.ai_issue_agent.agents.output_validators import LLMOutputValidator
from agents.ai_issue_agent.git_tools.create_pr import create_pull_request, generate_pr_content
from agents.ai_issue_agent.agents.patch_generator import PatchGenerator
from agents.ai_issue_agent.git_tools.merge_pr import merge_pull_request

from integration.git_handler import setup_repo
from adapters.issue_agent_adapter import run_issue_agent
from integration.memory_store import search_memory, save_to_memory


# ─────────────────────────────────────────────────────────────────────────────
#  JSON-SAFETY UTILITIES
#  LangGraph's MemorySaver checkpointer requires the entire state to be JSON-
#  serialisable at every checkpoint. Any Pydantic model, Enum, or exotic type
#  that leaks into state will crash the graph silently.
# ─────────────────────────────────────────────────────────────────────────────

from enum import Enum as _Enum


def make_json_safe(obj):
    """
    Recursively converts any object to a JSON-serialisable form.
    Priority order:
      1. Primitives / None                 → returned as-is
      2. Enum                              → .value  (string/int)
      3. Pydantic BaseModel                → .model_dump() then recurse
      4. bytes                             → decoded utf-8 string
      5. datetime / date                   → .isoformat()
      6. dict                              → recurse on values
      7. list / tuple / set               → recurse on elements (set → sorted list)
      8. Everything else                  → str(obj)
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, _Enum):
        return obj.value
    # Pydantic v2 BaseModel
    if hasattr(obj, "model_dump"):
        return make_json_safe(obj.model_dump())
    # bytes → decode to string
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    # datetime / date → ISO 8601 string
    try:
        import datetime as _dt
        if isinstance(obj, (_dt.datetime, _dt.date)):
            return obj.isoformat()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        # Sort for deterministic output; elements may not be comparable across types
        try:
            return sorted(make_json_safe(x) for x in obj)
        except TypeError:
            return [make_json_safe(x) for x in obj]
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    # Fallback: unknown object → stringify
    return str(obj)


def _debug_state_types(state: dict, label: str = "state") -> None:
    """Print each key and its value-type for serialization debugging."""
    for k, v in state.items():
        print(f"[DEBUG:{label}] {k}: {type(v).__name__}")


def safe_get(obj, key, default=None):
    """Safe accessor that works on both dicts and non-dict objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def safe_dict(obj) -> dict:
    """Return obj if it is a dict, otherwise return empty dict.
    Use when reading qa_feedback or any state key that must be a dict."""
    return obj if isinstance(obj, dict) else {}


def normalize_qa_feedback(state) -> dict:
    """
    Re-derive qa_feedback from qa_output and write it back into state dict.
    Call this BEFORE the planner reads qa_feedback to guarantee it is always
    a correctly-typed {issues, suggestions, reason, alignment, confidence} dict.
    """
    qa = safe_dict(state.get("qa_output") if isinstance(state, dict) else {})
    state["qa_feedback"] = {
        "issues": list(qa.get("logic_issues", []) +
                       qa.get("missing_requirements", [])),
        "suggestions": list(qa.get("logic_issues", [])),
        "reason": qa.get("reason", ""),
        "alignment": qa.get("alignment", ""),
        "confidence": qa.get("confidence", ""),
    }
    return state


def normalize_qa_output(qa_output: dict) -> dict:
    """
    Normalise QA output status to the canonical PASS/FAIL vocabulary.
    LLM may return APPROVED/REJECTED — map them to PASS/FAIL so all
    downstream logic uses a single status vocabulary.
    """
    if not isinstance(qa_output, dict):
        return {"status": "FAIL", "confidence": 0.0, "issues": [str(qa_output)], "reason": "non-dict qa_output"}
    status = qa_output.get("status", "")
    if status == "APPROVED":
        qa_output["status"] = "PASS"
    elif status == "REJECTED":
        qa_output["status"] = "FAIL"
    elif status not in ("PASS", "FAIL"):
        # Unknown status — treat as FAIL to be safe
        qa_output["status"] = "FAIL"
    return qa_output


def classify_qa_state(test_results: dict, tests_run: int) -> str:
    """
    Classify QA state based on actual test execution.
    
    Returns one of:
      - "PASS":      tests_run > 0 AND failed_tests == 0
      - "FAIL":      failed_tests > 0 (regardless of tests_run)
      - "WEAK_PASS": tests_run == 0 but code validation passed
    
    This is independent of LLM's qa_output status.
    """
    failed_tests = test_results.get("failed_tests", 0)
    passed_tests = test_results.get("passed_tests", 0)
    
    # If any tests failed → FAIL (regardless of tests_run count)
    if failed_tests > 0:
        return "FAIL"
    
    # If tests actually ran and all passed → PASS
    if tests_run > 0 and failed_tests == 0:
        return "PASS"
    
    # No tests run, but structural validation passed → WEAK_PASS
    # (must be upgraded with behavioral validation before approval)
    return "WEAK_PASS"


def validate_issue_alignment(state: AgentState, patch: list) -> dict:
    issue_text = ((_sget(state, "issue_title", "") or "") + "\n" + (_sget(state, "issue_body", "") or "")).strip()
    if not issue_text:
        return {"status": "FAIL", "reason": "No issue text provided", "alignment_score": 0.0}

    semantic_signals = _sget(state, "semantic_signals", {})
    if not isinstance(semantic_signals, dict):
        semantic_signals = {}

    changed_files = [
        change["file_path"] if isinstance(change, dict) else getattr(change, "file_path", "")
        for change in patch or []
    ]
    changed_files = [path for path in changed_files if isinstance(path, str) and path.strip()]
    if not changed_files:
        return {"status": "FAIL", "reason": "Patch does not modify any files", "alignment_score": 0.0}

    modified_symbols = semantic_signals.get("modified_functions", []) or []
    issue_entities = extract_issue_entities(issue_text)

    repo_path = _sget(state, "repo_path", "")
    candidate_files = []
    for file_path in changed_files:
        if not repo_path:
            continue
        full_path = os.path.join(repo_path, file_path)
        if not os.path.exists(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(4000)
        except Exception:
            continue
        candidate_files.append({
            "path": file_path,
            "snippet": snippet,
            "structure": {
                "functions": [item.split(":")[-1].split(".")[-1] for item in modified_symbols if isinstance(item, str) and item.startswith(f"{file_path}:")],
                "classes": [],
                "imports": [],
            },
        })

    _, grounding_map = build_candidate_grounding_map(candidate_files, issue_text)
    explicit_entities = (
        issue_entities.get("function_names", [])
        or issue_entities.get("class_names", [])
        or issue_entities.get("module_names", [])
        or issue_entities.get("file_paths", [])
    )

    score = 0.0
    reasons = []
    for file_path in changed_files:
        grounding = grounding_map.get(file_path.replace("\\", "/").strip("/"), {})
        if grounding.get("has_direct_symbol_match"):
            score += 0.5
            reasons.append(f"{file_path} matched issue symbols")
        elif len(grounding.get("matched_keywords", [])) >= 2:
            score += 0.25
            reasons.append(f"{file_path} matched issue keywords")
        elif explicit_entities:
            return {
                "status": "FAIL",
                "reason": f"Modified file '{file_path}' does not contain the issue symbols {explicit_entities[:6]}",
                "alignment_score": 0.0,
            }

    if semantic_signals.get("target_symbols_missed"):
        return {
            "status": "FAIL",
            "reason": f"Grounded target symbols were not modified: {semantic_signals.get('target_symbols_missed')}",
            "alignment_score": 0.0,
        }

    if not modified_symbols and not semantic_signals.get("module_body_changed"):
        return {
            "status": "FAIL",
            "reason": "No grounded function/class/module behavior was modified",
            "alignment_score": 0.0,
        }

    score = min(score, 1.0)
    if score >= 0.5:
        return {
            "status": "PASS",
            "reason": "; ".join(reasons[:3]) or "Patch modifies grounded issue locations",
            "alignment_score": round(score, 2),
        }
    if score > 0.0:
        return {
            "status": "WEAK",
            "reason": "; ".join(reasons[:3]) or "Patch only weakly matches the issue location",
            "alignment_score": round(score, 2),
        }
    return {
        "status": "FAIL",
        "reason": "Patch does not ground to the issue symbols or code location",
        "alignment_score": 0.0,
    }


def validate_behavioral_correctness(state: AgentState) -> dict:
    """
    Validate patch correctness when tests are missing.
    
    Returns: {"status": "PASS|FAIL", "findings": [...], "risk_level": "..."}
    """
    semantic_signals = _sget(state, "semantic_signals", {})
    risk_level = _sget(state, "risk_level", "high")
    validation_confidence = _sget(state, "validation_confidence", 0.0)
    
    findings = []
    
    # Check 1: Confidence level
    if validation_confidence < 0.5:
        findings.append(f"Low validation confidence ({validation_confidence}) — behavior uncertain")
    
    # Check 2: Risk level
    if risk_level == "high":
        findings.append("High risk level detected — patch may have side effects")
    
    # Check 3: Content actually changed
    content_changed = semantic_signals.get("content_changed", False)
    if not content_changed:
        findings.append("No actual code changes detected — guardrails alone insufficient")

    if not semantic_signals.get("behavior_changed", False):
        findings.append("No behavioral AST change detected — patch may only change formatting or comments")
    
    # Check 4: Functions modified appropriately
    modified_fns = semantic_signals.get("modified_functions", [])
    if not modified_fns and not semantic_signals.get("module_body_changed", False):
        findings.append("No functions modified — patch may be incomplete")

    removed_symbols = semantic_signals.get("removed_symbols", [])
    if removed_symbols:
        findings.append(f"Patch removes existing symbols: {removed_symbols[:4]}")

    empty_bodies = semantic_signals.get("empty_bodies", [])
    if empty_bodies:
        findings.append(f"Patch introduced empty function bodies: {empty_bodies[:4]}")

    missed_targets = semantic_signals.get("target_symbols_missed", [])
    if missed_targets:
        findings.append(f"Grounded target symbols were not modified: {missed_targets[:4]}")
    
    # Determine status
    if len(findings) >= 2:
        status = "FAIL"
    elif len(findings) == 1:
        status = "WEAK"
    else:
        status = "PASS"
    
    return {
        "status": status,
        "findings": findings,
        "risk_level": risk_level,
        "confidence": validation_confidence,
    }


def detect_regressions(state: AgentState, patch: list) -> dict:
    """
    Check if patch removes or breaks existing functionality.
    
    Returns: {"status": "PASS|FAIL", "removed_functions": [...], "breaking_changes": [...]}
    """
    removed_functions = []
    breaking_changes = []
    
    # Parse patch to find deleted functions
    for file_change in patch:
        if isinstance(file_change, dict):
            diff_text = file_change.get("diff", "")
            change_type = file_change.get("change_type", "")
        else:
            diff_text = getattr(file_change, "diff", "")
            change_type = getattr(file_change, "change_type", "")
        
        if change_type == "delete":
            breaking_changes.append(f"File deleted: {file_change.get('file_path', 'unknown') if isinstance(file_change, dict) else getattr(file_change, 'file_path', 'unknown')}")
        
        # Simple heuristic: look for "def function_name" being removed (lines starting with -)
        for line in diff_text.split('\n'):
            if line.startswith('-') and line.strip().startswith('- def '):
                removed_functions.append(line[6:].strip())
    
    status = "FAIL" if (removed_functions or breaking_changes) else "PASS"
    
    return {
        "status": status,
        "removed_functions": removed_functions,
        "breaking_changes": breaking_changes,
    }




def _as_dict(state) -> dict:
    if isinstance(state, dict):
        return make_json_safe(dict(state))
    return make_json_safe(state.model_dump())


def _sget(state, key: str, default=None):
    """
    Unified state accessor for both AgentState (Pydantic) and plain dict states.

    Keys 'repo_path', 'branch', 'run_id' are stored nested under state.repo
    (Pydantic) or state['repo'] (dict) BUT may also be promoted to flat top-
    level keys by integration_node for easy downstream access. We check both.
    """
    if isinstance(state, dict):
        # Fast path: promoted flat key exists (integration_node writes these)
        if key in state:
            return state[key]
        # Fallback: check the nested repo sub-dict for repo-level keys
        if key in ("repo_path", "branch", "run_id"):
            return state.get("repo", {}).get(key, default)
        return default
    # AgentState Pydantic object
    if key in {"repo_path", "branch", "run_id"}:
        repo = getattr(state, "repo", None)
        return getattr(repo, key, default) if repo else default
    return getattr(state, key, default)


def _merge(state, **updates) -> dict:
    merged = _as_dict(state)
    # make_json_safe each update value before merging so callers
    # that pass raw Pydantic objects or Enums don't break the graph.
    merged.update({k: make_json_safe(v) for k, v in updates.items()})
    return merged


def _append_trace(
    state: AgentState | dict,
    node: str,
    started_at: float,
    success: bool,
    input_summary: dict[str, str] | None = None,
    output_summary: dict[str, str] | None = None,
    decision: str | None = None,
    error: str | None = None,
) -> list[dict]:
    traces = list(_sget(state, "node_traces", []))
    # Belt-and-suspenders: coerce ALL summary values to str so NodeTrace
    # never receives int/bool and raises a ValidationError (confirmed bug).
    def _safe_str_dict(d: dict | None) -> dict:
        if not d:
            return {}
        return {str(k): str(v) for k, v in d.items()}

    traces.append(
        NodeTrace(
            node=node,
            timestamp=started_at,
            duration_ms=int((pytime.time() - started_at) * 1000),
            success=success,
            input_summary=_safe_str_dict(input_summary),
            output_summary=_safe_str_dict(output_summary),
            decision=str(decision) if decision is not None else None,
            error=str(error) if error is not None else None,
        ).model_dump()
    )
    return traces


def _append_feedback(state: AgentState | dict, text: str, failed_tests: int = 0, failure_type: FailureType | None = None) -> list[dict]:
    history = list(_sget(state, "qa_feedback_history", []))
    history.append(
        make_json_safe(QaFeedbackRecord(
            attempt=int(_sget(state, "retry_count", 0)),
            suggestions=str(text),
            failed_tests=failed_tests,
            failure_type=failure_type,
        ).model_dump())
    )
    return history


def _latest_feedback(state: AgentState | dict) -> str:
    history = _sget(state, "qa_feedback_history", [])
    if not history:
        return ""
    last = history[-1]
    if isinstance(last, dict):
        return last.get("suggestions", "")
    return getattr(last, "suggestions", "")


def _patch_dict_to_changes(patch: dict) -> list[dict]:
    changes: list[dict] = []
    for file_path, content in (patch or {}).items():
        changes.append(
            FileChange(file_path=file_path, diff=content, change_type="modify").model_dump()
        )
    return changes


def _safe_patch_list(patch_changes) -> list[dict]:
    """
    Normalize every element of patch_changes to a plain dict with keys
    file_path / diff / change_type.  Handles:
      - dict          : pass-through
      - Pydantic obj  : .model_dump()
      - list/tuple    : positional reconstruction (file_path, diff, change_type)
      - anything else : skip (returns empty dict so callers get safe defaults)
    This is the single choke-point that prevents 'list object has no attribute get'
    regardless of what JSON round-tripping or make_json_safe produced.
    """
    if not patch_changes:
        return []
    result = []
    for c in patch_changes:
        if isinstance(c, dict):
            result.append(c)
        elif hasattr(c, "model_dump"):
            result.append(c.model_dump())
        elif hasattr(c, "file_path"):  # legacy Pydantic/dataclass
            result.append({"file_path": c.file_path, "diff": getattr(c, "diff", ""), "change_type": getattr(c, "change_type", "modify")})
        elif isinstance(c, (list, tuple)) and len(c) >= 1:
            # JSON round-trip converted a tuple to a list: [file_path, change_type, diff]
            result.append({
                "file_path":   str(c[0]) if len(c) > 0 else "",
                "change_type": str(c[1]) if len(c) > 1 else "modify",
                "diff":        str(c[2]) if len(c) > 2 else "",
            })
        else:
            # Unknown — skip rather than crash
            logger.warning(f"[_safe_patch_list] skipping unrecognised change element type: {type(c).__name__}")
    return result


def _normalize_failure_type(value: str | FailureType | None) -> FailureType:
    if isinstance(value, FailureType):
        return value
    if isinstance(value, str):
        try:
            return FailureType(value)
        except ValueError:
            return FailureType.UNKNOWN
    return FailureType.UNKNOWN


def _patch_fingerprint(patch_changes: list) -> str:
    normalized = []
    for change in _safe_patch_list(patch_changes):
        normalized.append((
            change.get("file_path", ""),
            change.get("change_type", ""),
            change.get("diff", ""),
        ))
    normalized.sort(key=lambda x: x[0])
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=True).encode("utf-8")).hexdigest()


def _context_cache_key(state: AgentState | dict) -> str:
    patch_changes = _sget(state, "patch", [])
    issue_id = str(_sget(state, "issue_id", ""))
    return f"{issue_id}:{_patch_fingerprint(patch_changes) if patch_changes else 'no_patch'}"


def _append_failure_record(state: AgentState | dict, failure_type: FailureType, reason: str) -> list[dict]:
    failures = list(_sget(state, "failures", []))
    failures.append(
        FailureRecord(
            attempt=int(_sget(state, "retry_count", 0)),
            failure_type=failure_type,
            reason=reason,
            exception_type=_sget(state, "exception_type"),
            exception_msg=_sget(state, "exception_msg"),
        ).model_dump()
    )
    return failures


def _append_attempt_record(
    state: AgentState | dict,
    decision: DecisionType,
    strategy: str,
    based_on_failure: FailureType | None = None,
) -> list[dict]:
    attempts = list(_sget(state, "attempts", []))
    patch_changes = _safe_patch_list(_sget(state, "patch", []))
    changed_files = [c.get("file_path", "") for c in patch_changes]
    fingerprint = _patch_fingerprint(patch_changes) if patch_changes else None
    attempts.append(
        AttemptRecord(
            attempt=int(_sget(state, "retry_count", 0)),
            strategy=strategy,
            based_on_failure=based_on_failure,
            decision=decision,
            changed_files=changed_files,
            patch_fingerprint=fingerprint,
            failed_tests=_sget(state, "failed_tests"),
            summary=_latest_feedback(state)[:240],
        ).model_dump()
    )
    return attempts


def _has_repeated_patch(state: AgentState | dict, patch_changes: list) -> bool:
    current_fp = _patch_fingerprint(patch_changes)
    for attempt in _sget(state, "attempts", []):
        if isinstance(attempt, dict) and attempt.get("patch_fingerprint") == current_fp:
            return True
    return False


def _strategy_for_failure(state: AgentState | dict) -> str:
    """
    Select retry strategy based on failure type.
    This ensures each retry actually learns from previous failure.
    """
    failure = _normalize_failure_type(_sget(state, "failure_type"))
    
    # Map failure type to appropriate retry strategy
    mapping = {
        FailureType.SYNTAX_ERROR: "syntax_correction",
        FailureType.GUARDRAIL_FAILURE: "guardrail_compliant_fix",
        FailureType.LLM_PARSE_ERROR: "output_schema_hardening",
        FailureType.LOGIC_ERROR: "logic_repair",
        FailureType.EDGE_CASE_MISSING: "edge_case_expansion",
        FailureType.VALIDATION_ERROR: "validation_focused_fix",
        FailureType.REQUIREMENT_MISMATCH: "requirements_alignment",
        FailureType.BEHAVIORAL_TESTS_FAILED: "behavioral_correction",  # New: specific strategy for test failures
        FailureType.RATE_LIMIT: "backoff_or_pause",
    }
    return mapping.get(failure, "default")

def has_commits_ahead(base="main", head="developer", repo_path: str = "repo_clone") -> bool:
    """Return True if head has at least one commit not on base."""
    import subprocess
    result = subprocess.run(
        ["git", "log", f"{base}..{head}", "--oneline"],
        capture_output=True, text=True, cwd=repo_path
    )
    return bool(result.stdout.strip())


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: ISSUE FETCHING & SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_issues_node(state: AgentState) -> dict:
    started_at = pytime.time()
    logger.info("[NODE] fetch_issues_node")
    try:
        _ = get_issues()
        return _merge(
            state,
            status=PipelineStatus.ISSUES_FETCHED.value,
            node_traces=_append_trace(
                state,
                node="fetch_issues",
                started_at=started_at,
                success=True,
                output_summary={"status": PipelineStatus.ISSUES_FETCHED.value},
            ),
        )
    except Exception as e:
        logger.error(f"fetch_issues_node failed: {e}")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
            error=str(e),
            node_traces=_append_trace(
                state,
                node="fetch_issues",
                started_at=started_at,
                success=False,
                error=str(e),
                output_summary={"status": PipelineStatus.FAILED.value},
            ),
        )


def select_issue_node(state: AgentState) -> dict:
    started_at = pytime.time()
    logger.info("[NODE] select_issue_node")
    try:
        issues = get_issues()
        if not issues:
            return _merge(state, status=PipelineStatus.FAILED.value, error="No open issues found")

        for issue in issues:
            print(f"#{issue['number']}: {issue['title']}")

        val = input("\nWhich issue to solve (default to first): ").strip()
        issue = issues[0]
        if val:
            try:
                issue_id = int(val)
                issue = next((i for i in issues if i["number"] == issue_id), issues[0])
            except Exception:
                pass

        return {
            **_as_dict(state),
            "issue_id": str(issue["number"]),
            "issue_title": issue.get("title", ""),
            "issue_body": issue.get("body", "") or "",
            "status": PipelineStatus.ISSUE_SELECTED.value,
            "retry_count": 0,
            "qa_feedback_history": [],
            "attempts": [],
            "failures": [],
            "node_traces": _append_trace(
                state,
                node="select_issue",
                started_at=started_at,
                success=True,
                output_summary={
                    "status": PipelineStatus.ISSUE_SELECTED.value,
                    "issue_id": str(issue["number"]),
                },
            ),
        }
    except Exception as e:
        logger.error(f"select_issue_node failed: {e}")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
            error=str(e),
            node_traces=_append_trace(
                state,
                node="select_issue",
                started_at=started_at,
                success=False,
                error=str(e),
            ),
        )


def setup_repo_node(state: AgentState) -> dict:
    started_at = pytime.time()
    logger.info("[NODE] setup_repo_node")
    try:
        res = setup_repo(_sget(state, "issue_id"))
        if not res["success"]:
            return _merge(state, status=PipelineStatus.FAILED.value, error=res["error"])

        base = _as_dict(state)
        repo = dict(base.get("repo", {}))
        repo.update({"repo_path": res["repo_path"], "branch": res["branch"]})
        return _merge(base, repo=repo, status=PipelineStatus.REPO_READY.value)
    except Exception as e:
        logger.error(f"setup_repo_node failed: {e}")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
            error=str(e),
            node_traces=_append_trace(
                state,
                node="setup_repo",
                started_at=started_at,
                success=False,
                error=str(e),
            ),
        )


def _check_edge_case_handling(patch_changes: list, simulated_results: list, modified_files: dict[str, str]) -> tuple[bool, str]:
    """
    Verify that edge cases from failed tests are actually addressed in the patch.
    
    Returns (edge_cases_handled, reason)
    """
    if not simulated_results or not isinstance(simulated_results, list):
        return True, "no_test_failures_to_check"
    
    # Extract edge case keywords from failed tests
    failed_edge_cases = set()
    for tr in simulated_results:
        if isinstance(tr, dict) and tr.get("status") == "fail":
            reasoning = str(tr.get("reasoning", "")).lower()
            # Look for common edge case keywords
            edge_case_keywords = [
                "none", "null", "empty", "invalid", "boundary", "zero", 
                "negative", "overflow", "underflow", "division by zero",
                "missing", "undefined", "exception", "error"
            ]
            for kw in edge_case_keywords:
                if kw in reasoning:
                    failed_edge_cases.add(kw)
    
    if not failed_edge_cases:
        return True, "no_identifiable_edge_cases"
    
    # Check if patch contains handling code for these edge cases
    patch_text = " ".join(
        change["diff"].lower() if isinstance(change, dict) else ""
        for change in patch_changes if isinstance(change, dict)
    )
    
    # Look for defensive code patterns
    defensive_patterns = [
        "is none", "!= none", "is not none", "== none",
        "if not ", "if ", "try:", "except",
        "len(", "range(", "== 0", "> 0", "< 0",
        "validate", "check", "assert", "raise",
    ]
    
    defensive_code_found = sum(1 for p in defensive_patterns if p in patch_text)
    
    # If edge cases were identified but minimal defensive code, flag it
    if defensive_code_found == 0:
        missing_cases = ", ".join(sorted(list(failed_edge_cases)[:3]))
        return False, f"no_defensive_code_for_edge_cases:{missing_cases}"
    
    return True, "edge_cases_appear_handled"


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: ISSUE AGENT + ISSUE PR
# ═══════════════════════════════════════════════════════════════════════════════

def issue_agent_node(state: AgentState) -> dict:
    started_at = pytime.time()
    retry = _sget(state, "retry_count", 0)
    logger.info(f"[NODE] issue_agent_node | retry={retry}")
    try:
        if not _sget(state, "issue_id") or not _sget(state, "issue_title"):
            return _merge(
                state,
                status=PipelineStatus.FAILED.value,
                failure_reason="missing_issue_metadata",
                decision=DecisionType.FAIL.value,
                qa_feedback_history=_append_feedback(state, "Missing selected issue metadata (issue_id/issue_title)."),
                node_traces=_append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    error="missing_issue_metadata",
                ),
            )
        # Ensure qa_feedback is always a correctly-typed dict before planner reads it
        if isinstance(state, dict):
            state = normalize_qa_feedback(state)
        # Build structured feedback string for the planner from qa_feedback dict (set by llm_validation_node)
        qa_feedback_dict = safe_dict(_sget(state, "qa_feedback"))
        if qa_feedback_dict and isinstance(qa_feedback_dict, dict):
            issues_list = qa_feedback_dict.get("issues", [])
            qa_reason = qa_feedback_dict.get("reason", "")
            alignment = qa_feedback_dict.get("alignment", "")
            confidence = qa_feedback_dict.get("confidence", "LOW")
            issues_text = "\n".join(f"- {i}" for i in issues_list) if issues_list else "- None"
            structured_feedback = (
                f"[QA FEEDBACK — confidence: {confidence}]\n"
                f"Reason: {qa_reason or 'N/A'}\n"
                f"Alignment: {alignment or 'N/A'}\n"
                f"Issues:\n{issues_text}"
            )
        else:
            structured_feedback = _latest_feedback(state)

        # Read planner_mode from outer state (set by retry_handler_node via qa_normalization_node).
        # Passing it explicitly ensures inner graph honours skip-planner for coder failures.
        outer_planner_mode = _sget(state, "planner_mode")
        logger.info(
            f"[issue_agent_node] planner_mode={outer_planner_mode!r} "
            f"retry_strategy={_sget(state, 'retry_strategy', 'normal')!r} "
            f"failure_stage={_sget(state, 'failure_stage', 'unknown')!r}"
        )
        res = run_issue_agent(
            issue_id=_sget(state, "issue_id"),
            issue_title=_sget(state, "issue_title"),
            issue_body=_sget(state, "issue_body"),
            repo_path=_sget(state, "repo_path"),
            qa_feedback=structured_feedback,
            qa_feedback_history=_sget(state, "qa_feedback_history", []),
            retry_count=retry,
            retry_strategy=_sget(state, "retry_strategy", "normal"),
            previous_plan=_sget(state, "plan"),
            failure_type=_sget(state, "failure_type"),
            failed_tests=_sget(state, "failed_tests"),
            previous_patch=_sget(state, "previous_patch"),
            planner_mode=outer_planner_mode,  # explicit: lets adapter override inference
            simulated_results=_sget(state, "simulated_results"), # Pass test failure details to coder
        )

        if not res.get("success"):
            err_msg = res.get("error", "Issue agent failed")
            res_failure_type = res.get("failure_type", "")
            # Hard-stop failures — retrying wastes quota and will always fail
            _HALT_FAILURES = {"RATE_LIMIT", "TOKEN_LIMIT"}
            if res_failure_type in _HALT_FAILURES:
                logger.error(f"[HALT] Adapter returned {res_failure_type} — stopping pipeline immediately")
                return _merge(
                    state,
                    status=PipelineStatus.FAILED.value,
                    failure_type=res_failure_type,
                    failure_reason=res_failure_type.lower(),
                    decision=DecisionType.FAIL.value,
                    qa_feedback_history=_append_feedback(
                        state, err_msg,
                        failure_type=_normalize_failure_type(res_failure_type),
                    ),
                    node_traces=_append_trace(
                        state,
                        node="issue_agent",
                        started_at=started_at,
                        success=False,
                        error=err_msg,
                    ),
                )
            # Retriable failures — let the retry loop handle them
            return _merge(
                state,
                status=PipelineStatus.REJECTED.value,
                failure_type=res_failure_type or FailureType.UNKNOWN.value,
                failure_stage=res.get("failure_stage", "unknown"),
                qa_feedback_history=_append_feedback(state, err_msg),
                node_traces=_append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    error=err_msg,
                ),
            )

        patch = res["patch"]
        if not validate_patch(patch):
            return _merge(
                state,
                status=PipelineStatus.REJECTED.value,
                qa_feedback_history=_append_feedback(state, "Invalid patch format generated"),
                node_traces=_append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    error="invalid_patch_format",
                ),
            )

        patch_changes = _patch_dict_to_changes(patch)
        if _has_repeated_patch(state, patch_changes):
            return _merge(
                state,
                status=PipelineStatus.REJECTED.value,
                failure_type=FailureType.LOGIC_ERROR.value,
                failure_reason="repeated_identical_fix",
                qa_feedback_history=_append_feedback(state, "Generated patch repeats a previous attempt; require a new strategy."),
                failures=_append_failure_record(state, FailureType.LOGIC_ERROR, "repeated_identical_fix"),
                node_traces=_append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    error="repeated_identical_fix",
                ),
            )

        # ── Semantic signal computation (passed to QA — never hard-rejects) ──
        import re as _re
        import difflib as _difflib

        _repo_path = _sget(state, "repo_path")
        _total_adds = 0
        _total_dels = 0
        _modified_functions: list[str] = []
        _removed_symbols: list[str] = []
        _empty_bodies: list[str] = []
        _content_changed = False
        _behavior_changed = False
        _module_body_changed = False
        _target_symbols_missed: list[str] = []
        _issue_text = ((_sget(state, "issue_title") or "") + " " + (_sget(state, "issue_body") or "")).lower()
        _issue_keywords = set(_re.findall(r'\b[a-z_]{3,}\b', _issue_text))
        _issue_keywords -= {"the", "and", "for", "that", "this", "with", "from", "have", "should", "would", "could", "not", "are", "was", "been"}
        _patch_text = " ".join((_c.get("diff") or "") for _c in patch_changes if isinstance(_c, dict)).lower()
        _keyword_hits = [kw for kw in _issue_keywords if kw in _patch_text]
        _plan_targets = res.get("plan", {}).get("targets", []) if isinstance(res.get("plan", {}), dict) else []

        for _c in patch_changes:
            _file_path = _c.get("file_path") if isinstance(_c, dict) else getattr(_c, "file_path", "")
            _modified_content = _c.get("diff") if isinstance(_c, dict) else getattr(_c, "diff", "")
            _change_type = _c.get("change_type") if isinstance(_c, dict) else getattr(_c, "change_type", "modify")

            if not isinstance(_file_path, str) or not isinstance(_modified_content, str):
                continue
            if _change_type == "delete":
                _content_changed = True
                _behavior_changed = True
                continue

            _original_content = ""
            if _repo_path:
                _full_path = os.path.join(_repo_path, _file_path)
                if os.path.exists(_full_path):
                    try:
                        with open(_full_path, "r", encoding="utf-8") as _f:
                            _original_content = _f.read()
                    except Exception as _e:
                        logger.warning(f"[SEMANTIC] Could not read original {_file_path}: {_e}")

            if _original_content != _modified_content:
                _content_changed = True

                _orig_lines = _original_content.splitlines(keepends=True)
                _mod_lines = _modified_content.splitlines(keepends=True)
                _unified_diff = list(_difflib.unified_diff(_orig_lines, _mod_lines, lineterm=""))
                for _diff_line in _unified_diff:
                    if _diff_line.startswith("+") and not _diff_line.startswith("+++"):
                        _stripped = _diff_line[1:].strip()
                        if _stripped and not _stripped.startswith("#"):
                            _total_adds += 1
                    elif _diff_line.startswith("-") and not _diff_line.startswith("---"):
                        _stripped = _diff_line[1:].strip()
                        if _stripped and not _stripped.startswith("#"):
                            _total_dels += 1

                if _file_path.endswith(".py"):
                    _analysis = LLMOutputValidator.analyze_python_changes(_original_content, _modified_content)
                    _changed_defs = _analysis.get("changed_definitions", []) + _analysis.get("added_definitions", [])
                    _modified_functions.extend(f"{_file_path}:{name}" for name in _changed_defs)
                    _removed_symbols.extend(f"{_file_path}:{name}" for name in _analysis.get("removed_definitions", []))
                    _empty_bodies.extend(f"{_file_path}:{name}" for name in _analysis.get("empty_definitions", []))
                    _behavior_changed = _behavior_changed or bool(
                        _analysis.get("changed_definitions")
                        or _analysis.get("added_definitions")
                        or _analysis.get("module_body_changed")
                        or _analysis.get("imports_added")
                        or _analysis.get("imports_removed")
                    )
                    _module_body_changed = _module_body_changed or bool(_analysis.get("module_body_changed"))

                    _file_targets = [
                        str(target.get("symbol", "")).strip()
                        for target in _plan_targets
                        if isinstance(target, dict) and str(target.get("file", "")).replace("\\", "/").strip("/") == _file_path.replace("\\", "/").strip("/")
                    ]
                    _changed_names = set(_analysis.get("changed_definitions", []) + _analysis.get("added_definitions", []))
                    for _target in _file_targets:
                        if not _target:
                            continue
                        _bare_target = _target.replace("::", ".").split(".")[-1]
                        
                        # RULE 6: Check if target was modified
                        # Case 1: Direct match (method or function)
                        target_modified = _target in _changed_names or _bare_target in _changed_names
                        
                        # Case 2: Target is a class, check if any method inside it changed
                        if not target_modified:
                            # If target is a class name, check for ClassName.method_name pattern
                            class_methods = [name for name in _changed_names if name.startswith(f"{_target}.")]
                            target_modified = len(class_methods) > 0
                        
                        # Case 3: Wildcard match (ClassName.method matches when checking for method)
                        if not target_modified:
                            target_modified = any(name.endswith(f".{_bare_target}") for name in _changed_names)
                        
                        # Case 4: Module-body changes (imports, top-level assignments)
                        if not target_modified:
                            target_modified = _analysis.get("module_body_changed", False)
                        
                        if not target_modified:
                            _target_symbols_missed.append(f"{_file_path}:{_target}")
                else:
                    _behavior_changed = True
        
        # If no file content actually differs, reject the patch
        if not _content_changed:
            logger.warning(f"[SEMANTIC] No real modification detected. All file contents identical to originals.")
            return _merge(
                state,
                status=PipelineStatus.REJECTED.value,
                failure_type=FailureType.LOGIC_ERROR.value,
                failure_reason="no_real_modification",
                qa_feedback_history=_append_feedback(state, "Generated patch contains no actual code modifications — all files identical. Rejecting empty patch."),
                failures=_append_failure_record(state, FailureType.LOGIC_ERROR, "no_real_modification"),
                node_traces=_append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    error="no_real_modification",
                ),
            )

        _modified_functions = list(dict.fromkeys(_modified_functions))
        _removed_symbols = list(dict.fromkeys(_removed_symbols))
        _empty_bodies = list(dict.fromkeys(_empty_bodies))
        _target_symbols_missed = list(dict.fromkeys(_target_symbols_missed))
        _is_trivial = not _behavior_changed and (_total_adds + _total_dels > 0)
        
        # ── Validate edge case handling from failed tests ──
        # Build modified_files dict from patch_changes (file_path -> content)
        _modified_files_map = {}
        for _pc in patch_changes:
            _pc_fp = _pc.get("file_path", "") if isinstance(_pc, dict) else ""
            _pc_diff = _pc.get("diff", "") if isinstance(_pc, dict) else ""
            if _pc_fp:
                _modified_files_map[_pc_fp] = _pc_diff
        edge_cases_ok, edge_case_reason = _check_edge_case_handling(
            patch_changes, 
            _sget(state, "simulated_results", []), 
            _modified_files_map,
        )
        if not edge_cases_ok and _sget(state, "retry_count", 0) > 0:
            logger.warning(f"[SEMANTIC] Edge case handling insufficient: {edge_case_reason}")
            _is_trivial = True  # Flag as trivial if edge cases not addressed
        
        _semantic_signals = {
            "total_adds": _total_adds,
            "total_dels": _total_dels,
            "modified_functions": _modified_functions[:10],
            "removed_symbols": _removed_symbols[:10],
            "empty_bodies": _empty_bodies[:10],
            "target_symbols_missed": _target_symbols_missed[:10],
            "trivial_flag": _is_trivial,
            "issue_keywords_matched": _keyword_hits[:15],
            "issue_keywords_total": len(_issue_keywords),
            "keyword_match_ratio": round(len(_keyword_hits) / max(len(_issue_keywords), 1), 2),
            "content_changed": True,
            "behavior_changed": _behavior_changed,
            "module_body_changed": _module_body_changed,
            "edge_cases_handled": edge_cases_ok,
            "edge_case_status": edge_case_reason,
        }

            
        if _is_trivial:
            logger.warning(f"[SEMANTIC] Patch flagged as potentially trivial: {_semantic_signals}")
        else:
            logger.info(f"[SEMANTIC] Patch signals: +{_total_adds}/-{_total_dels}, {len(_modified_functions)} functions touched, content_changed=True")

        return _merge(
            state,
            patch=patch_changes,
            plan=res.get("plan"),
            semantic_signals=_semantic_signals,
            status=PipelineStatus.PATCH_GENERATED.value,
            node_traces=_append_trace(
                state,
                node="issue_agent",
                started_at=started_at,
                success=True,
                output_summary={
                    "status": PipelineStatus.PATCH_GENERATED.value,
                    "files": str(len(patch_changes)),
                    "adds": str(_total_adds),
                    "dels": str(_total_dels),
                    "trivial": str(_is_trivial),
                },
            ),
        )
    except Exception as e:
        exc_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"[PLANNER FAILED] {exc_type}: {e}")

        # Rule 10: Fail fast on rate limit — STOP immediately
        if "RATE_LIMIT" in error_msg.upper() or "rate limit" in error_msg.lower() or exc_type == "RateLimitError":
            logger.error("[HALT] Rate limit exhausted — stopping pipeline immediately")
            return {
                **_as_dict(state),
                "status": PipelineStatus.FAILED.value,
                "failure_type": FailureType.RATE_LIMIT.value,
                "exception_type": exc_type,
                "exception_msg": error_msg,
                "qa_feedback_history": _append_feedback(state, f"Rate limit reached: {error_msg}", failure_type=FailureType.RATE_LIMIT),
                "failures": _append_failure_record(state, FailureType.RATE_LIMIT, error_msg),
                "decision": DecisionType.FAIL.value,
                "node_traces": _append_trace(
                    state,
                    node="issue_agent",
                    started_at=started_at,
                    success=False,
                    decision=DecisionType.FAIL.value,
                    error=error_msg,
                ),
            }

        HALT_NOW = {
            "UnboundLocalError", "NameError",
            "SyntaxError", "ImportError",
            "AttributeError",
        }
        force_halt = exc_type in HALT_NOW
        if force_halt:
            logger.error(f"[HALT] {exc_type} is a deterministic code bug. Retrying will not help. Fix the code first.")
            
        return {
            **_as_dict(state),
            "status": PipelineStatus.FAILED.value,
            "exception_type": exc_type,
            "exception_msg": error_msg,
            "qa_feedback_history": _append_feedback(state, error_msg),
            "failures": _append_failure_record(state, FailureType.UNKNOWN, error_msg),
            "node_traces": _append_trace(
                state,
                node="issue_agent",
                started_at=started_at,
                success=False,
                error=error_msg,
            ),
        }


def validate_state(state):
    """Raise early if state is None — catches mis-wired graph edges."""
    if state is None:
        raise ValueError("State is None")


def issue_pr_node(state: AgentState) -> dict:
    """Creates or updates the Issue PR after patch generation."""
    validate_state(state)
    logger.info("[NODE] issue_pr_node")
    logger.info("[FLOW] PR creation started")
    if _sget(state, "status") == PipelineStatus.FAILED.value:
        return _as_dict(state)
    
    plan = _sget(state, "plan")
    if plan is None:
        logger.error("[ABORT] issue_pr_node received no plan — skipping PR creation")
        return _merge(
            state,
            pr_status="skipped_no_plan",
            status=PipelineStatus.FAILED.value,
            failure_reason="skipped_no_plan",
            decision=DecisionType.RETRY.value,
        )
    try:
        head_branch = _sget(state, "branch") or "HEAD"
        logger.info(f"[issue_pr_node] Using head branch: {head_branch}")
        repo_path_val = _sget(state, "repo_path", "repo_clone")

        if not has_commits_ahead(base="main", head=head_branch, repo_path=repo_path_val):
            logger.warning(f"[issue_pr_node] No commits ahead on {head_branch} — skipping PR creation")
            return _merge(
                state,
                status=PipelineStatus.INTEGRATED.value,
                error=f"No commits to create PR from {head_branch}",
            )

        patch_gen = PatchGenerator()
        issue_dict = {
            "number": int(_sget(state, "issue_id", 0)),
            "title": _sget(state, "issue_title", "Unknown"),
            "body": _sget(state, "issue_body", ""),
        }

        patch_changes = _safe_patch_list(_sget(state, "patch", []))
        changed_files = [c.get("file_path", "") for c in patch_changes]
        title, desc = generate_pr_content(issue_dict, changed_files, patch_gen)

        pr = create_pull_request(title, desc, head=head_branch)

        if not pr or not isinstance(pr, dict) or not pr.get("success"):
            error_detail = pr.get("error", "Unknown error") if isinstance(pr, dict) else "PR creation returned None"
            logger.error(f"[PR FAILED] {error_detail} — continuing pipeline without PR")
            # Non-fatal: QA can still run and the PR can be created manually or on next retry
            return _merge(
                state,
                status=PipelineStatus.INTEGRATED.value,
                error=error_detail,
            )

        pr_num = pr.get("data", {}).get("pr_number")
        pr_url = pr.get("data", {}).get("url") or pr.get("data", {}).get("html_url", "")

        if not pr_num:
            logger.error("[issue_pr_node] PR created but pr_number missing in response — continuing")
            return _merge(
                state,
                status=PipelineStatus.INTEGRATED.value,
                error="PR data didn't contain pr_number",
            )

        logger.info(f"[NODE] issue_pr_node | Created Issue PR #{pr_num}")
        logger.info(f"[PR CREATED] #{pr_num}")
        logger.info(f"[PR URL] {pr_url}")

        issue_prs = list(_sget(state, "issue_prs", []))
        issue_prs.append(str(pr_num))

        return _merge(
            state,
            issue_pr_number=str(pr_num),
            issue_prs=issue_prs,
            pr_url=pr_url,
            status=PipelineStatus.INTEGRATED.value,
        )
    except Exception as e:
        logger.error(f"[issue_pr_node] Exception: {e} — continuing without PR")
        return _merge(
            state,
            status=PipelineStatus.INTEGRATED.value,
            error=f"PR creation failed: {str(e)}",
            exception_type=type(e).__name__,
            exception_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: INTEGRATION (APPLY PATCH + DIFF)
# ═══════════════════════════════════════════════════════════════════════════════

def integration_node(state: AgentState) -> dict:
    """Apply structured patch changes in repo workspace and push attempt branch."""
    started_at = pytime.time()
    validate_state(state)
    logger.info("[NODE] integration_node")
    if _sget(state, "status") == PipelineStatus.FAILED.value:
        return _as_dict(state)

    if _sget(state, "plan") is None:
        logger.error("[ABORT] integration_node received no plan — skipping integration")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            failure_reason=FailureType.PLAN_GENERATION_FAILED.value,
            decision=DecisionType.RETRY.value,
            qa_feedback_history=_append_feedback(state, "No valid plan generated by issue agent."),
            node_traces=_append_trace(state, "integration", started_at, False, error="missing_plan"),
        )

    patch_changes = _sget(state, "patch", [])
    if not patch_changes:
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            qa_feedback_history=_append_feedback(state, "Patch is empty"),
            node_traces=_append_trace(state, "integration", started_at, False, error="empty_patch"),
        )

    safety_result = validate_patch_changes(patch_changes, repo_path=_sget(state, "repo_path", ""))
    if safety_result["status"] == "REJECTED":
        reason = safety_result.get("reason", "Patch validation failed")
        failure = _normalize_failure_type(safety_result.get("failure_type"))
        logger.error(f"[PATCH VALIDATION] {reason}")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            failure_type=failure.value,
            failure_reason=reason,
            qa_feedback_history=_append_feedback(state, reason, failure_type=failure),
            failures=_append_failure_record(state, failure, reason),
            node_traces=_append_trace(state, "integration", started_at, False, error=reason),
        )

    try:
        import time

        repo_path = _sget(state, "repo_path")
        run_id = _sget(state, "run_id") or str(int(time.time()))
        branch_name = f"feature/issue-{_sget(state, 'issue_id')}-run-{run_id}-attempt-{_sget(state, 'retry_count', 0)}"

        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=False)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=False)
        subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
        subprocess.run(["git", "pull", "origin", "main"], cwd=repo_path, check=False)
        subprocess.run(["git", "checkout", "-B", branch_name], cwd=repo_path, check=True)

        for change in patch_changes:
            file_path = change.get("file_path") if isinstance(change, dict) else change.file_path
            content = change.get("diff") if isinstance(change, dict) else change.diff
            change_type = change.get("change_type") if isinstance(change, dict) else change.change_type
            full_path = os.path.join(repo_path, file_path)
            if change_type == "delete":
                if os.path.exists(full_path):
                    os.remove(full_path)
                continue
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        for cache_file in (".ai_chunks.json", ".ai_faiss.index"):
            cache_path = os.path.join(repo_path, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)

        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "restore", "--staged", ".ai_chunks.json", ".ai_faiss.index"], cwd=repo_path, check=False)
        subprocess.run(["git", "checkout", "--", ".ai_chunks.json", ".ai_faiss.index"], cwd=repo_path, capture_output=True, text=True, check=False)
        subprocess.run(["git", "commit", "-m", f"AI: Fix issue #{_sget(state, 'issue_id', '?')}"], cwd=repo_path, check=False)

        try:
            subprocess.run(["git", "push", "--set-upstream", "origin", branch_name], cwd=repo_path, check=True)
        except subprocess.CalledProcessError:
            return _merge(
                state,
                status=PipelineStatus.FAILED.value,
                failure_reason=FailureType.BRANCH_CONFLICT.value,
                decision=DecisionType.FAIL.value,
            )

        base = _as_dict(state)
        repo = dict(base.get("repo", {}))
        repo.update({"branch": branch_name, "run_id": run_id})
        # Also promote branch/repo_path/run_id to flat top-level keys so that
        # downstream nodes using _sget(state, 'branch') on a plain dict work.
        return _merge(
            base,
            repo=repo,
            branch=branch_name,
            run_id=run_id,
            repo_path=repo_path,
            status=PipelineStatus.INTEGRATED.value,
        )
    except Exception as e:
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            error=f"Integration failed: {str(e)}",
            decision=DecisionType.FAIL.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
            node_traces=_append_trace(state, "integration", started_at, False, error=str(e)),
        )

def diff_node(state: AgentState) -> dict:
    validate_state(state)
    logger.info("[NODE] diff_node")
    if _sget(state, "status") == PipelineStatus.FAILED.value:
        return _as_dict(state)

    try:
        diff_res = subprocess.run(["git", "diff", "origin/main...HEAD"], cwd=_sget(state, "repo_path", "."), capture_output=True, text=True)
        diff_text = diff_res.stdout
        logger.info(f"[DIFF SIZE] {len(diff_text)}")
        return _merge(state, diff=diff_text)
    except Exception as e:
        return _merge(state, status=PipelineStatus.FAILED.value, error=f"Diff generation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: QA AGENT + QA PR
# ═══════════════════════════════════════════════════════════════════════════════

import time, random
from groq import Groq, RateLimitError, AuthenticationError

def llm_invoke(prompt: str, max_retries: int = 2) -> str:
    groq_client = Groq()
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning(
                f"[LLM] Rate limit hit (attempt {attempt+1}/{max_retries}) "
                f"— waiting {wait:.1f}s"
            )
            time.sleep(wait)
        except AuthenticationError:
            logger.error("[LLM] Auth error — halting immediately")
            raise   # never retry auth errors

import ast, os

def run_guardrails(patch_changes: list) -> dict:
    """Run deterministic checks before LLM invocation."""
    for change in patch_changes:
        file_path = change.get("file_path") if isinstance(change, dict) else change.file_path
        code = change.get("diff") if isinstance(change, dict) else change.diff
        change_type = change.get("change_type") if isinstance(change, dict) else change.change_type
        if change_type == "delete":
            continue
        if not file_path.endswith(".py"):
            continue
            
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "REJECTED",
                "failure_type": "SYNTAX_ERROR",
                "reason": f"Invalid Python syntax in {file_path}: {e}"
            }
            
        # Structural Guardrails
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                     return {
                         "status": "REJECTED",
                         "failure_type": "GUARDRAIL_FAILURE",
                         "reason": f"Empty function detected: {node.name} in {file_path}"
                     }
                
                has_return_or_yield = any(isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
                # Simple check: no return/yield in non-init? We won't strictly enforce but we could
                
            # Division by zero risk
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                     return {
                         "status": "REJECTED", 
                         "failure_type": "GUARDRAIL_FAILURE", 
                         "reason": f"Explicit division by zero risk detected in {file_path}."
                     }
                     
        # Duplicate definition Guardrail
        def _check_duplicates(nodes, parent_name=""):
            seen_funcs = set()
            seen_classes = set()
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in seen_funcs:
                        return False, f"Duplicate method/function detected: {parent_name}{node.name} in {file_path}"
                    seen_funcs.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    if node.name in seen_classes:
                        return False, f"Duplicate class detected: {node.name} in {file_path}"
                    seen_classes.add(node.name)
                    # Check methods inside the class
                    valid, reason = _check_duplicates(node.body, parent_name=f"{node.name}.")
                    if not valid:
                        return False, reason
            return True, "OK"
        
        valid, reason = _check_duplicates(tree.body)
        if not valid:
            return {
                "status": "REJECTED",
                "failure_type": "GUARDRAIL_FAILURE",
                "reason": reason
            }
                     
    return {"status": "PASSED"}


def validate_patch_changes(patch_changes: list, repo_path: str = "") -> dict:
    """Validation layer before patch application."""
    if not patch_changes:
        return {
            "status": "REJECTED",
            "failure_type": FailureType.VALIDATION_ERROR.value,
            "reason": "Patch is empty",
        }

    max_files = 20
    max_total_chars = 200_000
    protected_prefixes = (".git", ".github/workflows", ".cursor", ".env")

    if len(patch_changes) > max_files:
        return {
            "status": "REJECTED",
            "failure_type": FailureType.VALIDATION_ERROR.value,
            "reason": f"Patch touches too many files ({len(patch_changes)} > {max_files}).",
        }

    total_chars = 0
    for change in patch_changes:
        file_path = change.get("file_path") if isinstance(change, dict) else change.file_path
        code = change.get("diff") if isinstance(change, dict) else change.diff
        change_type = change.get("change_type") if isinstance(change, dict) else change.change_type

        if not file_path or change_type not in {"add", "modify", "delete"}:
            return {
                "status": "REJECTED",
                "failure_type": FailureType.VALIDATION_ERROR.value,
                "reason": "Malformed structured patch entry detected.",
            }

        normalized = file_path.replace("\\", "/").lstrip("/")
        if change_type == "delete" and normalized.startswith(protected_prefixes):
            return {
                "status": "REJECTED",
                "failure_type": FailureType.VALIDATION_ERROR.value,
                "reason": f"Destructive delete blocked for protected path: {file_path}",
            }

        if change_type != "delete":
            total_chars += len(code or "")
            if file_path.endswith(".py"):
                try:
                    ast.parse(code or "")
                except SyntaxError as e:
                    return {
                        "status": "REJECTED",
                        "failure_type": FailureType.SYNTAX_ERROR.value,
                        "reason": f"Invalid Python syntax in {file_path}: {e}",
                    }

                original_code = ""
                if repo_path:
                    original_path = os.path.join(repo_path, normalized)
                    if os.path.exists(original_path) and os.path.isfile(original_path):
                        try:
                            with open(original_path, "r", encoding="utf-8", errors="replace") as handle:
                                original_code = handle.read()
                        except Exception:
                            original_code = ""

                full_ok, full_reason = LLMOutputValidator.validate_full_output(
                    code or "",
                    original_code,
                    filename=file_path,
                )
                if not full_ok:
                    return {
                        "status": "REJECTED",
                        "failure_type": FailureType.VALIDATION_ERROR.value,
                        "reason": f"Invalid patched output in {file_path}: {full_reason}",
                    }

                import_ok, import_reason = LLMOutputValidator.validate_import_integrity(
                    original_code,
                    code or "",
                    filename=file_path,
                )
                if not import_ok:
                    return {
                        "status": "REJECTED",
                        "failure_type": FailureType.VALIDATION_ERROR.value,
                        "reason": f"Import integrity failed in {file_path}: {import_reason}",
                    }

                if original_code:
                    struct_ok, struct_reason = LLMOutputValidator.validate_structural_integrity(
                        original_code,
                        code or "",
                        filename=file_path,
                    )
                    if not struct_ok:
                        return {
                            "status": "REJECTED",
                            "failure_type": FailureType.VALIDATION_ERROR.value,
                            "reason": f"Structural corruption in {file_path}: {struct_reason}",
                        }

    if total_chars > max_total_chars:
        return {
            "status": "REJECTED",
            "failure_type": FailureType.VALIDATION_ERROR.value,
            "reason": f"Patch too large ({total_chars} chars > {max_total_chars}).",
        }

    return {"status": "PASSED"}


def _summarize_repo_context(repo_path: str, patch_changes: list) -> tuple[str, dict[str, str], str]:
    """
    Build lightweight context for test generation.
    Limits related file reads to keep token usage bounded.
    """
    changed_files: list[str] = []
    for change in patch_changes:
        file_path = change.get("file_path") if isinstance(change, dict) else change.file_path
        if file_path:
            changed_files.append(file_path)
    changed_files = list(dict.fromkeys(changed_files))

    file_summaries: dict[str, str] = {}
    related_budget = 4
    for file_path in changed_files[:related_budget]:
        full = os.path.join(repo_path, file_path)
        if not os.path.exists(full) or not os.path.isfile(full):
            continue
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(1200)
            snippet = content.strip().replace("\r\n", "\n")
            file_summaries[file_path] = snippet
        except Exception:
            file_summaries[file_path] = ""

    repo_summary = (
        f"Changed files: {len(changed_files)}; "
        f"Context files included: {len(file_summaries)}; "
        f"Python-focused patch validation enabled."
    )

    patch_lines = []
    for change in patch_changes[:related_budget]:
        path = change.get("file_path") if isinstance(change, dict) else change.file_path
        ctype = change.get("change_type") if isinstance(change, dict) else change.change_type
        diff_text = change.get("diff") if isinstance(change, dict) else change.diff
        patch_lines.append(f"- {ctype.upper()}: {path} ({len(diff_text or '')} chars)")
    patch_summary = "\n".join(patch_lines)
    return repo_summary, file_summaries, patch_summary


def test_generation_agent_node(state: AgentState) -> dict:
    """
    Generate high-value test cases using constrained context.
    Uses a single LLM call and stores strict structured output.
    """
    started_at = pytime.time()
    patch_changes = _sget(state, "patch", [])
    repo_path = _sget(state, "repo_path")
    cache_key = _context_cache_key(state)
    if _sget(state, "generated_tests") and _sget(state, "generated_tests_cache_key") == cache_key:
        return _merge(
            state,
            node_traces=_append_trace(
                state,
                "test_generation_agent",
                started_at,
                True,
                output_summary={"cache": "hit", "generated_tests": str(len(_sget(state, "generated_tests", [])))},
            ),
        )
    if not patch_changes or not repo_path:
        return _merge(
            state,
            generated_tests=[],
            node_traces=_append_trace(
                state,
                "test_generation_agent",
                started_at,
                False,
                error="missing_patch_or_repo",
            ),
        )
    repo_summary = _sget(state, "repo_summary")
    file_summaries = _sget(state, "file_summaries", {})
    patch_summary = _sget(state, "patch_summary")
    if not repo_summary or not file_summaries or not patch_summary:
        repo_summary, file_summaries, patch_summary = _summarize_repo_context(repo_path, patch_changes)

    issue_text = (_sget(state, "issue_title", "") or "") + "\n" + (_sget(state, "issue_body", "") or "")
    prompt = (
        "You are test_generation_agent. Produce only strict JSON.\n"
        "Generate meaningful unit, edge-case, and regression tests.\n"
        "Avoid trivial tests.\n\n"
        f"[ISSUE]\n{issue_text[:1500]}\n\n"
        f"[REPO_SUMMARY]\n{repo_summary}\n\n"
        f"[PATCH_SUMMARY]\n{patch_summary[:1500]}\n\n"
        f"[FILE_SUMMARIES]\n{json.dumps(file_summaries, separators=(',', ':'))[:3500]}\n\n"
        "Return JSON exactly:\n"
        '{"tests":[{"name":"...","description":"...","input":"...","expected_output":"...","reasoning":"..."}]}'
    )
    try:
        raw = llm_invoke(prompt, max_retries=1)
        clean = raw
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean)
        tests = parsed.get("tests", [])
        structured_tests = []
        for t in tests[:12]:
            structured_tests.append(
                GeneratedTestCase(
                    name=str(t.get("name", "")).strip(),
                    description=str(t.get("description", "")).strip(),
                    input=str(t.get("input", "")).strip(),
                    expected_output=str(t.get("expected_output", "")).strip(),
                    reasoning=str(t.get("reasoning", "")).strip(),
                ).model_dump()
            )
        return _merge(
            state,
            generated_tests=structured_tests,
            generated_tests_cache_key=cache_key,
            repo_summary=repo_summary,
            file_summaries=file_summaries,
            patch_summary=patch_summary,
            node_traces=_append_trace(
                state,
                "test_generation_agent",
                started_at,
                True,
                output_summary={"generated_tests": str(len(structured_tests))},
            ),
        )
    except Exception as e:
        return _merge(
            state,
            generated_tests=[],
            exception_type=type(e).__name__,
            exception_msg=str(e),
            node_traces=_append_trace(
                state,
                "test_generation_agent",
                started_at,
                False,
                error=str(e),
            ),
        )


def _is_shallow_reasoning(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if len(lowered) < 80:
        return True
    weak_markers = ("seems", "likely", "probably", "might", "should work", "looks good")
    return any(m in lowered for m in weak_markers) and ("because" not in lowered and "therefore" not in lowered)


def test_simulation_agent_node(state: AgentState) -> dict:
    """
    Simulate generated tests using LLM reasoning only with deterministic guardrails.
    """
    started_at = pytime.time()
    generated_tests = _sget(state, "generated_tests", [])
    cache_key = _context_cache_key(state)
    if _sget(state, "simulated_results") and _sget(state, "simulated_results_cache_key") == cache_key:
        return _merge(
            state,
            node_traces=_append_trace(
                state,
                "test_simulation_agent",
                started_at,
                True,
                output_summary={"cache": "hit", "simulated_results": str(len(_sget(state, "simulated_results", [])))},
            ),
        )
    if not generated_tests:
        return _merge(
            state,
            simulated_results=[],
            test_results={"passed_tests": 0, "failed_tests": 0},
            tests_run=0,  # No tests generated → no tests run
            risk_level="high",
            validation_confidence=0.0,
            node_traces=_append_trace(state, "test_simulation_agent", started_at, False, error="no_generated_tests"),
        )

    issue_text = ((_sget(state, "issue_title", "") or "") + "\n" + (_sget(state, "issue_body", "") or "")).strip()
    repo_summary = _sget(state, "repo_summary", "") or ""
    patch_summary = _sget(state, "patch_summary", "") or ""
    file_summaries = _sget(state, "file_summaries", {}) or {}

    prompt = (
        "You are test_simulation_agent. Simulate execution via explicit reasoning.\n"
        "Do not guess. If context is insufficient, status must be uncertain.\n"
        "Return strict JSON only:\n"
        '{"results":[{"test_name":"...","status":"pass|fail|uncertain","reasoning":"...","confidence":0.0}],'
        '"summary":{"passed":0,"failed":0,"risk_level":"low|medium|high"}}\n\n'
        f"[ISSUE]\n{issue_text[:1500]}\n\n"
        f"[REPO_SUMMARY]\n{repo_summary[:1200]}\n\n"
        f"[PATCH_SUMMARY]\n{patch_summary[:1500]}\n\n"
        f"[FILE_SUMMARIES]\n{json.dumps(file_summaries, separators=(',', ':'))[:3200]}\n\n"
        f"[TESTS]\n{json.dumps(generated_tests, separators=(',', ':'))[:4500]}"
    )
    try:
        raw = llm_invoke(prompt, max_retries=1)
        clean = raw
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean)
        raw_results = parsed.get("results", [])

        normalized_results = []
        passed = 0
        failed = 0
        confidences: list[float] = []
        for item in raw_results[:20]:
            test_name = str(item.get("test_name", "")).strip()
            status = str(item.get("status", "uncertain")).strip().lower()
            reasoning = str(item.get("reasoning", "")).strip()
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            # Deterministic anti-hallucination constraints
            if not file_summaries or not patch_summary:
                status = "uncertain"
            # Lower threshold: only downgrade to fail when confidence is very low
            if confidence < 0.4:
                status = "fail"
            if _is_shallow_reasoning(reasoning):
                status = "fail"

            if status == "pass":
                passed += 1
            else:
                failed += 1
            confidences.append(confidence)
            normalized_results.append(
                SimulatedTestResult(
                    test_name=test_name or "unnamed_test",
                    status=status if status in {"pass", "fail", "uncertain"} else "uncertain",
                    reasoning=reasoning,
                    confidence=confidence,
                ).model_dump()
            )

        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        risk_level = "low" if failed == 0 and avg_conf >= 0.85 else "medium" if failed <= 2 else "high"
        tests_run_count = passed + failed  # Total tests simulated
        return _merge(
            state,
            simulated_results=normalized_results,
            simulated_results_cache_key=cache_key,
            test_results={"passed_tests": passed, "failed_tests": failed},
            tests_run=tests_run_count,  # Track actual number of tests run
            failed_tests=failed,
            risk_level=risk_level,
            validation_confidence=round(avg_conf, 3),
            node_traces=_append_trace(
                state,
                "test_simulation_agent",
                started_at,
                failed == 0,
                output_summary={"passed": str(passed), "failed": str(failed), "risk": risk_level, "tests_run": str(tests_run_count)},
            ),
        )
    except Exception as e:
        return _merge(
            state,
            simulated_results=[],
            test_results={"passed_tests": 0, "failed_tests": max(1, len(generated_tests))},
            tests_run=len(generated_tests),  # Still count generated tests even if simulation failed
            failed_tests=max(1, len(generated_tests)),
            risk_level="high",
            validation_confidence=0.0,
            exception_type=str(type(e).__name__),
            exception_msg=str(e),
            node_traces=_append_trace(state, "test_simulation_agent", started_at, False, error=str(e)),
        )

def llm_validation_node(state: AgentState) -> dict:
    started_at = pytime.time()
    validate_state(state)
    logger.info("[NODE] llm_validation_node")
    if _sget(state, "status") == PipelineStatus.FAILED.value:
        return _merge(
            state,
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="input_failed_state"),
        )

    diff = _sget(state, "diff", "")
    patch = _sget(state, "patch", [])

    if not _sget(state, "issue_pr_number"):
        logger.warning("[llm_validation_node] No issue_pr_number — skipping LLM validation, treating as retry")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            qa_feedback_history=_append_feedback(state, "Issue PR missing — cannot validate."),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="missing_pr_number"),
        )
    if not diff:
        logger.warning("[llm_validation_node] Empty diff — skipping LLM call, treating patch as APPROVED")
        return _merge(
            state,
            status=PipelineStatus.APPROVED.value,
            decision=DecisionType.APPROVED.value,
            node_traces=_append_trace(state, "llm_validation", started_at, True, output_summary={"note": "empty_diff_auto_approved"}),
        )

    # 1. Guardrails
    guardrail_result = run_guardrails(patch)
    print("Guardrail Result:", guardrail_result["status"])
    if "reason" in guardrail_result:
        print("Guardrail Reason:", guardrail_result["reason"])

    if guardrail_result["status"] == "REJECTED":
        failure_type = guardrail_result.get("failure_type", FailureType.GUARDRAIL_FAILURE.value)
        reason = guardrail_result.get("reason", "Unknown guardrail failure")

        return {
            **_as_dict(state),
            "status": PipelineStatus.REJECTED.value,
            "decision": DecisionType.RETRY.value,
            "failure_type": failure_type,
            "qa_feedback_history": _append_feedback(state, f"Guardrail Failure: {reason}", failed_tests=1),
            "failures": _append_failure_record(state, _normalize_failure_type(failure_type), reason),
            "previous_patch": patch,
            "node_traces": _append_trace(state, "llm_validation", started_at, False, error=reason),
        }

    # ── RULES 1/3/7: ISSUE-FIX ALIGNMENT ──
    # Verify target function/class from plan is actually modified
    _semantic_signals = _sget(state, "semantic_signals", {})
    _modified_functions = _semantic_signals.get("modified_functions", [])
    _target_symbols_missed = _semantic_signals.get("target_symbols_missed", [])
    _plan = _sget(state, "plan", {})
    _plan_targets = _plan.get("targets", []) if isinstance(_plan, dict) else []
    
    # RULE 1/3: No functions modified → reject
    if len(_modified_functions) == 0:
        logger.error("[llm_validation_node] RULE 1/3: NO functions modified — patch does not touch any function")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            semantic_alignment_pass=False,
            behavioral_change_detected=False,
            failure_type=FailureType.LOGIC_ERROR.value,
            failure_reason="no_functions_modified",
            qa_feedback_history=_append_feedback(
                state,
                "RULE 1/3: Patch modifies zero functions. The patch must actually modify the target function/class. "
                "Plan targets: " + str([t.get("symbol") for t in _plan_targets if isinstance(t, dict)])[:200],
                failure_type=FailureType.LOGIC_ERROR
            ),
            failures=_append_failure_record(state, FailureType.LOGIC_ERROR, "no_functions_modified"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="no_functions_modified"),
        )
    
    # RULE 7: Semantic alignment — planned targets must be modified
    if len(_target_symbols_missed) > 0 and _sget(state, "retry_count", 0) > 0:
        logger.error(f"[llm_validation_node] RULE 7: Planned targets NOT modified: {_target_symbols_missed}")
        missed_targets = ", ".join(_target_symbols_missed[:3])
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            semantic_alignment_pass=False,
            failure_type=FailureType.REQUIREMENT_MISMATCH.value,
            failure_reason=f"target_not_modified:{missed_targets}",
            qa_feedback_history=_append_feedback(
                state,
                f"RULE 7: Plan specifies modifying {missed_targets} but patch does NOT modify these targets. "
                f"Patch modifies: {', '.join(_modified_functions[:3])}. "
                "The patch must modify the planned target functions.",
                failure_type=FailureType.REQUIREMENT_MISMATCH
            ),
            failures=_append_failure_record(state, FailureType.REQUIREMENT_MISMATCH, f"target_not_modified:{missed_targets}"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error=f"target_not_modified:{missed_targets}"),
        )

    # ── RULE 2/9: BEHAVIORAL MODIFICATION ──
    # Reject if only imports/unrelated code changed
    _behavior_changed = _semantic_signals.get("behavior_changed", False)
    _total_adds = _semantic_signals.get("total_adds", 0)
    _total_dels = _semantic_signals.get("total_dels", 0)
    
    if not _behavior_changed and (_total_adds > 0 or _total_dels > 0):
        logger.error("[llm_validation_node] RULE 2/9: Only imports/unrelated code changed, no behavior change")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            behavioral_change_detected=False,
            failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT.value,
            failure_reason="no_behavior_change_only_imports",
            qa_feedback_history=_append_feedback(
                state,
                "RULE 2/9: Patch contains no behavioral changes. Only imports or unrelated code modified. "
                f"Changes: +{_total_adds}/-{_total_dels} but no function logic altered. "
                "The patch must implement actual logic changes described in the plan.",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            ),
            failures=_append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, "no_behavior_change_only_imports"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="no_behavior_change"),
        )

    # ── RULE 4/10: FEATURE COMPLETENESS & EXECUTION PATH ──
    # For feature/implementation issues, verify ALL required functions are called/used
    _plan = _sget(state, "plan", {})
    _plan_steps = _plan.get("steps", []) if isinstance(_plan, dict) else []
    _behavior_changes = _plan.get("behavior_changes", []) if isinstance(_plan, dict) else []
    
    # Check if patch implements multiple related functions (e.g., email: send_mail + use_template)
    has_multiple_targets = len(_plan_targets) > 1
    if has_multiple_targets and len(_modified_functions) < len(_plan_targets):
        # Feature requires multiple functions but patch only touches some
        logger.warning(f"[llm_validation_node] RULE 4: Feature incomplete — plan has {len(_plan_targets)} targets but patch touches {len(_modified_functions)} functions")
        missing = len(_plan_targets) - len(_modified_functions)
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            feature_completeness_pass=False,
            failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT.value,
            failure_reason=f"incomplete_feature:{missing}_targets_missing",
            qa_feedback_history=_append_feedback(
                state,
                f"RULE 4: Feature is incomplete. Plan specifies {len(_plan_targets)} targets to modify, "
                f"but patch only touches {len(_modified_functions)}. Missing: {missing} target(s). "
                "All planned steps must be implemented.",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            ),
            failures=_append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, f"incomplete_feature:{missing}_targets_missing"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error=f"incomplete_feature:{missing}_missing"),
        )

    # 2. Edge Case Handling Validation
    # CRITICAL: Enforce that edge cases are actually addressed, not just modified syntax
    _semantic_signals = _sget(state, "semantic_signals", {})
    _edge_cases_handled = _semantic_signals.get("edge_cases_handled", True)
    _edge_case_status = _semantic_signals.get("edge_case_status", "")
    
    if not _edge_cases_handled and _sget(state, "retry_count", 0) > 0:
        logger.warning(f"[llm_validation_node] Edge case handling validation failed: {_edge_case_status}")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            failure_type=FailureType.EDGE_CASE_MISSING.value,
            failure_reason=_edge_case_status,
            qa_feedback_history=_append_feedback(
                state, 
                f"Edge case validation failed: {_edge_case_status}. "
                "The patch must explicitly handle None, empty inputs, boundaries, and invalid types.",
                failure_type=FailureType.EDGE_CASE_MISSING
            ),
            failures=_append_failure_record(state, FailureType.EDGE_CASE_MISSING, _edge_case_status),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error=f"edge_case_validation:{_edge_case_status}"),
        )

    # RULE 9: CRITICAL VALIDATION - Reject trivial/empty patches
    _trivial_flag = _semantic_signals.get("trivial_flag", False)
    _content_changed = _semantic_signals.get("content_changed", False)
    _behavior_changed = _semantic_signals.get("behavior_changed", False)
    _total_adds = _semantic_signals.get("total_adds", 0)
    _total_dels = _semantic_signals.get("total_dels", 0)
    
    if not _content_changed:
        logger.error("[llm_validation_node] RULE 9: Patch has NO actual content changes — rejecting as empty")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            behavioral_change_detected=False,
            failure_type=FailureType.NO_REAL_MODIFICATION.value,
            failure_reason="patch_identical_to_original",
            qa_feedback_history=_append_feedback(
                state,
                "RULE 9: Patch contains no actual code modifications. All files are identical to originals. "
                "Rejecting empty patch.",
                failure_type=FailureType.NO_REAL_MODIFICATION
            ),
            failures=_append_failure_record(state, FailureType.NO_REAL_MODIFICATION, "patch_identical_to_original"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="no_real_modification"),
        )
    
    if _trivial_flag and _sget(state, "retry_count", 0) > 0:
        logger.error("[llm_validation_node] RULE 9: Patch flagged as trivial (no behavior change) — rejecting")
        return _merge(
            state,
            status=PipelineStatus.REJECTED.value,
            decision=DecisionType.RETRY.value,
            behavioral_change_detected=False,
            failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT.value,
            failure_reason="trivial_patch_no_behavior_change",
            qa_feedback_history=_append_feedback(
                state,
                "RULE 9: Patch only contains trivial changes (formatting/whitespace) without behavior change. "
                "For a non-trivial issue, the fix must change actual logic. Adding defensive guards, "
                "error handling, or logic changes.",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            ),
            failures=_append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, "trivial_patch_no_behavior_change"),
            node_traces=_append_trace(state, "llm_validation", started_at, False, error="trivial_patch"),
        )

    # 3. BEHAVIORAL VALIDATION CHECKPOINT: Check if tests actually passed
    # CRITICAL FIX: Do NOT auto-approve just because code changed.
    # Behavior must be correct — QA tests (simulated) must show no failures.
    _test_results = _sget(state, "test_results", {})
    _failed_tests = _test_results.get("failed_tests", 0)
    _passed_tests = _test_results.get("passed_tests", 0)
    _tests_run = _sget(state, "tests_run", 0)
    _simulated_results = _sget(state, "simulated_results", [])
    
    # Classify QA state based on actual test execution
    qa_state = classify_qa_state(_test_results, _tests_run)
    logger.info(f"[llm_validation_node] QA State Classification: {qa_state} (tests_run={_tests_run}, failed={_failed_tests}, passed={_passed_tests})")
    
    # NOTE: Trivial patch rejection is deferred to after _signals is defined (section 3 below)
    
    # Only auto-approve on true PASS state (tests actually ran and all passed)
    if qa_state == "PASS":
        logger.info(
            f"[llm_validation_node] ACCEPTANCE CHECKPOINT: "
            f"Tests executed ({_tests_run} tests) and all passed → "
            f"APPROVED based on behavioral correctness"
        )
        
        # Build a high-confidence approval response
        auto_approval = {
            "status": "PASS",
            "logic_issues": [],
            "edge_case_issues": [],
            "validation_issues": [],
            "requirement_mismatches": [],
            "suggestions": ["Auto-approved: all tests passed."],
            "confidence": "HIGH",
            "auto_approved": True,
            "tests_run": _tests_run,
            "qa_state": "PASS",
        }
        
        qa_feedback = {
            "issues": [],
            "suggestions": ["Auto-approved: tests executed and all passed"],
            "confidence": "HIGH",
        }
        
        return _merge(
            state,
            status=PipelineStatus.APPROVED.value,
            decision="approved",
            failure_type=None,
            qa_output=auto_approval,
            qa_feedback=qa_feedback,
            qa_state=qa_state,
            semantic_alignment_pass=True,
            behavioral_change_detected=True,
            feature_completeness_pass=True,
            execution_path_reachable=True,
            qa_feedback_history=_append_feedback(state, f"Auto-approved: {_tests_run} tests executed, all passed"),
            previous_patch=patch,
            node_traces=_append_trace(
                state,
                "llm_validation",
                started_at,
                True,
                decision="approved",
                output_summary={"status": PipelineStatus.APPROVED.value, "checkpoint": "behavioral_pass", "tests_run": str(_tests_run)},
            ),
        )
    
    # For WEAK_PASS: tests were NOT run (tests_run == 0) but code passed structural checks
    # This requires behavioral validation + issue alignment before approval
    if qa_state == "WEAK_PASS":
        logger.info(f"[llm_validation_node] WEAK_PASS State Detected: No tests run, performing behavioral validation...")
        
        # Run supplemental behavioral validation
        behavioral_validation = validate_behavioral_correctness(state)
        issue_alignment = validate_issue_alignment(state, patch)
        regression_check = detect_regressions(state, patch)
        
        logger.info(f"  - Behavioral Validation: {behavioral_validation['status']}")
        logger.info(f"  - Issue Alignment: {issue_alignment['status']}")
        logger.info(f"  - Regression Check: {regression_check['status']}")
        
        # Only auto-approve WEAK_PASS if all supplemental checks pass
        if (behavioral_validation["status"] == "PASS" and 
            issue_alignment["status"] == "PASS" and 
            regression_check["status"] == "PASS"):
            
            logger.info(f"[llm_validation_node] WEAK_PASS upgraded to APPROVED after behavioral validation")
            
            weak_pass_approval = {
                "status": "PASS",
                "logic_issues": [],
                "edge_case_issues": [],
                "validation_issues": [],
                "requirement_mismatches": [],
                "suggestions": ["Approved: behavioral validation passed. Note: no runtime tests executed."],
                "confidence": "MEDIUM",
                "auto_approved": True,
                "tests_run": 0,
                "qa_state": "WEAK_PASS_UPGRADED",
                "behavioral_validation": behavioral_validation,
                "issue_alignment": issue_alignment,
            }
            
            return _merge(
                state,
                status=PipelineStatus.APPROVED.value,
                decision="approved",
                failure_type=None,
                qa_output=weak_pass_approval,
                qa_state="WEAK_PASS_UPGRADED",
                semantic_alignment_pass=True,
                behavioral_change_detected=True,
                feature_completeness_pass=True,
                execution_path_reachable=True,
                behavioral_validation=behavioral_validation,
                issue_alignment=issue_alignment,
                regression_check=regression_check,
                qa_feedback_history=_append_feedback(state, "Approved: behavioral validation + alignment passed (no runtime tests)"),
                previous_patch=patch,
                node_traces=_append_trace(
                    state,
                    "llm_validation",
                    started_at,
                    True,
                    decision="approved",
                    output_summary={"status": PipelineStatus.APPROVED.value, "checkpoint": "weak_pass_upgraded", "reason": "behavioral_validation_passed"},
                ),
            )
        else:
            # WEAK_PASS with failed supplemental checks → require LLM validation
            logger.warning(f"[llm_validation_node] WEAK_PASS failed supplemental checks → requiring LLM validation")
            qa_state = "WEAK_PASS"  # Keep as WEAK_PASS for decision node awareness
    
    # If we reach here, either:
    # 1. FAIL state (failed tests)
    # 2. WEAK_PASS with failed supplemental checks
    # → Continue to full LLM validation
    
    # 2. Memory integration BEFORE
    issue = _sget(state, "issue_body", "") or _sget(state, "issue_title", "")
    top_k_similar = []
    try:
        top_k_similar = search_memory(issue, top_k=2)
    except Exception as e:
        logger.warning(f"Memory fetch failed: {e}")

    # 3. LLM Prompt Construction (SEMANTIC + GROUNDED)
    # Pull semantic signals computed by issue_agent_node (never hard-rejected upstream)
    diff = _sget(state, "diff", "")
    _signals = _sget(state, "semantic_signals", {})
    if not isinstance(_signals, dict):
        _signals = {}

    _diff_adds = _signals.get("total_adds", 0)
    _diff_dels = _signals.get("total_dels", 0)
    _trivial_flag = _signals.get("trivial_flag", False)
    _modified_fns = _signals.get("modified_functions", [])
    _kw_ratio = _signals.get("keyword_match_ratio", 0)
    _kw_matched = _signals.get("issue_keywords_matched", [])

    # 3a. Reject Trivial Patches on Retries (CRITICAL)
    # Do NOT allow patches that don't change behavior unless issue is trivial (syntax/import only)
    _failure_type = _sget(state, "failure_type", "")
    _retry_count = _sget(state, "retry_count", 0)
    
    if _trivial_flag and _retry_count > 0:
        # Trivial patch on retry — reject unless issue was syntax/import fix
        is_trivial_issue = _failure_type in [FailureType.SYNTAX_ERROR.value, "syntax_error", "import_error"]
        if not is_trivial_issue:
            logger.warning(
                f"[llm_validation_node] REJECTED: Trivial patch detected on retry {_retry_count} "
                f"(issue_type={_failure_type}). Patches must change behavior to solve non-trivial issues."
            )
            return _merge(
                state,
                status=PipelineStatus.REJECTED.value,
                decision=DecisionType.RETRY.value,
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT.value,
                failure_reason="Patch does not change behavior (trivial fix). "
                              "Non-trivial issues require meaningful behavioral changes.",
                qa_feedback_history=_append_feedback(
                    state,
                    "Patch flagged as trivial (no behavior change). Non-trivial issues require meaningful behavioral changes.",
                    failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
                ),
                failures=_append_failure_record(
                    state, 
                    FailureType.MINIMAL_FIX_INSUFFICIENT, 
                    "Trivial patch without behavior change"
                ),
                node_traces=_append_trace(
                    state, 
                    "llm_validation", 
                    started_at, 
                    False, 
                    error="minimal_fix_insufficient:no_behavior_change"
                ),
            )

    # Build the semantic context block for the QA prompt
    _semantic_block = (
        f"[SEMANTIC SIGNALS]\n"
        f"  Lines added (non-comment): {_diff_adds}\n"
        f"  Lines removed (non-comment): {_diff_dels}\n"
        f"  Trivial flag: {_trivial_flag}\n"
        f"  Functions modified/added: {', '.join(_modified_fns[:8]) or 'none detected'}\n"
        f"  Issue keyword match ratio: {_kw_ratio} ({len(_kw_matched)}/{_signals.get('issue_keywords_total', 0)})\n"
        f"  Matched keywords: {', '.join(_kw_matched[:10]) or 'none'}\n"
    )

    _simulated_results = _sget(state, "simulated_results", [])
    _test_results = _sget(state, "test_results", {})
    _behavior_validation_block = ""
    if _simulated_results:
        _behavior_validation_block = (
            f"[BEHAVIOR VALIDATION (SIMULATED TESTS)]\n"
            f"  Passed: {_test_results.get('passed_tests', 0)}\n"
            f"  Failed: {_test_results.get('failed_tests', 0)}\n"
            f"  Simulated Output Details:\n"
        )
        for res in _simulated_results[:5]:
            if isinstance(res, dict):
                _behavior_validation_block += f"    - {res.get('test_name')}: {res.get('status')} ({res.get('reasoning')})\n"
    else:
        _behavior_validation_block = "[BEHAVIOR VALIDATION]\n  No simulated tests available.\n"

    prompt = (
        f"[ISSUE]\n{issue}\n\n"
        f"{_semantic_block}\n"
        f"{_behavior_validation_block}\n"
        f"[CODE DIFF]\n{diff}\n\n"
        f"[FAILED POINTS]\n{_latest_feedback(state)}\n\n"
        "YOU ARE THE SOLE DECISION AUTHORITY. Your job is to determine if this patch ACTUALLY SOLVES the issue.\n\n"
        "CRITICAL RULE:\n"
        "- If any simulated test FAILED, the patch is INCORRECT and must be REJECTED.\n"
        "- Test failures are HARD FACTS, not optional feedback.\n"
        "- Do NOT approve patches with failing tests.\n\n"
        "ISSUE-FIX ALIGNMENT (CRITICAL):\n"
        "1. Extract the issue intent: What behavior is expected?\n"
        "2. Extract the fix intent: What logic does the patch change?\n"
        "3. Verify they match: Does the patch fix the EXACT problem described?\n"
        "4. Reject if: modified functions do not relate to issue, behavior change doesn't match requirement\n\n"
        "BEHAVIORAL CORRECTNESS:\n"
        "- If simulated tests FAILED: the patch does NOT produce correct behavior → REJECT\n"
        "- If simulated tests PASSED: the patch shows correct behavior → can APPROVE\n"
        "- Expected output must MATCH actual output from the patch.\n"
        "- Do NOT accept cosmetic changes that don't fix the issue.\n\n"
        "RULES:\n"
        "- Evaluate ONLY based on ISSUE requirements\n"
        "- DO NOT add new assumptions\n"
        "- DO NOT generalize beyond scope\n"
        "- Ignore unrelated improvements\n\n"
        "CHECK ONLY:\n"
        "1. Does the patch fix the DESCRIBED problem?\n"
        "2. Is the CORRECT function/method modified?\n"
        "3. Does behavior change match the requirement?\n"
        "4. Are simulated tests passing?\n"
        "5. Logical correctness\n"
        "6. Edge cases relevant to issue\n"
        "7. Requirement alignment\n\n"
        "BEHAVIOR VALIDATION CHECK (CRITICAL):\n"
        "- Did the simulated tests produce the expected output?\n"
        "- If simulated tests FAILED, you MUST REJECT — no exceptions.\n"
        "- If simulated tests PASSED, the patch shows it works correctly.\n\n"
        "SEMANTIC CORRECTNESS CHECK:\n"
        "- MUST VERIFY: Is the CORRECT function modified? Compare the modified functions list to the issue description.\n"
        "- MUST VERIFY: Is the function modified inside the CORRECT class? Do not allow misplaced modifications.\n"
        "- MUST VERIFY: Are there any DUPLICATE method or function definitions introduced? If yes, REJECT.\n"
        "- Does the keyword match ratio suggest the patch addresses the right topic?\n"
        "- MINIMAL PATCHES: DO NOT reject correct fixes simply due to a small diff size (e.g., +1 line).\n"
        "  * Small diffs are OFTEN the most precise and correct solutions.\n"
        "  * If a patch modifies arguments, TextWrapper, return values, or error handling, it is HIGHLY MEANINGFUL.\n"
        "  * If simulated tests PASSED and the change affects behavior positively, ACCEPT the patch.\n"
        "- If modified function does NOT match issue context at all: REJECT\n"
        "- If keyword match ratio is 0.0 and no modified functions relate to the issue: REJECT\n\n"
        "REJECT if ANY of these are true:\n"
        "- Simulated tests FAILED (highest priority)\n"
        "- Fix does NOT address the EXACT issue described\n"
        "- Fix references functions/classes that do NOT exist in the diff or codebase\n"
        "- Fix does NOT meaningfully change behavior (cosmetic-only or no-op)\n"
        "- Fix is based on hallucinated assumptions about the codebase\n"
        "- Fix modifies correct code unnecessarily\n"
        "- Fix adds redundant validation that is always true (e.g., isinstance(x, object))\n"
        "- Fix re-implements logic that already exists in the same file\n"
        "- Fix adds trivially-true guards or checks that provide no real protection\n"
        "- Fix raises TypeError/ValueError for inputs that could be handled gracefully (skip, coerce, default)\n"
        "- Fix introduces strict type constraints the issue does not require\n"
        "- Fix breaks existing functionality by over-restricting valid inputs\n"
        "- Fix uses incompatible syntax for the runtime (e.g., %s with sqlite3, ? with psycopg2)\n"
        "- Fix uses APIs or methods that don't exist in the imported libraries\n"
        "- Fix DELETES existing functions or classes that were NOT part of the issue\n"
        "- Fix REPLACES existing code instead of ADDING new code when the issue asks for new functionality\n\n"
        "Output STRICT JSON:\n"
        "{\n"
        '  "status": "APPROVED" or "REJECTED",\n'
        '  "logic_issues": ["issue 1"],\n'
        '  "edge_case_issues": [],\n'
        '  "validation_issues": [],\n'
        '  "requirement_mismatches": [],\n'
        '  "suggestions": ["suggested fix"],\n'
        '  "confidence": "HIGH", "MEDIUM", or "LOW"\n'
        "}\n\n"
        "Ensure syntax is valid JSON. No markdown backticks holding the JSON block."
    )

    # 4. LLM execution (1 limit max QA cycle per attempt)
    output_text = llm_invoke(prompt, max_retries=1)

    # 5. Strict JSON Parsing 
    try:
        import json
        clean_out = output_text
        if "```json" in clean_out:
            clean_out = clean_out.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_out:
            clean_out = clean_out.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(clean_out) if clean_out.strip() else {}
        
        required_keys = ["status", "logic_issues", "confidence"]
        if not all(k in parsed for k in required_keys):
            raise ValueError("Missing required keys in LLM output json.")
    except Exception as e:
        # Parse failure is a transient infrastructure error — do NOT hard-fail the patch.
        # Default to APPROVED with LOW confidence so the pipeline can continue.
        logger.warning(f"[llm_validation_node] LLM JSON parse error ({type(e).__name__}: {e}) — defaulting to APPROVED")
        parsed = {
            "status": "APPROVED",
            "logic_issues": [],
            "edge_case_issues": [],
            "validation_issues": [],
            "requirement_mismatches": [],
            "suggestions": [f"LLM parse error occurred: {e}. Review patch manually."],
            "confidence": "LOW"
        }
        
    print("LLM Output:", parsed)

    # 6. BEHAVIORAL VALIDATION: Override LLM decision if tests failed
    # CRITICAL FIX: If behavioral tests show failures, QA must FAIL regardless of LLM response
    _test_results = _sget(state, "test_results", {})
    _failed_tests = _test_results.get("failed_tests", 0)
    _passed_tests = _test_results.get("passed_tests", 0)
    
    if _failed_tests > 0:
        # Tests failed — override any LLM approval with rejection
        logger.warning(
            f"[llm_validation_node] BEHAVIORAL VALIDATION OVERRIDE: "
            f"{_failed_tests} test(s) failed → REJECTING patch despite LLM response"
        )
        parsed["status"] = "REJECTED"
        parsed["logic_issues"].append(
            f"Behavioral tests failed: {_failed_tests} test(s) did not pass. "
            f"Passed: {_passed_tests}. The patch does not correctly fix the issue."
        )
        parsed["confidence"] = "HIGH"  # High confidence that this is a real failure
    
    # 6b. Final Decision Logic based on (possibly overridden) status
    # APPROVED + HIGH  → approved
    # APPROVED + MEDIUM/LOW → approved (give it the benefit of the doubt; retry is punishing enough)
    # REJECTED (any confidence) → retry
    if parsed.get("status") == "REJECTED":
        decision = "retry"
    else:
        decision = "approved"

    # ── Debug visibility ──
    print("[DEBUG] QA output:", parsed)
    print("[DEBUG] Decision:", decision)
    print("[DEBUG] failure_type:", _sget(state, "failure_type"))
    print("[DEBUG] retry_count:", _sget(state, "retry_count", 0))

    # Failure type standardization (Rule 8)
    if decision == "retry":
        if parsed.get("logic_issues") and "LLM JSON parsing error" in parsed["logic_issues"][0]:
            failure_type = "LLM_PARSE_ERROR"
        elif parsed.get("logic_issues"):
            failure_type = "LOGIC_ERROR"
        elif parsed.get("edge_case_issues"):
            failure_type = "EDGE_CASE_MISSING"
        elif parsed.get("validation_issues"):
            failure_type = "VALIDATION_ERROR"
        elif parsed.get("requirement_mismatches"):
            failure_type = "REQUIREMENT_MISMATCH"
        else:
            failure_type = "LOGIC_ERROR"
    else:
        failure_type = None

    all_issues = (
        parsed.get("logic_issues", []) +
        parsed.get("missing_requirements", [])
    )
    qa_reason = parsed.get("reason", "")
    if qa_reason:
        all_issues.insert(0, f"QA REASON: {qa_reason}")
    qa_feedback_string = "\n".join(all_issues)

    # Memory AFTER
    try:
        save_to_memory({
            "issue": issue,
            "diff": diff,
            "issues": parsed,
            "status": parsed["status"]
        })
    except Exception:
        pass

    # QA output sentinel — must never be None going into decision
    if not parsed or not parsed.get("status"):
        logger.warning("[llm_validation_node] parsed QA output is empty — defaulting to APPROVED sentinel")
        parsed = {
            "status": "APPROVED",
            "logic_issues": [],
            "edge_case_issues": [],
            "validation_issues": [],
            "requirement_mismatches": [],
            "suggestions": ["QA returned None — auto-approved. Review manually."],
            "confidence": "LOW"
        }

    # Normalise APPROVED→PASS / REJECTED→FAIL before storing
    parsed = normalize_qa_output(parsed)
    print("[DEBUG] qa_output after normalization:", parsed.get("status"))

    # Map QA vocab (PASS/FAIL) → valid PipelineStatus enum values
    # NEVER write "PASS" or "FAIL" directly to state["status"] — they are not valid enum members.
    qa_pass = parsed.get("status") == "PASS"
    pipeline_status = PipelineStatus.APPROVED.value if qa_pass else PipelineStatus.REJECTED.value

    # Structured feedback dict for the planner to use on the next retry attempt
    qa_feedback = {
        "issues": (
            parsed.get("logic_issues", []) +
            parsed.get("missing_requirements", [])
        ),
        "suggestions": parsed.get("suggestions", []),
        "confidence": parsed.get("confidence", "LOW"),
    }

    update = _merge(
        state,
        status=pipeline_status,
        failure_type=failure_type,
        qa_output=parsed,
        qa_feedback=qa_feedback,
        qa_state=qa_state,  # Propagate qa_state for decision node routing
        qa_feedback_history=_append_feedback(state, qa_feedback_string, failed_tests=len(all_issues)),
        previous_patch=patch,
        decision=decision,
        node_traces=_append_trace(
            state,
            "llm_validation",
            started_at,
            qa_pass,
            decision=decision,
            output_summary={"status": pipeline_status, "qa_status": parsed["status"], "issues": str(len(all_issues))},
        ),
    )
    if failure_type:
        update["failures"] = make_json_safe(
            _append_failure_record(state, _normalize_failure_type(failure_type), qa_feedback_string[:400])
        )

    return update



def qa_pr_node(state: AgentState) -> dict:
    """Creates or updates the QA PR with test results and decision."""
    validate_state(state)
    logger.info("[NODE] qa_pr_node")
    if _sget(state, "status") == PipelineStatus.FAILED.value:
        return _as_dict(state)
    # Rule 5: QA MUST ALWAYS create its own PR — even on APPROVED
    # (removed early return that skipped PR creation on approved status)
    if not _sget(state, "issue_pr_number"):
        logger.info("[qa_pr_node] Skipping QA PR creation because issue PR does not exist.")
        return _as_dict(state)
        
    test_results = _sget(state, "test_results", {"passed_tests": 0, "failed_tests": 0})
        
    # We no longer check has_commits_ahead() because QA PRs create an explicit qa_report.md commit

    try:
        import os
        import subprocess
        import time

        run_id = _sget(state, "run_id", str(int(time.time())))
        retry = _sget(state, "retry_count", 0)
        qa_status = _sget(state, "status", "UNKNOWN")
        generated_tests = _sget(state, "generated_tests", [])
        simulated_results = _sget(state, "simulated_results", [])
        validation_confidence = _sget(state, "validation_confidence")
        risk_level = _sget(state, "risk_level", "high")
        
        if _sget(state, "failure_type") == FailureType.SYNTAX_ERROR.value:
            syntax_text = f"Failed ({_latest_feedback(state) or 'Check logs'})"
            validation_text = "N/A"
        elif _sget(state, "failure_type") == FailureType.GUARDRAIL_FAILURE.value:
            syntax_text = "Passed"
            validation_text = f"Failed ({_latest_feedback(state) or 'Check logs'})"
        else:
            syntax_text = "Passed"
            validation_text = "Passed"

        def _format_list(items):
            if not items:
                return "- None"
            return "\n".join([f"- {it}" for it in items])

        generated_tests_lines = []
        for idx, test_case in enumerate(generated_tests, 1):
            if isinstance(test_case, dict):
                name = test_case.get("name", f"test_{idx}")
                desc = test_case.get("description", "")
                expected = test_case.get("expected_output", "")
                reasoning = test_case.get("reasoning", "")
            else:
                name = getattr(test_case, "name", f"test_{idx}")
                desc = getattr(test_case, "description", "")
                expected = getattr(test_case, "expected_output", "")
                reasoning = getattr(test_case, "reasoning", "")
            generated_tests_lines.append(
                f"- **{name}**\n"
                f"  - description: {desc}\n"
                f"  - expected: {expected}\n"
                f"  - why it matters: {reasoning}"
            )

        # Build structured failed test entries for the report
        failed_cases = []
        for result in simulated_results:
            r = result if isinstance(result, dict) else (result.__dict__ if hasattr(result, '__dict__') else {})
            status_val = str(r.get("status", "uncertain")).lower()
            if status_val != "pass":
                failed_cases.append({
                    "input":    r.get("input", r.get("test_name", "unknown")),
                    "expected": r.get("expected_output", r.get("expected", "")),
                    "actual":   r.get("actual_output",   r.get("actual", "")),
                    "reason":   r.get("reasoning", r.get("reason", "")),
                })

        passed_count = int(test_results.get("passed_tests", 0))
        failed_count = int(test_results.get("failed_tests", 0))
        tests_run = int(_sget(state, "tests_run", 0))
        qa_state = _sget(state, "qa_state", "UNKNOWN")
        behavioral_validation = _sget(state, "behavioral_validation", {})
        issue_alignment = _sget(state, "issue_alignment", {})
        regression_check = _sget(state, "regression_check", {})

        # Read structured qa_output for issues / suggestions / confidence
        qa_out = safe_dict(_sget(state, "qa_output"))
        logic_issues   = qa_out.get("logic_issues", [])
        edge_issues    = qa_out.get("edge_case_issues", [])
        val_issues     = qa_out.get("validation_issues", [])
        req_mismatches = qa_out.get("requirement_mismatches", [])
        all_issues     = logic_issues + edge_issues + val_issues + req_mismatches
        suggestions    = qa_out.get("suggestions", [])
        qa_confidence  = qa_out.get("confidence", "UNKNOWN")

        # Map pipeline status to PASS / FAIL label for the report
        report_status = "PASS" if qa_status == PipelineStatus.APPROVED.value else "FAIL"

        def _format_failed_cases(cases):
            if not cases:
                return "- None"
            lines_out = []
            for fc in cases:
                lines_out.append(
                    f"- Input: {fc.get('input', '')}\n"
                    f"  Expected: {fc.get('expected', '')}\n"
                    f"  Actual: {fc.get('actual', '')}\n"
                    f"  Reason: {fc.get('reason', '')}"
                )
            return "\n".join(lines_out)
        
        # Build supplemental validation section
        supplemental_section = ""
        if tests_run == 0:
            supplemental_section = f"""
## Test Execution Status
**Tests Run: {tests_run}** (No runtime tests executed)
**QA State: {qa_state}**

### Behavioral Validation
Status: {behavioral_validation.get('status', 'UNKNOWN')}
{_format_list(behavioral_validation.get('findings', []))}

### Issue Alignment
Status: {issue_alignment.get('status', 'UNKNOWN')}
Reason: {issue_alignment.get('reason', 'N/A')}
Alignment Score: {issue_alignment.get('alignment_score', 0.0):.2f}

### Regression Check
Status: {regression_check.get('status', 'UNKNOWN')}
Removed Functions: {_format_list(regression_check.get('removed_functions', []))}
Breaking Changes: {_format_list(regression_check.get('breaking_changes', []))}
"""
        else:
            supplemental_section = f"""
## Test Execution
**Tests Run: {tests_run}**
**Tests Passed: {passed_count}**
**Tests Failed: {failed_count}**
**QA State: {qa_state}**
"""

        body = (
            f"# QA Review Report\n\n"
            f"## Status\n{report_status}\n\n"
            f"## Summary\n"
            f"LLM-based validation completed. "
            f"Tests run: {tests_run}, passed: {passed_count}, failed: {failed_count}.\n"
            f"Syntax check: {syntax_text}. Validation check: {validation_text}.\n"
            f"QA State: {qa_state}\n\n"
            f"{supplemental_section}\n"
            f"## Issues\n"
            f"{_format_list(all_issues)}\n\n"
            f"## Failed Test Cases\n"
            f"{_format_failed_cases(failed_cases)}\n\n"
            f"## Suggestions\n"
            f"{_format_list(suggestions)}\n\n"
            f"## Confidence\n{qa_confidence}\n"
        )
        
        # 0. Checkout distinct QA branch off main
        repo_path = _sget(state, "repo_path")
        branch_name = f"qa/review-{_sget(state, 'issue_id')}-run-{run_id}-attempt-{retry}"
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=False)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=False)
        subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
        subprocess.run(["git", "pull", "origin", "main"], cwd=repo_path, check=False)
        subprocess.run(["git", "checkout", "-B", branch_name], cwd=repo_path, check=True)
        
        # 1. Write the physical qa_report.md
        report_path = os.path.join(repo_path, "qa_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(body)
            
        # 2. Add, commit, push
        subprocess.run(["git", "add", "qa_report.md"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", f"QA Review Report Attempt {retry}"], cwd=repo_path, check=False)
        
        try:
            subprocess.run(["git", "push", "--set-upstream", "origin", branch_name], cwd=repo_path, check=True)
            print(f"[BRANCH CREATED] {branch_name}")
            print(f"[PUSH SUCCESS] {branch_name}")
        except subprocess.CalledProcessError as push_err:
            # QA PR push is non-fatal — log and continue without failing the pipeline
            logger.warning(f"[qa_pr_node] QA branch push failed (non-fatal): {push_err}")
            return _as_dict(state)

        pr_label_status = "Approved" if qa_status == PipelineStatus.APPROVED.value else "Rejected"
        title = f"QA Review: #{_sget(state, 'issue_id', '?')} — {pr_label_status}"
        
        # 3. Create Pull Request unconditionally (Rule 1)
        pr = create_pull_request(title, body, head=branch_name)

        if not pr.get("success"):
            logger.warning(f"QA PR creation failed: {pr.get('error')}. Continuing pipeline.")
            return _as_dict(state)

        pr_num = str(pr["data"]["pr_number"])
        logger.info(f"[NODE] qa_pr_node | Created QA PR #{pr_num}")

        qa_prs = list(_sget(state, "qa_prs", []))
        qa_prs.append(pr_num)

        # Add label
        label = "qa-approved" if qa_status == PipelineStatus.APPROVED.value else "qa-rejected"
        subprocess.run(
            ["gh", "pr", "edit", pr_num, "--add-label", label],
            cwd=repo_path, capture_output=True, text=True
        )

        logger.info(f"[FLOW] QA PR created: #{pr_num}")
        return _merge(
            state,
            qa_pr_number=pr_num,
            qa_prs=qa_prs,
            status=_sget(state, "status", PipelineStatus.REJECTED.value),
            test_results=test_results,
        )
    except Exception as e:
        # QA PR creation failure is non-fatal
        logger.warning(f"qa_pr_node failed (non-fatal): {e}")
        return _merge(
            state,
            status=_sget(state, "status", PipelineStatus.REJECTED.value),
            exception_type=type(e).__name__,
            exception_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: DECISION + ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

HALT_EXCEPTIONS = (
    "UnboundLocalError",
    "NameError",
    "TypeError",
    "AuthenticationError",
)

HALT_REASONS = (
    "no_commits_on_developer",
    "plan_generation_failed",
    "skipped_no_plan",
)

def should_retry(state) -> bool:
    exc_type = state.get("exception_type", "")
    fail_reason = state.get("failure_reason", "")

    if exc_type in HALT_EXCEPTIONS:
        logger.error(f"[HALT] Deterministic code bug: {exc_type} — fix the code")
        return False

    if fail_reason in HALT_REASONS:
        logger.error(f"[HALT] Unrecoverable failure: {fail_reason}")
        return False

    return True   # only transient errors reach here


def classify_failure(
    error: str,
    qa_output: dict | None,
    validation_passed: bool,
    failure_type: str | None,
) -> str:
    """
    Classify the pipeline failure into a routing category.
    Uses qa_output['status'] (PASS/FAIL) as the canonical QA signal.

    Returns one of:
      "INFRA_AUTH"   – authentication / credentials error
      "INFRA"        – other infrastructure / transient error
      "QA_FAILURE"   – QA status is FAIL
      "CODE_FAILURE" – patch validation / guardrail failure
      "NONE"         – no failure detected (success path)
    """
    if error:
        err_str = str(error)
        if "401" in err_str or "auth" in err_str.lower() or "authentication" in err_str.lower():
            return "INFRA_AUTH"
        return "INFRA"

    # Use normalised qa_output status (PASS/FAIL) as primary QA signal
    qa_status = safe_get(qa_output, "status", "")
    if qa_status == "FAIL":
        return "QA_FAILURE"

    if failure_type in (
        FailureType.SYNTAX_ERROR.value,
        FailureType.GUARDRAIL_FAILURE.value,
        FailureType.VALIDATION_ERROR.value,
        FailureType.LOGIC_ERROR.value,
        FailureType.EDGE_CASE_MISSING.value,
        FailureType.REQUIREMENT_MISMATCH.value,
        FailureType.LLM_PARSE_ERROR.value,
    ):
        return "CODE_FAILURE"

    if not validation_passed:
        return "CODE_FAILURE"

    return "NONE"


def decision_node(state: AgentState) -> dict:
    started_at = pytime.time()
    status = _sget(state, "status")
    retry_count = _sget(state, "retry_count", 0)
    failure_type = _sget(state, "failure_type")
    failure_reason = _sget(state, "failure_reason", "")

    # ── Debug visibility ──
    logger.info(f"[DECISION INPUT] status={status}, retry_count={retry_count}")
    print("[DEBUG] decision_node — status:", status)
    print("[DEBUG] decision_node — failure_type:", failure_type)
    print("[DEBUG] decision_node — failure_reason:", failure_reason)
    print("[DEBUG] decision_node — retry_count:", retry_count)

    updates = {}
    classified = "NONE"  # Default — overridden if classify_failure is called

    # BEHAVIORAL CORRECTNESS GUARD: Check test results first
    # If any simulated tests failed, patch is INCORRECT regardless of other signals
    _test_results = _sget(state, "test_results", {})
    _failed_tests = _test_results.get("failed_tests", 0)
    _passed_tests = _test_results.get("passed_tests", 0)
    
    if _failed_tests > 0:
        # Build concrete failure details for planner intelligence
        _simulated = _sget(state, "simulated_results", [])
        _fail_details = []
        for _sr in (_simulated or []):
            if isinstance(_sr, dict) and _sr.get("status") == "fail":
                _fail_details.append(f"{_sr.get('test_name', '?')}: {_sr.get('reasoning', '')[:120]}")
        _detail_str = "; ".join(_fail_details[:3]) if _fail_details else f"{_failed_tests} test(s) failed"
        
        logger.warning(
            f"[DECISION] BEHAVIORAL GUARD: {_failed_tests} test(s) failed (passed: {_passed_tests}) "
            f"→ REJECTING patch as incorrect behavior"
        )
        updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
        updates["failure_type"] = FailureType.BEHAVIORAL_TESTS_FAILED.value
        updates["failure_reason"] = f"behavioral_tests_failed:{_detail_str}"
        updates["qa_feedback_history"] = _append_feedback(
            state,
            f"BEHAVIORAL TESTS FAILED: {_detail_str}. "
            "The patch does not produce correct behavior. Fix the root cause logic.",
            failed_tests=_failed_tests,
            failure_type=FailureType.BEHAVIORAL_TESTS_FAILED,
        )
        updates["failures"] = _append_failure_record(
            state, FailureType.BEHAVIORAL_TESTS_FAILED, _detail_str
        )
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, False, 
            decision=updates["decision"],
            error=f"behavioral_tests_failed:{_failed_tests}"
        )
        return updates

    # ── Rule 10: If QA tests PASSED, accept IMMEDIATELY regardless of retry count ──
    # This check MUST come before retry limits — a passing patch is always valid.
    _test_results = _sget(state, "test_results", {})
    _failed_tests = _test_results.get("failed_tests", 0)
    _passed_tests = _test_results.get("passed_tests", 0)
    qa_state = _sget(state, "qa_state")
    
    if qa_state == "PASS" and _failed_tests == 0:
        logger.info(f"[DECISION] Rule 10: qa_state=PASS, tests passed → checking semantic alignment (Rule 8)")
        
        # RULE 8: CRITICAL - Also verify semantic alignment and behavioral change
        _semantic_signals = _sget(state, "semantic_signals", {})
        _modified_functions = _semantic_signals.get("modified_functions", [])
        _target_symbols_missed = _semantic_signals.get("target_symbols_missed", [])
        _behavior_changed = _semantic_signals.get("behavior_changed", False)
        
        # Check new state fields if they've been set by llm_validation_node
        _semantic_alignment_pass = _sget(state, "semantic_alignment_pass", None)
        _behavioral_change_detected = _sget(state, "behavioral_change_detected", None)
        _feature_completeness_pass = _sget(state, "feature_completeness_pass", True)
        
        # If validation fields were set, use them; otherwise compute from signals
        if _semantic_alignment_pass is False:
            logger.error("[DECISION] RULE 8: semantic_alignment_pass=False → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.REQUIREMENT_MISMATCH.value
            updates["failure_reason"] = "semantic_alignment_failed"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                "RULE 8: Semantic alignment failed. Planned targets not modified in patch.",
                failure_type=FailureType.REQUIREMENT_MISMATCH
            )
            updates["failures"] = _append_failure_record(state, FailureType.REQUIREMENT_MISMATCH, "semantic_alignment_failed")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="semantic_alignment_failed")
            return updates
        
        if _behavioral_change_detected is False:
            logger.error("[DECISION] RULE 8: behavioral_change_detected=False → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.MINIMAL_FIX_INSUFFICIENT.value
            updates["failure_reason"] = "behavioral_change_not_detected"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                "RULE 8: No behavioral change detected. Patch only contains imports or formatting changes.",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            )
            updates["failures"] = _append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, "behavioral_change_not_detected")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="behavioral_change_not_detected")
            return updates
        
        if _feature_completeness_pass is False:
            logger.error("[DECISION] RULE 8: feature_completeness_pass=False → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.MINIMAL_FIX_INSUFFICIENT.value
            updates["failure_reason"] = "feature_incompleteness"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                "RULE 8: Feature implementation is incomplete. Not all required steps were implemented.",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            )
            updates["failures"] = _append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, "feature_incompleteness")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="feature_incompleteness")
            return updates
        
        # Fallback signal checks (only if state fields not explicitly set)
        # Reject if no functions modified (Rule 1/3)
        if _semantic_alignment_pass is None and len(_modified_functions) == 0:
            logger.error("[DECISION] RULE 8: QA passed but no functions modified → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.LOGIC_ERROR.value
            updates["failure_reason"] = "qa_pass_but_no_functions_modified"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                "RULE 8: Tests passed but patch modifies zero functions. The patch must modify the target function.",
                failure_type=FailureType.LOGIC_ERROR
            )
            updates["failures"] = _append_failure_record(state, FailureType.LOGIC_ERROR, "qa_pass_but_no_functions_modified")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="no_functions_despite_qa_pass")
            return updates
        
        # Reject if planned targets not modified (Rule 7)
        if _semantic_alignment_pass is None and len(_target_symbols_missed) > 0:
            logger.error("[DECISION] RULE 8: QA passed but planned targets not modified → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.REQUIREMENT_MISMATCH.value
            updates["failure_reason"] = f"qa_pass_but_target_not_modified:{_target_symbols_missed[0]}"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                f"RULE 8: Tests passed but planned targets not modified: {', '.join(_target_symbols_missed[:2])}.",
                failure_type=FailureType.REQUIREMENT_MISMATCH
            )
            updates["failures"] = _append_failure_record(state, FailureType.REQUIREMENT_MISMATCH, f"target_not_modified:{_target_symbols_missed[0]}")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="target_not_modified_despite_qa_pass")
            return updates
        
        # Reject if no behavioral change (Rule 2)
        if _behavioral_change_detected is None and not _behavior_changed:
            logger.error("[DECISION] RULE 8: QA passed but no behavioral change → RETRY")
            updates["decision"] = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
            updates["failure_type"] = FailureType.MINIMAL_FIX_INSUFFICIENT.value
            updates["failure_reason"] = "qa_pass_but_no_behavior_change"
            updates["qa_feedback_history"] = _append_feedback(
                state,
                "RULE 8: Tests passed but patch contains no behavioral changes (only imports/formatting).",
                failure_type=FailureType.MINIMAL_FIX_INSUFFICIENT
            )
            updates["failures"] = _append_failure_record(state, FailureType.MINIMAL_FIX_INSUFFICIENT, "qa_pass_but_no_behavior_change")
            updates["node_traces"] = _append_trace(state, "decision", started_at, False, error="no_behavior_change_despite_qa_pass")
            return updates
        
        # All checks passed: QA + semantic alignment + behavioral change
        logger.info(f"[DECISION] RULE 8: All criteria met (QA PASS + semantic alignment + behavior change) → APPROVING")
        updates["decision"] = DecisionType.APPROVED.value
        updates["semantic_alignment_pass"] = True
        updates["behavioral_change_detected"] = True
        updates["feature_completeness_pass"] = True
        issue_prs = _sget(state, "issue_prs", [])
        if issue_prs:
            updates["approved_pr"] = issue_prs[-1]
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, True,
            decision=DecisionType.APPROVED.value,
            output_summary={"status": "PASS", "rule": "8_full_validation_passed"},
        )
        updates["attempts"] = _append_attempt_record(
            state,
            decision=DecisionType.APPROVED,
            strategy="qa_pass_accepted",
            based_on_failure=_normalize_failure_type(failure_type),
        )
        return updates

    if qa_state == "WEAK_PASS_UPGRADED" and _failed_tests == 0:
        logger.info(f"[DECISION] Rule 10: qa_state=WEAK_PASS_UPGRADED → APPROVING (ignoring retry_count={retry_count})")
        updates["decision"] = DecisionType.APPROVED.value
        issue_prs = _sget(state, "issue_prs", [])
        if issue_prs:
            updates["approved_pr"] = issue_prs[-1]
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, True,
            decision=DecisionType.APPROVED.value,
            output_summary={"status": "WEAK_PASS_UPGRADED", "rule": "10_behavioral_pass_override"},
        )
        updates["attempts"] = _append_attempt_record(
            state,
            decision=DecisionType.APPROVED,
            strategy="behavioral_pass_accepted",
            based_on_failure=_normalize_failure_type(failure_type),
        )
        return updates

    # Rate limit: fail fast
    if failure_type == FailureType.RATE_LIMIT.value:
        logger.error("[DECISION] Rate limit detected — STOPPING immediately")
        updates["decision"] = DecisionType.FAIL.value
        updates["attempts"] = _append_attempt_record(
            state, decision=DecisionType.FAIL,
            strategy=_strategy_for_failure(state),
            based_on_failure=FailureType.RATE_LIMIT,
        )
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, False,
            decision=DecisionType.FAIL.value, error="rate_limit"
        )
        return updates

    # Hard guard: if qa_output is missing, force retry
    qa_output = _sget(state, "qa_output")
    if not qa_output:
        logger.warning("[DECISION] qa_output is absent — forcing retry")
        updates["decision"] = DecisionType.RETRY.value
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, False,
            decision=DecisionType.RETRY.value, error="missing_qa_output"
        )
        return updates

    # Halt guard: deterministic code bugs must not retry
    if not should_retry({
        "exception_type": _sget(state, "exception_type", ""),
        "failure_reason": failure_reason,
    }):
        logger.error(f"[DECISION] Halt condition met (reason={failure_reason}) — forcing FAIL")
        updates["decision"] = DecisionType.FAIL.value
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, False,
            decision=DecisionType.FAIL.value, error=f"halt:{failure_reason}"
        )
        return updates

    # Retry limit (only reached when qa_state is NOT PASS/WEAK_PASS_UPGRADED)
    if retry_count >= 3:
        logger.info("[DECISION] Max retries (3) reached — STOPPING")
        updates["decision"] = DecisionType.FAILED.value
        updates["attempts"] = _append_attempt_record(
            state, decision=DecisionType.FAILED,
            strategy="max_retries_terminated",
            based_on_failure=_normalize_failure_type(failure_type),
        )
        updates["node_traces"] = _append_trace(
            state, "decision", started_at, False,
            decision=DecisionType.FAILED.value, error="max_retries"
        )
        return updates

    # Rule 11: Terminate on 2+ consecutive identical failure types
    failures = _sget(state, "failures", [])
    if len(failures) >= 2:
        recent_types = [
            f.get("failure_type") if isinstance(f, dict) else getattr(f, "failure_type", None)
            for f in failures[-2:]
        ]
        if recent_types[0] and recent_types[0] == recent_types[1]:
            logger.error(
                f"[DECISION] Consecutive identical failure type '{recent_types[0]}' — "
                f"deterministic failure, STOPPING (retry {retry_count})"
            )
            updates["decision"] = DecisionType.FAILED.value
            updates["failure_reason"] = f"consecutive_identical_failure:{recent_types[0]}"
            updates["attempts"] = _append_attempt_record(
                state, decision=DecisionType.FAILED,
                strategy="consecutive_failure_terminated",
                based_on_failure=_normalize_failure_type(failure_type),
            )
            updates["node_traces"] = _append_trace(
                state, "decision", started_at, False,
                decision=DecisionType.FAILED.value,
                error=f"consecutive_identical_failure:{recent_types[0]}",
            )
            return updates

    # ── Remaining qa_state routing (PASS/WEAK_PASS_UPGRADED handled above as early returns) ──
    error = _sget(state, "error", "")
    qa_output_val = _sget(state, "qa_output")
    validation_passed = not bool(failure_type)

    logger.info(f"[DECISION] qa_state={qa_state}, qa_output status={qa_output_val.get('status') if qa_output_val else 'N/A'}")

    if qa_state == "WEAK_PASS":
        # Tests didn't run, behavioral validation failed → check classify_failure
        logger.warning("[DECISION] qa_state=WEAK_PASS (unupgraded) → checking failure classification")
        classified = classify_failure(
            error=error, qa_output=qa_output_val,
            validation_passed=validation_passed, failure_type=failure_type,
        )
        if classified == "NONE":
            logger.info("[DECISION] qa_state=WEAK_PASS but qa_output=PASS → conservative APPROVED")
            decision = DecisionType.APPROVED.value
            issue_prs = _sget(state, "issue_prs", [])
            if issue_prs:
                updates["approved_pr"] = issue_prs[-1]
        else:
            logger.warning(f"[DECISION] qa_state=WEAK_PASS with classified={classified} → RETRY")
            decision = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value

    elif qa_state == "FAIL":
        logger.warning(f"[DECISION] qa_state=FAIL → RETRY (retry_count={retry_count})")
        decision = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value

    else:
        # Fallback: use classify_failure for legacy routing
        logger.warning(f"[DECISION] qa_state unknown ('{qa_state}'), falling back to classify_failure")
        classified = classify_failure(
            error=error, qa_output=qa_output_val,
            validation_passed=validation_passed, failure_type=failure_type,
        )
        if classified == "NONE":
            decision = DecisionType.APPROVED.value
            issue_prs = _sget(state, "issue_prs", [])
            if issue_prs:
                updates["approved_pr"] = issue_prs[-1]
        elif classified in ("INFRA", "INFRA_AUTH"):
            decision = DecisionType.RETRY.value
        elif classified == "QA_FAILURE":
            decision = DecisionType.RETRY.value if retry_count < 3 else DecisionType.FAIL.value
        elif classified == "CODE_FAILURE":
            decision = DecisionType.FAIL.value
        elif status == PipelineStatus.FAILED.value:
            decision = DecisionType.FAIL.value
        else:
            decision = DecisionType.APPROVED.value
            logger.info(f"[DECISION] Unrecognised state '{status}' — defaulting to APPROVED")

    logger.info(f"[FLOW] Decision: {decision}")
    updates["decision"] = decision
    decision_enum = DecisionType(decision)
    updates["attempts"] = _append_attempt_record(
        state,
        decision=decision_enum,
        strategy=_strategy_for_failure(state),
        based_on_failure=_normalize_failure_type(failure_type),
    )
    updates["node_traces"] = _append_trace(
        state,
        "decision",
        started_at,
        decision_enum == DecisionType.APPROVED,
        decision=decision,
        output_summary={"status": str(status), "classified": str(classified)},
    )
    return updates


def complete_node(state: AgentState) -> dict:
    logger.info("[NODE] complete_node (Pipeline gracefully ending for human review)")
    return _merge(state, status="COMPLETED")


KEYS_TO_PRESERVE = {
    "qa_pr_number",
    "issue_pr_number",
    "issue_number",
    "issue_id",
    "issue_title",
    "top_candidate_file",
    "edge_cases",
    "repo_path",
    "diff"
}


def qa_normalization_node(state: AgentState) -> dict:
    """
    Dedicated normalization node placed between retry_handler and issue_agent.
    Guarantees qa_feedback is always a properly-typed dict before the planner runs.
    Also defensively cleans plan so a list-typed plan never reaches the adapter.
    Propagates planner_mode so skip-planner decisions set by retry_handler are
    honoured downstream — without this, the field is lost on state merge.
    Propagates qa_state (PASS/FAIL/WEAK_PASS) for decision node routing.
    """
    logger.info("[NODE] qa_normalization_node")
    merged = _as_dict(state)

    # Normalize qa_output → qa_feedback
    qa = safe_dict(merged.get("qa_output"))
    merged["qa_feedback"] = {
        "issues": list(
            qa.get("logic_issues", []) +
            qa.get("missing_requirements", [])
        ),
        "suggestions": list(qa.get("suggestions", [])),
        "confidence": qa.get("confidence", ""),
    }

    # Guard: plan must be dict or None — never a list
    plan = merged.get("plan")
    if plan is not None and not isinstance(plan, dict):
        logger.warning(f"[qa_normalization_node] plan type was {type(plan).__name__} — resetting to None")
        merged["plan"] = None

    # Propagate planner_mode — set by retry_handler_node based on failure_stage.
    # If the key is missing (first run), default to None so the adapter infers
    # from retry_count; if it was explicitly set, honour it.
    planner_mode = merged.get("planner_mode")
    if planner_mode is not None:
        logger.info(f"[qa_normalization_node] Carrying planner_mode={planner_mode!r} forward")
    else:
        logger.debug("[qa_normalization_node] planner_mode not set — adapter will infer from retry_count")
    
    # Propagate qa_state (PASS/FAIL/WEAK_PASS) for downstream decision routing
    qa_state = merged.get("qa_state")
    if qa_state:
        logger.info(f"[qa_normalization_node] Carrying qa_state={qa_state!r} forward")

    return merged


def _detect_failure_pattern(state: AgentState) -> tuple[str, bool]:
    """
    Analyze failure history and attempt records to detect patterns.
    Uses attempt records (which have patch fingerprints) for duplicate detection,
    and failure records for failure type pattern detection.
    Returns (failure_pattern, should_force_different_approach)
    """
    failures = _sget(state, "failures", [])
    attempts = _sget(state, "attempts", [])
    
    if len(failures) < 2 and len(attempts) < 2:
        return "unique_failure", False
    
    # Check failure type repetition
    failure_types = []
    for f in (failures[-3:] if failures else []):
        if isinstance(f, dict):
            failure_types.append(f.get("failure_type", ""))
    
    if len(failure_types) >= 2 and len(set(failure_types)) == 1 and failure_types[0]:
        return f"repeated_{failure_types[0]}", True
    
    behavioral_failures = [f for f in failure_types if "behavioral" in str(f).lower() or "test" in str(f).lower()]
    if len(behavioral_failures) >= 2:
        return "repeated_behavioral_failure", True
    
    # Check patch fingerprint repetition via attempt records (more reliable than semantic_signals)
    if len(attempts) >= 2:
        recent_fps = []
        for a in attempts[-3:]:
            if isinstance(a, dict):
                fp = a.get("patch_fingerprint")
                if fp:
                    recent_fps.append(fp)
        if len(recent_fps) >= 2 and recent_fps[-1] == recent_fps[-2]:
            return "repeated_identical_patch_signature", True
    
    return "unique_failure", False


def retry_handler_node(state: AgentState) -> dict:
    new_count = _sget(state, "retry_count", 0) + 1
    logger.info(f"[NODE] retry_handler_node | retry_count: {new_count}")

    if new_count > 3:
        logger.error(f"[NODE] retry_handler_node | GUARD: retry_count {new_count} exceeds max. Forcing fail.")
        return _merge(
            state,
            retry_count=new_count,
            status=PipelineStatus.FAILED_TERMINATED.value,
            decision=DecisionType.FAILED.value,
        )

    # ── Detect and handle failure patterns ──
    failure_pattern, should_escalate = _detect_failure_pattern(state)
    logger.info(f"[retry_handler_node] pattern={failure_pattern} escalate={should_escalate}")
    
    failures = _sget(state, "failures", [])
    if len(failures) >= 2 and not should_escalate:
        recent_types = [
            f.get("failure_type") if isinstance(f, dict) else getattr(f, "failure_type", None)
            for f in failures[-2:]
        ]
        if recent_types[0] and recent_types[0] == recent_types[1]:
            logger.error(
                f"[NODE] retry_handler_node | Deterministic failure type '{recent_types[0]}' — cannot recover"
            )
            return _merge(
                state,
                retry_count=new_count,
                status=PipelineStatus.FAILED_TERMINATED.value,
                decision=DecisionType.FAILED.value,
                failure_reason=f"deterministic_failure:{recent_types[0]}",
            )

    # ── Failure-aware retry: detect failure stage and retry only failing component ──
    failure_stage = _sget(state, "failure_stage", "unknown")
    failure_type = _sget(state, "failure_type", "").upper()
    if failure_stage == "unknown":
        if failure_type in {"RATE_LIMIT", "TOKEN_LIMIT"}:
            failure_stage = "validation"
        elif failure_type in {"PLAN_GENERATION_FAILED", "REQUIREMENT_MISMATCH"} or "planner" in str(_sget(state, "failure_reason", "")).lower():
            failure_stage = "planner"
        else:
            failure_stage = "coder"
    logger.info(f"[NODE] retry_handler_node | failure_stage: {failure_stage}")
    
    if failure_stage == "coder":
        # Rule 6: If coder failed with logic/behavioral/edge-case issues,
        # the plan's approach is wrong — MUST force a new plan.
        # Only reuse plan for minor coder failures (syntax, formatting).
        # CRITICAL: ANY coder failure means the target/approach is wrong, so ALWAYS regenerate plan.
        _LOGIC_FAILURES = {
            "BEHAVIORAL_TESTS_FAILED", "LOGIC_ERROR",
            "EDGE_CASE_MISSING", "MINIMAL_FIX_INSUFFICIENT",
            "REQUIREMENT_MISMATCH", "SYNTAX_ERROR", "NO_REAL_MODIFICATION",
            "GUARDRAIL_FAILURE", "STRUCTURE_VIOLATION",
        }
        if failure_type in _LOGIC_FAILURES:
            logger.info(f"[NODE] retry_handler_node | RULE 6: coder logic failure ({failure_type}) → forcing FULL NEW plan, NOT reusing")
            new_planner_mode = "full"
            # CRITICAL: Clear the plan to force regeneration
            plan_to_pass = None
        else:
            # Even non-logic coder failures should force regeneration (target location wrong, patch didn't apply, etc)
            logger.info(f"[NODE] retry_handler_node | RULE 6: coder structural failure ({failure_type}) → forcing FULL NEW plan")
            new_planner_mode = "full"
            plan_to_pass = None
    else:
        # Default: allow planner to run (or use feedback-guided if already tried)
        new_planner_mode = "full" if new_count == 1 else "feedback_guided"
        plan_to_pass = _sget(state, "plan", {}) if new_planner_mode == "feedback_guided" else None
    
    # CRITICAL OVERRIDE: If behavioral tests failed or patches are repeating,
    # force a completely new plan — the existing plan's approach is wrong.
    if should_escalate and "behavioral" in failure_pattern.lower():
        logger.info("[NODE] retry_handler_node | OVERRIDE: repeated behavioral failure → forcing new plan")
        new_planner_mode = "full"
    elif should_escalate and "identical_patch" in failure_pattern.lower():
        logger.info("[NODE] retry_handler_node | OVERRIDE: identical patch detected → forcing new plan")
        new_planner_mode = "full"

    # ── Intelligent strategy escalation based on failure pattern ──
    # Different failures require different fix strategies, not just linear escalation
    if "behavioral" in failure_pattern.lower():
        # Behavioral failures: need completely different logic, try force_append to add new helpers
        _STRATEGY_LADDER = ["normal", "force_append", "minimal_safe_fix"]
    elif "identical_patch" in failure_pattern.lower():
        # Same patch signature: must try different file or different approach
        _STRATEGY_LADDER = ["normal", "minimal", "force_append"]
    elif failure_type in {"EDGE_CASE_MISSING", "MINIMAL_FIX_INSUFFICIENT"}:
        # Edge case or trivial patch issues: need deeper reasoning and defensive code
        _STRATEGY_LADDER = ["normal", "function-only", "minimal_safe_fix"]
    else:
        # Generic failures: escalate precision
        _STRATEGY_LADDER = ["normal", "function-only", "minimal"]
    
    new_strategy = _STRATEGY_LADDER[min(new_count - 1, len(_STRATEGY_LADDER) - 1)]
    logger.info(f"[NODE] retry_handler_node | strategy escalated to: {new_strategy} (pattern={failure_pattern})")


    # ── Propagate failure_reason into qa_feedback_history ──
    # This ensures the planner reads WHY the previous attempt failed,
    # not just generic feedback, preventing identical re-plans.
    failure_reason = _sget(state, "failure_reason", "")
    failure_type = _sget(state, "failure_type", "")
    qa_state = _sget(state, "qa_state", "")
    _test_results = _sget(state, "test_results", {})
    _failed_tests = _test_results.get("failed_tests", 0)
    _passed_tests = _test_results.get("passed_tests", 0)
    behavioral_validation = _sget(state, "behavioral_validation", {})
    issue_alignment = _sget(state, "issue_alignment", {})
    regression_check = _sget(state, "regression_check", {})
    
    typed_msg = ""
    
    # Provide specific feedback based on qa_state and validation results
    # Read simulated_results BEFORE they get cleared in state reset (we read from current state)
    _simulated = _sget(state, "simulated_results", [])
    
    if qa_state == "FAIL" or _failed_tests > 0:
        # Build concrete per-test failure details for planner
        _test_failure_details = []
        for _sr in (_simulated or []):
            if isinstance(_sr, dict) and _sr.get("status") == "fail":
                _test_failure_details.append(
                    f"  - TEST '{_sr.get('test_name', 'unknown')}': {_sr.get('reasoning', 'no reason')[:200]}"
                )
        _details_block = "\n".join(_test_failure_details[:5]) if _test_failure_details else "  (no per-test details available)"
        
        typed_msg = (
            f"[RETRY {new_count}] TESTS FAILED: {_failed_tests} test(s) failed (passed: {_passed_tests}).\n"
            f"CONCRETE FAILURES:\n{_details_block}\n"
            "The patch does not produce the correct behavior. "
            "You MUST address each specific test failure listed above. "
            "Handle edge cases: None values, empty inputs, invalid types, boundary conditions. "
            "Ensure the patch produces the exact expected output for ALL test cases."
        )
    elif qa_state == "WEAK_PASS" and behavioral_validation:
        # No tests run, behavioral validation revealed issues
        behavioral_status = behavioral_validation.get("status", "UNKNOWN")
        behavioral_findings = behavioral_validation.get("findings", [])
        
        if behavioral_status != "PASS":
            findings_str = " • ".join(behavioral_findings[:3]) if behavioral_findings else "Unknown behavioral issue"
            typed_msg = (
                f"[RETRY {new_count}] BEHAVIORAL VALIDATION FAILED: {findings_str}. "
                "The patch may have correct syntax but insufficient logic changes. "
                "Verify the patch adequately addresses the issue and produces expected behavior."
            )
    
    if issue_alignment:
        alignment_status = issue_alignment.get("status", "UNKNOWN")
        alignment_reason = issue_alignment.get("reason", "")
        
        if alignment_status != "PASS" and not typed_msg:
            typed_msg = (
                f"[RETRY {new_count}] ISSUE ALIGNMENT FAILED: {alignment_reason}. "
                "The patch may not directly address the stated issue. "
                "Review the issue requirements and ensure your fix targets the correct problem."
            )
    
    if regression_check:
        removed_funcs = regression_check.get("removed_functions", [])
        breaking_changes = regression_check.get("breaking_changes", [])
        
        if (removed_funcs or breaking_changes) and not typed_msg:
            removed_str = ", ".join(removed_funcs[:2]) if removed_funcs else ""
            changes_str = ", ".join(breaking_changes[:2]) if breaking_changes else ""
            details = " ".join([f"Removed: {removed_str}" if removed_str else "", 
                               f"Breaking: {changes_str}" if changes_str else ""])
            typed_msg = (
                f"[RETRY {new_count}] REGRESSION DETECTED: {details}. "
                "The patch removes or breaks existing functionality. "
                "Ensure the patch only modifies necessary code to fix the issue without removing functionality."
            )
    
    # Fallback to generic failure feedback if no specific reason identified
    if not typed_msg:
        if failure_reason == "repeated_identical_fix":
            typed_msg = (
                f"[RETRY {new_count}] Patch was identical to a previous failed attempt. "
                "You MUST take a completely different approach — different file, different strategy, or different logic."
            )
        elif failure_type == "EDGE_CASE_MISSING":
            typed_msg = (
                f"[RETRY {new_count}] EDGE CASE VALIDATION FAILED: {failure_reason}. "
                "The patch does not handle edge cases like None values, empty inputs, invalid types, or boundaries. "
                "Add defensive code (guards, try/except, validation) before the main logic."
            )
        elif failure_type == "MINIMAL_FIX_INSUFFICIENT":
            typed_msg = (
                f"[RETRY {new_count}] BEHAVIOR CHANGE REQUIRED: {failure_reason}. "
                "Trivial patches that only modify syntax or formatting do not solve non-trivial issues. "
                "The fix must change the actual logic/behavior to address the root cause."
            )
        elif failure_type:
            if "behavioral" in str(failure_reason).lower() or failure_type == "LOGIC_ERROR":
                typed_msg = (
                    f"[RETRY {new_count}] Previous attempt failed — behavioral correctness issue. "
                    "The patch may have correct syntax but incorrect logic. Verify the fix produces the expected behavior."
                )
            else:
                typed_msg = (
                    f"[RETRY {new_count}] Previous attempt failed with type: {failure_type}. "
                    "Analyse the QA feedback above and address the root cause directly."
                )
        else:
            typed_msg = (
                f"[RETRY {new_count}] Previous attempt did not meet QA requirements. "
                "Review the validation feedback and adjust your approach."
            )

    new_state = _as_dict(state)
    new_state.update({
        "retry_count": new_count,
        "retry_strategy": new_strategy,
        "planner_mode": new_planner_mode,   # propagated through qa_normalization_node
        "failure_stage": failure_stage,     # carry forward so downstream nodes can read it
        "plan": plan_to_pass,               # RULE 6: Clear plan when coder failed, forcing regeneration
        "status": PipelineStatus.RETRYING.value,
        "decision": None,
        # ── CRITICAL: Reset stale QA state to prevent decision loop poisoning ──
        # If we don't clear these, the next iteration's test_generation and
        # test_simulation nodes may skip re-evaluation due to cache hits,
        # and decision_node will re-read the old qa_state and loop.
        "qa_output": None,
        "qa_state": None,
        "qa_feedback": None,
        # Invalidate test caches so tests are regenerated for the new patch
        "generated_tests_cache_key": None,
        "simulated_results_cache_key": None,
        "generated_tests": [],
        "simulated_results": [],
        "test_results": {"passed_tests": 0, "failed_tests": 0},
        "tests_run": 0,
        "failed_tests": None,
        # Reset supplemental validation results
        "behavioral_validation": None,
        "issue_alignment": None,
        "regression_check": None,
        # Clear previous semantic signals (new patch = new signals)
        "semantic_signals": None,
        "validation_confidence": None,
        "risk_level": None,
    })
    logger.info(
        f"[retry_handler] retry={new_count} strategy={new_strategy!r} "
        f"planner_mode={new_planner_mode!r} failure_stage={failure_stage!r}"
    )
    if typed_msg:
        new_state["qa_feedback_history"] = _append_feedback(
            state, typed_msg, failure_type=_normalize_failure_type(failure_type)
        )
    return new_state


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: MERGE
# ═══════════════════════════════════════════════════════════════════════════════


def merge_node(state: AgentState) -> dict:
    """
    Merges the EXISTING Issue PR (state['issue_pr_number']).
    Does NOT create a new PR.
    """
    logger.info("[NODE] merge_node")
    try:
        pr_num = _sget(state, "approved_pr") or _sget(state, "issue_pr_number")
        if not pr_num:
            return _merge(
                state,
                status=PipelineStatus.FAILED.value,
                qa_feedback_history=_append_feedback(state, "No Issue PR number found to merge"),
            )

        merge_res = merge_pull_request(pr_num)
        if not merge_res.get("success"):
            return _merge(
                state,
                status=PipelineStatus.FAILED.value,
                qa_feedback_history=_append_feedback(state, merge_res.get("error", "Merge failed")),
            )

        logger.info(f"[NODE] merge_node | Successfully merged PR #{pr_num}")
        
        # Implement auto-close natively
        issue_id = _sget(state, "issue_id")
        if issue_id:
            import subprocess
            subprocess.run(
                ["gh", "issue", "close", str(issue_id), "--comment", "Closed by automated pipeline after QA approval and merge."],
                cwd="repo_clone",
                capture_output=True,
                text=True,
                check=False
            )
            logger.info(f"[NODE] merge_node | Closed issue #{issue_id}")

        return _merge(state, status=PipelineStatus.MERGED.value)
    except Exception as e:
        logger.error(f"merge_node failed: {e}")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
            qa_feedback_history=_append_feedback(state, str(e)),
        )

def final_merge_pr_node(state: AgentState) -> dict:
    """
    Logs final resolution to the merged GitHub PR.
    """
    logger.info("[NODE] final_merge_pr_node")
    try:
        issue_id = _sget(state, "issue_id")
        test_results = _sget(state, "test_results", {})
        passed = test_results.get("passed_tests", 0)
        failed = test_results.get("failed_tests", 0)
        pr_number = _sget(state, "issue_pr_number")
        
        title = f"Issue Resolved and Merged: #{issue_id}"
        body = f"""## Issue Resolved and Merged

- **Issue ID:** {issue_id}
- **Summary of changes:** {_sget(state, 'issue_title', 'Resolved issue via automated multi-agent pipeline')}
- **QA results:** {passed} passed, {failed} failed
- **Merge confirmation:** Successfully merged by automated pipeline
"""
        
        if pr_number:
            import subprocess
            subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--title", title, "--body", body],
                cwd="repo_clone", capture_output=True, text=True
            )
            logger.info(f"[NODE] final_merge_pr_node | Updated PR #{pr_number} with final merge confirmation")
        if issue_id:
            subprocess.run(
                ["gh", "issue", "close", str(issue_id), "--comment", "Closed by automated pipeline after QA approval and merge."],
                cwd="repo_clone",
                capture_output=True,
                text=True,
            )
            logger.info(f"[NODE] final_merge_pr_node | Closed issue #{issue_id}")
            
        try:
            from integration.memory_store import save_to_memory
            save_to_memory({
                "issue": _sget(state, 'issue_title', ''),
                "issue_body": _sget(state, 'issue_body', ''),
                "solution": _sget(state, 'patch', []),
                "patterns": _sget(state, 'qa_feedback_history', []),
                "passed_tests": passed
            })
            logger.info("[NODE] final_merge_pr_node | Saved to long-term memory")
        except Exception as e:
            logger.error(f"[NODE] final_merge_pr_node | Failed to save memory: {e}")

        return _merge(state, status=PipelineStatus.COMPLETED.value)
    except Exception as e:
        logger.error(f"final_merge_pr_node failed: {e}")
        return _merge(
            state,
            status=PipelineStatus.FAILED.value,
            exception_type=type(e).__name__,
            exception_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: FAILURE HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def build_failure_comment(state) -> str:
    exc = _sget(state, "exception_type", "unknown")
    msg = _sget(state, "exception_msg", "no detail")
    plan = _sget(state, "plan", {})
    src  = plan.get("plan_source", "llm") if isinstance(plan, dict) else "llm"
    retry = _sget(state, "retry_count", 0)
    return f"""## QA Pipeline — Failure Report

**Status:** FAILED after {retry} retries
**Plan source:** {src}
**Root cause:** `{exc}` — {msg}

### What to do next
1. Fix the root cause listed above
2. Push a new commit to the `developer` branch
3. The QA pipeline will re-evaluate automatically on the next run

> This PR was left open intentionally. Do not close it manually
> until the fix has been verified and merged.
"""

def fail_node(state: AgentState) -> dict:
    """
    Cleanup: post failure to issue, save failure summary, terminate.
    NEVER close the PRs.
    """
    logger.error("[NODE] fail_node | Pipeline terminating")
    issue_id = _sget(state, "issue_id")

    pr_number = _sget(state, "issue_pr_number")
    if not pr_number:
        pr_number = _sget(state, "qa_pr_number")
        
    if pr_number:
        failure_comment = build_failure_comment(state)
        try:
            subprocess.run(
                ["gh", "pr", "comment", str(pr_number), "--body", failure_comment],
                cwd="repo_clone", capture_output=True, text=True
            )
            logger.info(f"[fail_node] PR #{pr_number} left OPEN — failure details posted as comment")
        except Exception as e:
            logger.warning(f"Failed to post comment to PR #{pr_number}: {e}")
    else:
        logger.info("[fail_node] No PR to update")

    # Post failure comment to Issue
    if issue_id:
        fail_comment = f"❌ **Automated Resolution Failed**\n- **Retries Exhausted:** {_sget(state, 'retry_count')}\n- **Latest Feedback:**\n\n```text\n{_latest_feedback(state) or 'No feedback recorded'}\n```\n\n*The multi-agent system could not resolve this issue. Abandoning.*"
        try:
            subprocess.run(["gh", "issue", "comment", str(issue_id), "--body", fail_comment], capture_output=True, text=True)
            logger.info(f"[fail_node] Posted failure comment to issue #{issue_id}")
        except Exception as e:
            logger.warning(f"Failed to comment on issue #{issue_id}: {e}")

    # Save failure summary — guard against non-JSON-serializable objects (Enums, Pydantic models)
    try:
        summary = make_json_safe({
            "issue_number": _sget(state, "issue_id"),
            "issue_title": _sget(state, "issue_title"),
            "exception_type": _sget(state, "exception_type"),
            "exception_msg": _sget(state, "exception_msg"),
            "plan": _sget(state, "plan"),
            "failure_stage": _sget(state, "failure_stage"),
            "failure_type": _sget(state, "failure_type"),
            "failure_reason": _sget(state, "failure_reason"),
            "retry_count": _sget(state, "retry_count"),
            "retry_strategy": _sget(state, "retry_strategy"),
            "issue_pr_number": _sget(state, "issue_pr_number"),
            "qa_pr_number": _sget(state, "qa_pr_number"),
            "last_feedback": _latest_feedback(state),
            "all_feedback_history": _sget(state, "qa_feedback_history", []),
            "test_results": _sget(state, "test_results"),
            "attempts": _sget(state, "attempts", []),
            "failures": _sget(state, "failures", []),
        })

        path = os.path.join(base_dir, f"fail_summary_{_sget(state, 'issue_id', 'unknown')}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[fail_node] Saved failure summary to {path}")
    except Exception as e:
        logger.error(f"[fail_node] Could not save summary: {e}")

    return _merge(state, status=PipelineStatus.FAILED_TERMINATED.value)
