import os
import sys
import importlib.util

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
issue_agent_dir = os.path.join(base_dir, "agents", "ai_issue_agent")
sys.path.insert(0, issue_agent_dir)


def _load_issue_graph_builder():
    graph_path = os.path.join(issue_agent_dir, "graph.py")
    spec = importlib.util.spec_from_file_location("ai_issue_agent_graph", graph_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load issue graph module from {graph_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "build_graph")


def _normalize_previous_plan(previous_plan):
    return previous_plan if isinstance(previous_plan, dict) and previous_plan else None


def _normalize_previous_patch(previous_patch):
    if isinstance(previous_patch, dict):
        return {
            str(path): str(content)
            for path, content in previous_patch.items()
            if isinstance(path, str) and isinstance(content, str) and path.strip() and content.strip()
        }

    if not isinstance(previous_patch, list):
        return {}

    normalized = {}
    for change in previous_patch:
        if isinstance(change, dict):
            file_path = change.get("file_path")
            content = change.get("diff")
            change_type = change.get("change_type", "modify")
        else:
            file_path = getattr(change, "file_path", None)
            content = getattr(change, "diff", None)
            change_type = getattr(change, "change_type", "modify")

        if change_type == "delete":
            continue
        if isinstance(file_path, str) and isinstance(content, str) and file_path.strip() and content.strip():
            normalized[file_path] = content

    return normalized

def _infer_failure_stage(result: dict) -> str:
    if not isinstance(result, dict):
        return "coder"
    failure_stage = result.get("failure_stage")
    if isinstance(failure_stage, str) and failure_stage.strip():
        return failure_stage
    status = result.get("status", "").lower()
    failure_type = str(result.get("failure_type", "")).upper()
    error = str(result.get("error", "")).lower()

    if failure_type in {"RATE_LIMIT", "TOKEN_LIMIT"}:
        return "validation"
    if "planner" in status or "planner" in error or failure_type == "PLAN_GENERATION_FAILED":
        return "planner"
    if "diff" in error or "patch" in error or "coder" in error or failure_type in {"LOGIC_ERROR", "NO_REAL_MODIFICATION"}:
        return "coder"
    if "validation" in error or "syntax" in error or failure_type == "VALIDATION_ERROR":
        return "validation"
    return "coder"


def run_issue_agent(
    issue_id: str, 
    issue_title: str, 
    issue_body: str, 
    repo_path: str, 
    qa_feedback: str = None, 
    qa_feedback_history: list = None, 
    retry_count: int = 0,
    retry_strategy: str = "normal",
    previous_plan: dict = None,
    failure_type: str = None,
    failed_tests: int = None,
    previous_patch: dict = None,
    planner_mode: str = None,   # explicit override from outer pipeline state
    simulated_results: list = None,  # per-test failure details from test_simulation_agent
) -> dict:
    """Adapter to translate inputs into issue agent's expected state and extract patch."""
    issue_text = f"#{issue_id}: {issue_title}\n\n{issue_body}"
    normalized_previous_plan = _normalize_previous_plan(previous_plan)
    normalized_previous_patch = _normalize_previous_patch(previous_patch)

    initial_state = {
        "issue": issue_text,
        "issue_meta": {
            "number": issue_id,
            "title": issue_title,
            "body": issue_body
        },
        "repo_path": repo_path,
        "plan": normalized_previous_plan,
        "context": "",
        "original_files": {},
        "code_diffs": {},
        "patched_files": {},
        "errors": "",
        "previous_diffs": {},
        "retries": retry_count,
        "retry_strategy": retry_strategy,
        "status": "running",
        "edge_cases": [],
        # Honour explicit planner_mode from outer pipeline state (set by retry_handler_node).
        # If None/unset, fall back to inferring from retry_count so first-run is always "full".
        "planner_mode": planner_mode if planner_mode is not None else (
            "feedback_guided" if retry_count > 0 else "full"
        ),
        "qa_feedback": qa_feedback,
        "qa_feedback_history": qa_feedback_history or [],
        "failure_type": failure_type,
        "failed_tests": failed_tests,
        "previous_patch": normalized_previous_patch,
        "simulated_results": simulated_results or [],
    }

    build_graph = _load_issue_graph_builder()
    graph = build_graph()

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        exc_name = type(e).__name__
        exc_msg = str(e)
        # Detect rate-limit / quota exhaustion
        is_rate_limit = (
            "ratelimit" in exc_name.lower()
            or "rate limit" in exc_msg.lower()
            or "429" in exc_msg
            or "quota" in exc_msg.lower()
        )
        # Detect token-budget overflow
        is_token_limit = (
            "tokenlimit" in exc_name.lower()
            or "413" in exc_msg
            or "context_length" in exc_msg.lower()
            or "maximum context" in exc_msg.lower()
        )

        # If rate/token limit hit and we have a previous valid patch,
        # reuse it instead of retrying uselessly.
        if (is_rate_limit or is_token_limit) and normalized_previous_patch:
            print(
                f"[ADAPTER] {exc_name} hit but previous valid patch exists "
                f"({len(normalized_previous_patch)} file(s)) — reusing."
            )
            return {
                "success": True,
                "patch": normalized_previous_patch,
                "plan": normalized_previous_plan or {},
            }

        if is_rate_limit:
            return {
                "success": False,
                "error": f"RATE_LIMIT: {exc_msg}",
                "failure_type": "RATE_LIMIT",
                "failure_stage": "validation",
            }
        if is_token_limit:
            return {
                "success": False,
                "error": f"TOKEN_LIMIT: {exc_msg}",
                "failure_type": "TOKEN_LIMIT",
                "failure_stage": "validation",
            }
        return {
            "success": False,
            "error": f"{exc_name}: {exc_msg}",
            "failure_type": "UNKNOWN",
            "failure_stage": _infer_failure_stage({"error": f"{exc_name}: {exc_msg}", "failure_type": "UNKNOWN"}),
        }

    status = result.get("status")
    patched_files = result.get("patched_files")
    if status not in ["success", "running"] or not patched_files:
        error_text = result.get("errors") or result.get("error") or "No patch generated by issue agent."
        return {
            "success": False,
            "error": f"Issue agent failed (status={status}): {error_text}",
            "failure_type": result.get("failure_type") or "UNKNOWN",
            "failure_stage": _infer_failure_stage(result),
        }

    return {
        "success": True,
        "patch": patched_files,
        "plan": result.get("plan", {})
    }
