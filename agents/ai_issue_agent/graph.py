"""
graph.py

LangGraph-based pipeline for solving GitHub issues.

Nodes:
  1. planner_node   -- Analyzes issue, produces JSON plan (files + steps)
  2. retrieval_node -- Reads planned files, builds relevant code context
  3. coder_node     -- Generates unified diffs for each file in the plan
  4. feedback_node  -- Validates diffs, runs quality checks, decides retry/success/failure

Graph flow:
  START -> planner -> retrieval -> coder -> feedback
                                    ^          |
                                    +-- retry -+

State is a pure TypedDict -- no hidden globals, no side effects on disk.
File writes happen AFTER the graph returns successfully (in main.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import difflib
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, END

# Ensure local module directories resolve first, independent of workspace package layout.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.join(_BASE_DIR, "agents")
_UTILS_DIR = os.path.join(_BASE_DIR, "utils")
for _p in (_AGENTS_DIR, _UTILS_DIR, _BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner import plan_issue
from retriever import HybridRetriever
from patch_generator import PatchGenerator
from output_validators import LLMOutputValidator
from quality_checker import run_quality_checklist
from edge_case_analyzer import analyze_edge_cases

try:
    from .config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

MAX_RETRIES = 1


class AgentState(TypedDict):
    issue: str                    # issue title + body
    issue_meta: dict              # {number, title, body} -- for PR creation
    repo_path: str                # path to cloned repo
    plan: dict                    # {files_to_modify, files_to_read, steps}
    context: str                  # concatenated relevant code from retrieval
    original_files: dict          # {filepath: original_content}
    code_diffs: dict              # {filepath: unified_diff_str}
    patched_files: dict           # {filepath: patched_content}
    errors: str                   # accumulated error descriptions
    previous_diffs: dict          # diffs from the last failed attempt
    retries: int                  # current retry count
    status: str                   # "running" | "success" | "failure" | "planner_failed"
    edge_cases: list              # edge cases from heuristic analyzer
    planner_mode: str             # "full" or "feedback_guided"

    qa_feedback: str              # feedback string from qa tests
    qa_feedback_history: list     # history map of older runs
    failure_type: str             # classification error type
    failed_tests: int             # test integer count
    previous_patch: dict          # older patch dicts mapping
    retry_strategy: str           # The current strategy ladder step
    failure_stage: str            # "planner", "retrieval", "coder", "feedback" - tracks where failure occurred
    previous_plan: dict           # previous plan for comparison
    simulated_results: list       # per-test pass/fail details from test_simulation_agent


def _is_plan_reusable(existing_plan: dict | None, failure_type: str | None, retry_strategy: str) -> bool:
    if not isinstance(existing_plan, dict):
        return False
    files_to_modify = existing_plan.get("files_to_modify")
    steps = existing_plan.get("steps")
    if not isinstance(files_to_modify, list) or not files_to_modify:
        return False
    if not isinstance(steps, list) or not steps:
        return False

    # Rule 6: If failure was in coder or logic, the plan's approach is wrong — MUST replan
    if failure_type in {
        "PLAN_GENERATION_FAILED", "REQUIREMENT_MISMATCH",
        "BEHAVIORAL_TESTS_FAILED", "LOGIC_ERROR",
        "EDGE_CASE_MISSING", "MINIMAL_FIX_INSUFFICIENT",
    }:
        return False
    # Rule 7: These strategies indicate the previous plan was inadequate
    if retry_strategy in {"simplified", "fallback", "force_append", "minimal_safe_fix"}:
        return False

    # Only reuse plan for minor failures (formatting, syntax, guardrail)
    _REUSABLE_FAILURES = {
        "SYNTAX_ERROR", "GUARDRAIL_FAILURE", "FORMATTING_ERROR",
        None, "", "UNKNOWN",
    }
    if failure_type not in _REUSABLE_FAILURES:
        return False

    return True


def _plan_hash(plan: dict | None) -> str | None:
    """Normalize and hash a plan for reliable equality checks."""
    if not isinstance(plan, dict):
        return None

    normalized_plan = {
        "files_to_modify": sorted(plan.get("files_to_modify", []) if isinstance(plan.get("files_to_modify", []), list) else []),
        "files_to_read": sorted(plan.get("files_to_read", []) if isinstance(plan.get("files_to_read", []), list) else []),
        "steps": plan.get("steps", []) if isinstance(plan.get("steps", []), list) else [],
        "targets": plan.get("targets", []) if isinstance(plan.get("targets", []), list) else [],
        "behavior_changes": plan.get("behavior_changes", []) if isinstance(plan.get("behavior_changes", []), list) else [],
    }

    try:
        plan_text = json.dumps(normalized_plan, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return None

    return hashlib.sha256(plan_text.encode("utf-8")).hexdigest()


def _are_plans_identical(plan1: dict | None, plan2: dict | None) -> bool:
    """Check if two plans are functionally identical."""
    if not isinstance(plan1, dict) or not isinstance(plan2, dict):
        return False

    hash1 = _plan_hash(plan1)
    hash2 = _plan_hash(plan2)
    if hash1 is None or hash2 is None:
        return False

    return hash1 == hash2


def _planner_candidate_limit(retry_strategy: str) -> int:
    return {
        "normal": 50,
        "simplified": 30,
        "minimal": 20,
        "fallback": 15,
    }.get(retry_strategy, 30)


# ---------------------------------------------------------------------------
# Error summarizer
# ---------------------------------------------------------------------------

def _summarize_errors(raw_errors: list[str]) -> str:
    """
    Summarize a list of error messages into a clean, concise block
    for the coder agent. Avoids passing raw logs.

    Args:
        raw_errors: List of error message strings.

    Returns:
        A deduplicated, truncated error summary string, or "" if empty.
    """
    if not isinstance(raw_errors, list):
        return ""

    if not raw_errors:
        return ""

    # Deduplicate and limit
    seen = set()
    unique = []
    for err in raw_errors:
        if not isinstance(err, str):
            err = str(err)
        key = err.strip().lower()[:100]
        if key and key not in seen:
            seen.add(key)
            unique.append(err.strip())

    if not unique:
        return ""

    summary_lines = [f"  - {e[:200]}" for e in unique[:8]]
    return "Error summary:\n" + "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    Analyze the issue and produce a structured JSON plan.
    Uses the retriever to find candidate files first.
    """
    print("\n" + "=" * 60)
    print("[NODE] planner_node")
    print("=" * 60)

    try:
        issue_text = state.get("issue", "")
        repo_path = state.get("repo_path", "")
        retry_strategy = state.get("retry_strategy", "normal")
        failure_type = state.get("failure_type")

        # --- Input validation ---
        if not issue_text or not isinstance(issue_text, str):
            print("[ERROR] planner_node: issue text is empty or invalid")
            return {"status": "planner_failed"}

        if not repo_path or not isinstance(repo_path, str):
            print("[ERROR] planner_node: repo_path is empty or invalid")
            return {"status": "planner_failed"}

        if not os.path.isdir(repo_path):
            print(f"[ERROR] planner_node: repo_path does not exist: {repo_path}")
            return {"status": "planner_failed"}

        # --- Cached plan reuse ---
        previous_plan = state.get("previous_plan")
        existing_plan = state.get("plan")
        if not isinstance(existing_plan, dict):
            existing_plan = None

        existing_plan_hash = _plan_hash(existing_plan)
        previous_plan_hash = _plan_hash(previous_plan)

        if existing_plan_hash and previous_plan_hash and existing_plan_hash == previous_plan_hash:
            print("[NODE] planner_node: Plan identical to previous. Skipping planner.")
            return {
                "status": "running",
                "plan": existing_plan,
                "previous_plan": existing_plan,
                "plan_hash": existing_plan_hash,
            }

        # --- Caching Optimization / Rule 7 (Disable LLM on Retries) ---
        planner_mode = state.get("planner_mode", "full")

        if planner_mode == "feedback_guided" and _is_plan_reusable(existing_plan, failure_type, retry_strategy):
            print(f"[NODE] planner_node: planner_mode=feedback_guided. Skipping LLM planner (Rule 7). Reusing cached plan.")
            return {
                "status": "running",
                "plan": existing_plan,
                "previous_plan": existing_plan,
                "plan_hash": existing_plan_hash,
            }

        # Build retrieval index to find candidate files
        retriever = HybridRetriever(repo_path=repo_path)
        retriever.build_index()

        # Get top candidates for the planner to evaluate
        candidates = retriever.query(issue_text, top_k=_planner_candidate_limit(retry_strategy))

        if not candidates:
            print("[ERROR] No candidate files found in repository")
            return {"status": "planner_failed"}

        retriever.print_candidates(candidates[:5])

        # Run heuristic edge case analysis
        sample_code = candidates[0].get("snippet", "") if candidates else ""
        edge_cases = analyze_edge_cases(issue_text, sample_code)

        # Call the planner with strict JSON output
        # max_retries=3: the planner uses progressive strictness on each attempt,
        # so giving it 3 retries prevents single-attempt JSON-parse fragility.
        plan = plan_issue(
            issue_text=issue_text, 
            candidate_files=candidates,
            qa_feedback=state.get("qa_feedback"),
            failure_type=state.get("failure_type"),
            failed_tests=state.get("failed_tests"),
            repo_path=repo_path,
            retry_strategy=retry_strategy,
            previous_patch=state.get("previous_patch"),
            simulated_results=state.get("simulated_results", []),
            max_retries=3,
        )

        if plan is None:
            print("[ERROR] Planner failed to generate a valid plan.")
            return {"status": "planner_failed", "edge_cases": edge_cases, "failure_stage": "planner"}

        return {
            "plan": plan,
            "previous_plan": existing_plan,  # Store current plan as previous for next comparison
            "plan_hash": _plan_hash(plan),
            "edge_cases": edge_cases,
            "status": "running",
        }
    except Exception as e:
        print(f"[ERROR] Planner node crashed: {e}")
        return {
            **state,
            "status": "planner_failed",
            "failure_stage": "planner",
            "error": f"Planner failed: {str(e)}",
            "decision": "fail"
        }


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# ---------------------------------------------------------------------------

def retrieval_node(state: AgentState) -> dict:
    """
    Read the files specified in the plan and build context.
    Reads both files_to_modify (as originals) and files_to_read (as context).
    Applies per-file size limits to prevent context overflow.
    Context size is reduced progressively based on retry_strategy:
      normal       -> 16000 chars/file (full file)
      function-only -> extract only the most relevant function (AST-aware)
      minimal      -> 6000 chars/file (tight budget)
    """
    print("\n" + "=" * 60)
    print("[NODE] retrieval_node")
    print("=" * 60)

    plan = state.get("plan", {})
    repo_path = state.get("repo_path", "")
    retry_strategy = state.get("retry_strategy", "normal")

    # --- Input validation ---
    if not isinstance(plan, dict) or not plan:
        print("[ERROR] retrieval_node: plan is empty or invalid")
        return {"status": "failure", "errors": "Plan is empty or invalid"}

    if not repo_path or not isinstance(repo_path, str):
        print("[ERROR] retrieval_node: repo_path is empty or invalid")
        return {"status": "failure", "errors": "repo_path is empty or invalid"}

    files_to_modify = plan.get("files_to_modify", [])
    files_to_read = plan.get("files_to_read", [])

    if not isinstance(files_to_modify, list) or not files_to_modify:
        print("[ERROR] retrieval_node: no files_to_modify in plan")
        return {"status": "failure", "errors": "Plan has no files_to_modify"}

    if not isinstance(files_to_read, list):
        files_to_read = []

    retriever = HybridRetriever(repo_path=repo_path)

    # --- Strategy-aware file read budget ---
    # function-only: extract only the relevant function to minimize context
    # minimal: tight char budget to keep prompt within token limits
    # normal/default: standard full-file read budget
    if retry_strategy == "function-only":
        max_chars_modify = 8_000   # enough for one function
        max_chars_context = 3_000
        use_section_extract = True
    elif retry_strategy in ("minimal", "minimal_safe_fix", "fallback"):
        max_chars_modify = 6_000
        max_chars_context = 2_000
        use_section_extract = False
    else:
        max_chars_modify = 16_000  # original budget
        max_chars_context = 6_000
        use_section_extract = False

    print(f"   [RETRIEVAL] strategy={retry_strategy!r} max_modify={max_chars_modify} use_section_extract={use_section_extract}")

    # Read original files (full content for patching, capped per strategy)
    original_files = retriever.get_file_contents(
        files_to_modify,
        max_chars_per_file=max_chars_modify,
        truncate=False,
    )

    if not original_files:
        print("[ERROR] Could not read any files to modify")
        return {"status": "failure", "errors": "No target files readable from plan"}

    # Check that all planned files were found
    missing = [f for f in files_to_modify if f not in original_files]
    if missing:
        print(f"   [WARN] Missing files from plan: {missing}")
        plan = dict(plan)
        plan["files_to_modify"] = [f for f in files_to_modify if f in original_files]
        if not plan["files_to_modify"]:
            return {"status": "failure", "errors": f"None of the planned files exist: {missing}"}

    # For "function-only" strategy and large files, narrow the retrieval context
    # to the relevant section while preserving the full file content for patching.
    focused_files = {}
    if use_section_extract or any(len(content) > 10_000 for content in original_files.values()):
        issue_text = state.get("issue", "")
        plan_steps = plan.get("steps", [])
        for fpath, content in original_files.items():
            if use_section_extract or len(content) > 10_000:
                section = PatchGenerator._extract_relevant_section(
                    content, fpath, issue_text, plan_steps, max_chars=min(max_chars_modify, 8_000)
                )
                focused_files[fpath] = section if section and section.strip() else content
                print(f"   [RETRIEVAL] Focused {fpath}: {len(content)} -> {len(focused_files[fpath])} chars")
            else:
                focused_files[fpath] = content
    else:
        focused_files = dict(original_files)

    # Read context-only files (smaller limit)
    context_files = retriever.get_file_contents(
        files_to_read,
        max_chars_per_file=max_chars_context,
        truncate=True,
    )

    # Extract relevant sections from context files to reduce token usage
    issue_text = state.get("issue", "")
    plan_steps = plan.get("steps", [])
    relevant_context_parts = []
    
    for fpath, content in context_files.items():
        relevant_section = PatchGenerator._extract_relevant_section(
            content, fpath, issue_text, plan_steps, max_chars=3000
        )
        if relevant_section and relevant_section.strip():
            relevant_context_parts.append(f"=== CONTEXT FILE: {fpath} ===\n{relevant_section}")

    context = "\n\n".join(relevant_context_parts) if relevant_context_parts else "(no additional context files)"

    print(f"   [OK] Loaded {len(original_files)} target file(s), {len(context_files)} context file(s)")
    for fpath in original_files:
        print(f"      [modify] {fpath} ({len(original_files[fpath])} chars)")
    for fpath in context_files:
        print(f"      [read]   {fpath} ({len(context_files[fpath])} chars)")

    return {
        "plan": plan,
        "original_files": original_files,
        "focused_files": focused_files,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Node 3: Coder
# ---------------------------------------------------------------------------

def coder_node(state: AgentState) -> dict:
    """
    Generate unified diffs for all files in the plan.
    Consumes errors and previous_diffs from past attempts to avoid repeating.
    Raises RuntimeError explicitly when output is invalid so the graph records
    a traceable failure instead of propagating a silent empty dict.
    """
    retries = state.get("retries", 0)
    if not isinstance(retries, int) or retries < 0:
        retries = 0

    print("\n" + "=" * 60)
    print(f"[NODE] coder_node (attempt {retries + 1}/{MAX_RETRIES + 1})")
    print("=" * 60)

    # --- Input validation: fail fast and explicitly ---
    plan = state.get("plan", {})
    context = state.get("context", "")
    original_files = state.get("original_files", {})
    issue_text = state.get("issue", "")

    if not isinstance(plan, dict) or not plan:
        raise RuntimeError(
            "coder_node: received empty or invalid plan — cannot generate patch. "
            "Check planner_node output and retrieval_node pass-through."
        )

    if not isinstance(original_files, dict) or not original_files:
        raise RuntimeError(
            "coder_node: no original_files available — cannot generate patch. "
            "Check retrieval_node output."
        )

    if not isinstance(issue_text, str) or not issue_text.strip():
        raise RuntimeError(
            "coder_node: empty issue_text — cannot generate a meaningful patch."
        )

    patch_gen = PatchGenerator()

    try:
        result = patch_gen.generate_diffs(
            plan=plan,
            context=context if isinstance(context, str) else "",
            original_files=original_files,
            issue_text=issue_text,
            errors=state.get("errors", "") if isinstance(state.get("errors", ""), str) else "",
            previous_diffs=state.get("previous_diffs") if isinstance(state.get("previous_diffs"), dict) else None,
            repo_path=state.get("repo_path", ""),
            edge_cases=state.get("edge_cases", []) if isinstance(state.get("edge_cases", []), list) else [],
            qa_feedback=state.get("qa_feedback"),
            failure_type=state.get("failure_type"),
            failed_tests=state.get("failed_tests"),
            previous_patch=state.get("previous_patch"),
            retry_count=retries,
            retry_strategy=state.get("retry_strategy", "normal"),
            simulated_results=state.get("simulated_results"),
        )
    except Exception as e:
        print(f"   [ERROR] Coder failed with exception: {e}")
        result = None
        # Make sure the error message reflects the exception so feedback_node / next retry knows
        state["errors"] = (state.get("errors", "") or "") + f"\n- Exception in generate_diffs: {e}"

    if result is None:
        new_retries = retries + 1
        err_msg = (state.get("errors", "") or "") + "\n- Coder produced no valid diffs."
        if new_retries > MAX_RETRIES:
            # Raise explicitly so the graph records a named exception, not a silent empty state
            raise RuntimeError(
                f"Coder exhausted {new_retries} attempt(s) without producing diffs. "
                f"Last errors: {err_msg.strip()[-300:]}"
            )
        print(f"   [WARN] generate_diffs returned None — will retry ({new_retries}/{MAX_RETRIES})")
        return {
            "code_diffs": {},
            "patched_files": {},
            "errors": err_msg,
            "retries": new_retries,
            "status": "running",
            "failure_stage": "coder",
        }

    code_diffs, patched_files = result

    # Guard: if generate_diffs returned a result tuple but it's empty, treat as failure
    if not code_diffs or not patched_files:
        new_retries = retries + 1
        err_msg = (state.get("errors", "") or "") + "\n- Coder returned empty diffs/patched_files tuple."
        if new_retries > MAX_RETRIES:
            raise RuntimeError(
                f"Coder produced empty output after {new_retries} attempt(s). "
                f"Errors: {err_msg.strip()[-300:]}"
            )
        return {
            "code_diffs": {},
            "patched_files": {},
            "errors": err_msg,
            "retries": new_retries,
            "status": "running",
            "failure_stage": "coder",
        }

    return {
        "code_diffs": code_diffs,
        "patched_files": patched_files,
    }


# ---------------------------------------------------------------------------
# Node 4: Feedback
# ---------------------------------------------------------------------------

def feedback_node(state: AgentState) -> dict:
    """
    Validate the generated diffs and patched files.
    Decides: success (all pass), retry (has fixable errors), or failure (exhausted).

    Validation steps:
      1. Check that diffs were actually generated
      2. Run quality checks on each patched file
      3. Validate diff scope (files must be in plan)
      4. ALL files must pass -- no partial application

    Does NOT write files to disk (graph is pure).
    """
    print("\n" + "=" * 60)
    print("[NODE] feedback_node")
    print("=" * 60)

    code_diffs = state.get("code_diffs", {})
    patched_files = state.get("patched_files", {})
    original_files = state.get("original_files", {})
    plan = state.get("plan", {})
    retries = state.get("retries", 0)
    issue_text = state.get("issue", "")
    edge_cases = state.get("edge_cases", [])

    # Defensive type normalization
    if not isinstance(code_diffs, dict):
        code_diffs = {}
    if not isinstance(patched_files, dict):
        patched_files = {}
    if not isinstance(original_files, dict):
        original_files = {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(retries, int) or retries < 0:
        retries = 0
    if not isinstance(issue_text, str):
        issue_text = ""
    if not isinstance(edge_cases, list):
        edge_cases = []

    error_list: list[str] = []

    # -- Early termination: detect repeated identical errors --
    prev_errors = state.get("errors", "")
    _prev_key = prev_errors.strip().lower()[:120] if isinstance(prev_errors, str) else ""

    # -- Check 1: Were diffs generated? --
    if not code_diffs or not patched_files:
        error_list.append("No diffs or patched files were generated")
        error_summary = _summarize_errors(error_list)
        new_retries = retries + 1

        # Terminate early if same error repeats (deterministic failure)
        _cur_key = error_summary.strip().lower()[:120]
        if _prev_key and _cur_key == _prev_key:
            print("[WARN] Repeated identical error — inner graph cannot recover; bubbling to outer retry")
            return {
                "retries": new_retries,
                "errors": error_summary,
                "previous_diffs": code_diffs,
                "status": "failure",
                "failure_stage": "coder",
                "retry_strategy": state.get("retry_strategy", "normal"),
            }

        if new_retries > MAX_RETRIES:
            print("[ERROR] Max retries exhausted -- no valid diffs generated")
            return {"retries": new_retries, "errors": error_summary, "status": "failure", "failure_stage": "coder"}

        # Handle rate limits
        if "rate limit" in error_summary.lower() or "429" in error_summary:
            print("[ERROR] Rate limit exhausted — terminating pipeline")
            return {"retries": new_retries, "errors": error_summary, "status": "failure", "failure_stage": "coder"}

        print(f"   [WARN] No diffs generated -- will retry ({new_retries}/{MAX_RETRIES})")
        return {
            "retries": new_retries,
            "errors": error_summary,
            "previous_diffs": code_diffs,
            "status": "running",
            "retry_strategy": state.get("retry_strategy", "normal"),
        }

    # -- Check 2: Validate each patched file --
    planned_files = set(plan.get("files_to_modify", []))
    all_valid = True

    for fpath, patched in patched_files.items():
        print(f"\n   [CHECK] Validating: {fpath}")

        # 2a. File must be in the plan
        if fpath not in planned_files:
            error_list.append(f"{fpath}: modified but not in plan -- out of scope")
            all_valid = False
            continue

        # 2b. Must have original to compare
        original = original_files.get(fpath, "")
        if not original:
            error_list.append(f"{fpath}: no original source available")
            all_valid = False
            continue

        # 2c. Patched content must be a non-empty string
        if not isinstance(patched, str) or not patched.strip():
            error_list.append(f"{fpath}: patched content is empty or invalid")
            all_valid = False
            continue

        # 2d. Run quality checklist (syntax, not identical, no markdown, etc.)
        try:
            quality_ok, quality_failures = run_quality_checklist(
                original, patched, issue_text, edge_cases=edge_cases, file_path=fpath,
            )
        except Exception as e:
            error_list.append(f"{fpath}: quality check crashed -- {str(e)[:150]}")
            all_valid = False
            continue

        if not quality_ok:
            for qf in quality_failures:
                error_list.append(f"{fpath}: quality issue -- {qf}")
            all_valid = False
            continue

        target_symbols = []
        target_class_methods = []  # list of (class_name, method_name) for verification
        for target in plan.get("targets", []) if isinstance(plan.get("targets"), list) else []:
            if not isinstance(target, dict):
                continue
            if target.get("file") == fpath or target.get("path") == fpath:
                symbol = target.get("symbol")
                class_name = target.get("class_name", "")
                symbol_type = target.get("symbol_type", "")
                if isinstance(symbol, str) and symbol.strip():
                    target_symbols.append(symbol.strip())
                    if class_name and symbol_type in ("method", "function"):
                        target_class_methods.append((class_name.strip(), symbol.strip()))

        # Rule 2: Verify class-method targeting in patched file via AST
        if fpath.endswith(".py") and target_class_methods:
            import ast as _ast
            try:
                patched_tree = _ast.parse(patched)
                for target_cls, target_method in target_class_methods:
                    # Find the class in the AST
                    cls_found = False
                    method_found = False
                    for node in _ast.iter_child_nodes(patched_tree):
                        if isinstance(node, _ast.ClassDef) and node.name == target_cls:
                            cls_found = True
                            for child in _ast.iter_child_nodes(node):
                                if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                    if child.name == target_method:
                                        method_found = True
                                        break
                            break
                    if not cls_found:
                        error_list.append(
                            f"{fpath}: target class '{target_cls}' not found in patched output"
                        )
                        all_valid = False
                    elif not method_found:
                        error_list.append(
                            f"{fpath}: method '{target_method}' not found inside class '{target_cls}'"
                        )
                        all_valid = False
                if not all_valid:
                    continue
            except SyntaxError:
                pass  # already caught by quality_checklist

        full_ok, full_reason = LLMOutputValidator.validate_full_output(
            patched,
            original,
            filename=fpath,
        )
        if not full_ok:
            error_list.append(f"{fpath}: output validation -- {full_reason}")
            all_valid = False
            continue

        struct_ok, struct_reason = LLMOutputValidator.validate_structural_integrity(
            original,
            patched,
            fpath,
        )
        if not struct_ok:
            error_list.append(f"{fpath}: structural integrity -- {struct_reason}")
            all_valid = False
            continue

        import_ok, import_reason = LLMOutputValidator.validate_import_integrity(
            original,
            patched,
            fpath,
        )
        if not import_ok:
            error_list.append(f"{fpath}: import integrity -- {import_reason}")
            all_valid = False
            continue

        behavior_ok, behavior_reason = LLMOutputValidator.validate_behavioral_change(
            original,
            patched,
            fpath,
            target_symbols=target_symbols,
        )
        if not behavior_ok:
            error_list.append(f"{fpath}: behavioral change -- {behavior_reason}")
            all_valid = False
            continue

        scope_ok, scope_reason = LLMOutputValidator.validate_per_function_scope(
            original,
            patched,
            max_changed_lines_per_function=120,
        )
        if not scope_ok:
            error_list.append(f"{fpath}: patch scope -- {scope_reason}")
            all_valid = False
            continue

        # 2e. Check diff size is reasonable (threshold varies by language)
        diff = list(difflib.unified_diff(
            original.splitlines(), patched.splitlines(),
        ))
        deletions = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        additions = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        orig_lines = len(original.splitlines())

        print(f"      Diff: +{additions} -{deletions} lines")

        # Import here to avoid circular dependency
        import sys as _sys, os as _os
        _qc_path = _os.path.join(_os.path.dirname(__file__), "utils")
        if _qc_path not in _sys.path:
            _sys.path.insert(0, _qc_path)
        from quality_checker import detect_language
        lang = detect_language(fpath)

        # HTML/template files can legitimately change more lines (layout refactors)
        deletion_threshold = 0.85 if lang == "html" else 0.75

        if orig_lines > 0 and deletions > orig_lines * deletion_threshold:
            error_list.append(
                f"{fpath}: too many deletions ({deletions}/{orig_lines}) -- "
                f"likely a full rewrite, not a targeted fix"
            )
            all_valid = False
            continue

        print(f"      [OK] {fpath} passed all checks")

    # -- Decision --
    if all_valid and not error_list:
        print("\n   [OK] All files passed validation!")
        return {"status": "success"}

    # Atomic rejection -- if ANY file fails, reject all
    error_summary = _summarize_errors(error_list)
    new_retries = retries + 1

    print(f"\n   [FAIL] Validation failed for {len(error_list)} check(s)")
    for err in error_list:
        print(f"      - {err[:150]}")

    # Terminate early if error repeats (same root cause = deterministic failure)
    _cur_key = error_summary.strip().lower()[:120]
    if _prev_key and _cur_key == _prev_key:
        print(f"\n   [WARN] Repeated identical validation failure — inner graph cannot recover; bubbling to outer retry")
        return {
            "retries": new_retries,
            "errors": error_summary,
            "previous_diffs": code_diffs,
            "status": "failure",
            "failure_stage": "coder",
            "retry_strategy": state.get("retry_strategy", "normal"),
        }

    if new_retries > MAX_RETRIES:
        print(f"\n   [ERROR] Max retries exhausted ({MAX_RETRIES})")
        return {
            "retries": new_retries,
            "errors": error_summary,
            "previous_diffs": code_diffs,
            "status": "failure",
            "failure_stage": "coder",
        }

    # Handle rate limits
    if "rate limit" in error_summary.lower() or "429" in error_summary:
        print("[ERROR] Rate limit exhausted — terminating pipeline")
        return {
            "retries": new_retries,
            "errors": error_summary,
            "previous_diffs": code_diffs,
            "status": "failure",
            "failure_stage": "coder",
        }

    print(f"\n   [RETRY] Will retry (attempt {new_retries + 1}/{MAX_RETRIES + 1})")
    return {
        "retries": new_retries,
        "errors": error_summary,
        "previous_diffs": code_diffs,
        "status": "running",
        "retry_strategy": state.get("retry_strategy", "normal"),
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_retry(state: AgentState) -> str:
    """
    Conditional edge after feedback_node.
    Returns "retry" to loop back to coder_node, or "end" to finish.
    """
    status = state.get("status", "failure")

    if status == "success":
        print("\n[OK] Graph complete -- success!")
        return "end"

    if status == "planner_failed":
        print("\n[ERROR] Graph complete -- planner failed")
        return "end"

    if status == "failure":
        print("\n[ERROR] Graph complete -- max retries exhausted")
        return "end"

    # status == "running" means retry
    return "retry"


def after_planner(state: AgentState) -> str:
    """Route after planner: proceed to retrieval or abort."""
    if state.get("status") == "planner_failed":
        return "end"
        
    plan = state.get("plan")
    if plan and isinstance(plan, dict) and plan.get("plan_source") == "fallback":
        return "end"
        
    return "continue"


def after_retrieval(state: AgentState) -> str:
    """Route after retrieval: proceed to coder or abort."""
    if state.get("status") == "failure":
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Flow:
      START -> planner_node --+-> retrieval_node --+-> coder_node -> feedback_node
                              |                    |         ^            |
                              +-> END (if failed)  +-> END   +-- retry --+
                                                                  (or end)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner_node", planner_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("coder_node", coder_node)
    graph.add_node("feedback_node", feedback_node)

    # Entry point
    graph.set_entry_point("planner_node")

    # Conditional: planner -> retrieval (or END on failure)
    graph.add_conditional_edges(
        "planner_node",
        after_planner,
        {
            "continue": "retrieval_node",
            "end": END,
        },
    )

    # Conditional: retrieval -> coder (or END on failure)
    graph.add_conditional_edges(
        "retrieval_node",
        after_retrieval,
        {
            "continue": "coder_node",
            "end": END,
        },
    )

    # Linear: coder -> feedback
    graph.add_edge("coder_node", "feedback_node")

    # Conditional: feedback -> retry (coder) or END
    graph.add_conditional_edges(
        "feedback_node",
        should_retry,
        {
            "retry": "coder_node",
            "end": END,
        },
    )

    return graph.compile()
