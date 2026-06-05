"""
agents/planner.py

Strict JSON planner agent. Analyzes a GitHub issue + candidate files
and returns a structured plan with:
  - files_to_modify: list of file paths to change
  - files_to_read: list of file paths for context only
  - steps: list of concise fix steps

NO free text allowed. Output is validated JSON or execution stops.
"""

import json
import os
import re
from typing import Any

from groq import Groq

try:
    from ..config import GROQ_API_KEY
except ImportError:
    from config import GROQ_API_KEY

try:
    from .issue_grounding import (
        build_candidate_grounding_map,
        extract_issue_entities,
        normalize_symbol,
        symbol_matches,
    )
except ImportError:
    from issue_grounding import (
        build_candidate_grounding_map,
        extract_issue_entities,
        normalize_symbol,
        symbol_matches,
    )

client = Groq(api_key=GROQ_API_KEY)

_SYMBOL_TYPES = {"function", "method", "class", "module", "file"}
_NO_CHANGE_TOKEN = "NO_CHANGE_NEEDED"


def _is_no_change_plan(plan: dict) -> bool:
    if not isinstance(plan, dict):
        return False
    steps = plan.get("steps", [])
    return bool(
        isinstance(steps, list)
        and steps
        and isinstance(steps[0], str)
        and _NO_CHANGE_TOKEN in steps[0].upper()
    )


def _is_vague_text(text: str, min_len: int = 12) -> bool:
    if not isinstance(text, str):
        return True
    cleaned = " ".join(text.strip().split())
    if len(cleaned) < min_len:
        return True
    vague_markers = {
        "fix issue", "update logic", "make change", "handle bug", "adjust code",
        "improve behavior", "modify code", "fix behavior", "update function",
    }
    return cleaned.lower() in vague_markers


def _normalize_target_entry(entry: Any) -> dict | None:
    if not isinstance(entry, dict):
        return None
    file_path = _normalize_path(entry.get("file", "") or entry.get("path", ""))
    symbol = normalize_symbol(entry.get("symbol", "") or entry.get("target", ""))
    symbol_type = str(entry.get("symbol_type", "") or entry.get("type", "")).strip().lower()
    # Normalize legacy "function" to "method" when class_name is present
    class_name = str(entry.get("class_name", "") or "").strip()
    if symbol_type == "function" and class_name:
        symbol_type = "method"
    # Also extract class_name from dotted symbol (e.g. "ClassName.method_name")
    if not class_name and "." in symbol:
        parts = symbol.rsplit(".", 1)
        if len(parts) == 2 and parts[0][0:1].isupper():
            class_name = parts[0]
            symbol = parts[1]
            if symbol_type == "function":
                symbol_type = "method"
    language = str(entry.get("language", "")).strip().lower() or "unknown"
    why = " ".join(str(entry.get("why", "")).split())
    expected_behavior = " ".join(str(entry.get("expected_behavior", "") or entry.get("expected_change", "")).split())
    input_output = " ".join(str(entry.get("input_output", "")).split())
    if not file_path or not symbol or symbol_type not in _SYMBOL_TYPES:
        return None
    return {
        "file": file_path,
        "symbol": symbol,
        "class_name": class_name,  # empty string if standalone function
        "symbol_type": symbol_type,
        "language": language,
        "why": why,
        "expected_behavior": expected_behavior,
        "input_output": input_output,
    }


def _normalize_behavior_change(entry: Any) -> dict | None:
    if not isinstance(entry, dict):
        return None
    file_path = _normalize_path(entry.get("file", "") or entry.get("path", ""))
    symbol = normalize_symbol(entry.get("symbol", "") or entry.get("target", ""))
    before = " ".join(str(entry.get("before", "")).split())
    after = " ".join(str(entry.get("after", "")).split())
    if not file_path or not symbol or not before or not after:
        return None
    return {"file": file_path, "symbol": symbol, "before": before, "after": after}

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_plan(plan: dict) -> tuple[bool, str]:
    """
    Validate that the plan dict has the required structure.

    Args:
        plan: A dict expected to contain files_to_modify, files_to_read, steps.

    Returns:
        (is_valid, reason_string).
    """
    if not isinstance(plan, dict):
        return False, "Plan is not a dict"

    for key in ("files_to_modify", "files_to_read", "steps", "targets", "behavior_changes"):
        if key not in plan:
            return False, f"Missing key: {key}"

    if not isinstance(plan["files_to_modify"], list):
        return False, "files_to_modify must be a list"

    steps = plan.get("steps", [])
    if _is_no_change_plan(plan):
        return True, "OK"

    if len(plan["files_to_modify"]) == 0:
        return False, "files_to_modify must be a non-empty list"

    # Validate every entry is a non-empty string
    for i, f in enumerate(plan["files_to_modify"]):
        if not isinstance(f, str) or not f.strip():
            return False, f"files_to_modify[{i}] is not a valid string"

    if not isinstance(plan["files_to_read"], list):
        return False, "files_to_read must be a list"

    for i, f in enumerate(plan["files_to_read"]):
        if not isinstance(f, str):
            return False, f"files_to_read[{i}] is not a string"

    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        return False, "steps must be a non-empty list"

    for i, s in enumerate(plan["steps"]):
        if not isinstance(s, str) or not s.strip():
            return False, f"steps[{i}] is not a valid string"

    if not isinstance(plan["targets"], list) or not plan["targets"]:
        return False, "targets must be a non-empty list"

    normalized_targets = []
    for i, target in enumerate(plan["targets"]):
        normalized = _normalize_target_entry(target)
        if normalized is None:
            return False, f"targets[{i}] is missing required grounding fields"
        if _is_vague_text(normalized["why"]):
            return False, f"targets[{i}].why is too vague"
        if _is_vague_text(normalized["expected_behavior"]):
            return False, f"targets[{i}].expected_behavior is too vague"
        if "->" not in normalized["input_output"] and "=>" not in normalized["input_output"]:
            return False, f"targets[{i}].input_output must describe an input -> output mapping"
        normalized_targets.append(normalized)
    plan["targets"] = normalized_targets

    if not isinstance(plan["behavior_changes"], list) or not plan["behavior_changes"]:
        return False, "behavior_changes must be a non-empty list"

    normalized_changes = []
    for i, change in enumerate(plan["behavior_changes"]):
        normalized = _normalize_behavior_change(change)
        if normalized is None:
            return False, f"behavior_changes[{i}] is missing before/after behavior details"
        if normalized["before"].lower() == normalized["after"].lower():
            return False, f"behavior_changes[{i}] must describe a real behavior change"
        normalized_changes.append(normalized)
    plan["behavior_changes"] = normalized_changes

    # Rule 1: Verify steps reference at least one target (file or symbol)
    # A plan whose steps don't mention any target is generic/hallucinated
    target_refs = set()
    for t in normalized_targets:
        target_refs.add(t["symbol"].lower())
        # Also add the file basename without extension
        base = os.path.splitext(os.path.basename(t["file"]))[0].lower()
        if base:
            target_refs.add(base)

    steps_text = " ".join(s.lower() for s in plan["steps"])
    steps_reference_any_target = any(ref in steps_text for ref in target_refs if ref)

    if not steps_reference_any_target:
        return False, (
            f"Plan steps do not reference any target symbol or file. "
            f"Targets: {[t['symbol'] for t in normalized_targets]}. "
            f"Steps must explicitly mention the function/class/file being modified."
        )

    # Cap files_to_modify at MAX_FILES_TO_MODIFY (default 10)
    MAX_FILES = 10
    if len(plan["files_to_modify"]) > MAX_FILES:
        print(f"[WARN] Planner returned {len(plan['files_to_modify'])} files; capping at {MAX_FILES}. Consider increasing MAX_FILES_TO_MODIFY.")
        plan["files_to_modify"] = plan["files_to_modify"][:MAX_FILES]

    # Strip whitespace from file paths
    plan["files_to_modify"] = [f.strip() for f in plan["files_to_modify"]]
    plan["files_to_read"] = [f.strip() for f in plan["files_to_read"] if isinstance(f, str)]

    return True, "OK"


def _validate_json_output(raw: str) -> tuple[bool, str]:
    """
    Strict JSON validation: ensure output is clean, parseable JSON with no markdown.
    
    Returns:
        (is_valid_json, error_message)
    """
    if not isinstance(raw, str) or not raw.strip():
        return False, "Empty output"
    
    # Check for markdown code fences (should have been stripped)
    if "```" in raw:
        return False, "Output contains markdown code fences (```). Return ONLY raw JSON."
    
    # Try direct parsing
    try:
        json.loads(raw)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error at position {e.pos}: {e.msg}"


def _normalize_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    return path.replace("\\", "/").strip().strip("/")


def _extract_solution_files(solution) -> list[str]:
    if isinstance(solution, dict):
        return [str(path) for path in solution.keys()]
    if isinstance(solution, list):
        files = []
        for item in solution:
            if isinstance(item, dict):
                path = item.get("file_path")
            else:
                path = getattr(item, "file_path", None)
            if isinstance(path, str) and path.strip():
                files.append(path)
        return files
    return []


def _candidate_path_scores(candidate_files: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in candidate_files:
        if not isinstance(candidate, dict):
            continue
        path = _normalize_path(candidate.get("path", ""))
        if not path:
            continue
        try:
            score = float(candidate.get("combined_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores[path] = max(score, scores.get(path, float("-inf")))
    return scores


def _validate_issue_alignment(
    issue_text: str,
    plan_files: list[str],
    candidate_files: list[dict],
    plan_targets: list[dict] | None = None,
) -> tuple[bool, str]:
    if not isinstance(issue_text, str) or not issue_text.strip():
        return True, "No issue text to validate against"
    if not isinstance(plan_files, list) or not plan_files:
        return True, "No files to validate"
    if not isinstance(candidate_files, list) or not candidate_files:
        return True, "No candidates to validate against"

    issue_entities, grounding_map = build_candidate_grounding_map(candidate_files, issue_text)
    explicit_entities = (
        issue_entities.get("function_names", [])
        or issue_entities.get("class_names", [])
        or issue_entities.get("module_names", [])
        or issue_entities.get("file_paths", [])
    )

    for plan_file in plan_files:
        normalized = _normalize_path(plan_file)
        grounding = grounding_map.get(normalized)
        if grounding is None:
            return False, f"Selected file is not present in grounded candidate set: {plan_file}"

        if explicit_entities and not grounding.get("has_direct_symbol_match"):
            return False, (
                f"Selected file '{plan_file}' does not contain the issue symbols "
                f"{explicit_entities[:6]}"
            )

        if not grounding.get("has_direct_symbol_match") and len(grounding.get("matched_keywords", [])) < 2:
            return False, (
                f"Selected file '{plan_file}' has weak issue grounding "
                f"(matched keywords: {grounding.get('matched_keywords', [])[:5]})"
            )

    if isinstance(plan_targets, list):
        target_files = {
            _normalize_path(target.get("file", ""))
            for target in plan_targets
            if isinstance(target, dict)
        }
        missing = [path for path in plan_files if _normalize_path(path) not in target_files]
        if missing:
            return False, f"Plan files missing explicit grounded targets: {missing[:3]}"

    return True, "Selected files align with issue symbols and grounded keywords"


def _sanitize_plan(
    plan: dict,
    candidate_files: list[dict],
    repo_path: str = "",
) -> tuple[dict | None, str]:
    if not isinstance(plan, dict):
        return None, "Plan is not a dict"

    candidate_scores = _candidate_path_scores(candidate_files)
    candidate_paths = set(candidate_scores)
    if not candidate_paths:
        return None, "No candidate paths available for validation"

    repo_real = os.path.realpath(repo_path) if isinstance(repo_path, str) and repo_path.strip() else ""

    def _is_allowed_path(path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized or normalized not in candidate_paths:
            return False
        if not repo_real:
            return True
        full_path = os.path.realpath(os.path.join(repo_real, normalized))
        return full_path.startswith(repo_real) and os.path.isfile(full_path)

    def _dedupe(paths: list[str]) -> list[str]:
        seen = set()
        ordered = []
        for item in paths:
            normalized = _normalize_path(item)
            if normalized and normalized not in seen and _is_allowed_path(normalized):
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    files_to_modify = _dedupe(plan.get("files_to_modify", []))
    files_to_read = [
        path for path in _dedupe(plan.get("files_to_read", []))
        if path not in files_to_modify
    ]
    if not files_to_modify:
        return None, "Planner selected no valid target files from the candidate list"

    steps = []
    seen_steps = set()
    for step in plan.get("steps", []):
        if not isinstance(step, str) or not step.strip():
            continue
        cleaned = step.strip()
        key = cleaned.lower()
        if key not in seen_steps:
            seen_steps.add(key)
            steps.append(cleaned)

    if not steps:
        return None, "Planner returned no usable steps"

    targets = []
    for target in plan.get("targets", []):
        normalized = _normalize_target_entry(target)
        if normalized is None:
            continue
        if normalized["file"] not in files_to_modify:
            return None, f"Grounded target points outside files_to_modify: {normalized['file']}"
        targets.append(normalized)
    if not targets:
        return None, "Planner returned no valid grounded targets"

    behavior_changes = []
    for change in plan.get("behavior_changes", []):
        normalized = _normalize_behavior_change(change)
        if normalized is None:
            continue
        if normalized["file"] not in files_to_modify:
            return None, f"Behavior change points outside files_to_modify: {normalized['file']}"
        behavior_changes.append(normalized)
    if not behavior_changes:
        return None, "Planner returned no valid behavior change descriptions"

    missing_step_refs = [
        path for path in files_to_modify
        if not any(path in step or os.path.basename(path) in step for step in steps)
    ]
    if missing_step_refs:
        return None, f"Planner steps do not reference target file(s): {', '.join(missing_step_refs[:3])}"

    # ── Validate function references against actual candidate code ──
    candidate_lookup = {}
    # Build a set of known symbols from candidate snippets
    known_symbols: set[str] = set()
    for c in candidate_files:
        if not isinstance(c, dict):
            continue
        candidate_path = _normalize_path(c.get("path", ""))
        if candidate_path:
            candidate_lookup[candidate_path] = c
        structure = c.get("structure", {})
        if isinstance(structure, dict):
            for fn in structure.get("functions", []):
                if isinstance(fn, str):
                    known_symbols.add(normalize_symbol(fn).lower())
            for cls in structure.get("classes", []):
                if isinstance(cls, str):
                    known_symbols.add(normalize_symbol(cls).lower())
            for mod in structure.get("imports", []):
                if isinstance(mod, str):
                    known_symbols.add(normalize_symbol(mod).lower())
        # Also extract function names from snippet text as fallback
        snippet = c.get("snippet", "")
        if isinstance(snippet, str):
            for match in re.finditer(r'def\s+(\w+)', snippet):
                known_symbols.add(match.group(1).lower())
            for match in re.finditer(r'class\s+(\w+)', snippet):
                known_symbols.add(match.group(1).lower())

    # Check each step for hallucinated function references
    # SAFE_BUILTINS: common names that are always valid regardless of the codebase
    _SAFE_NAMES = {
        "self", "cls", "init", "__init__", "main", "print", "len", "range",
        "str", "int", "float", "list", "dict", "set", "tuple", "type",
        "open", "read", "write", "close", "get", "set", "add", "update",
        "insert", "delete", "remove", "append", "extend", "pop", "clear",
    }
    _CREATION_VERBS = {"add", "create", "implement", "introduce", "new", "write", "define", "build"}
    if known_symbols:
        validated_steps = []
        for step in steps:
            # Extract function-like references from the step text
            step_refs = re.findall(r'(?:function|def|method|class)\s+(\w+)', step, re.IGNORECASE)
            step_refs += re.findall(r'(\w+)\(\)', step)  # matches foo()
            # Filter out safe built-in names
            step_refs = [r for r in step_refs if r.lower() not in _SAFE_NAMES]

            if not step_refs:
                # No explicit symbol references — keep as-is (generic step)
                validated_steps.append(step)
                continue

            # If step uses creation verbs, unknown symbols are EXPECTED (being created)
            step_lower = step.lower()
            is_creation_step = any(verb in step_lower for verb in _CREATION_VERBS)
            if is_creation_step:
                validated_steps.append(step)
                continue

            hallucinated = [r for r in step_refs if r.lower() not in known_symbols]
            known_refs = [r for r in step_refs if r.lower() in known_symbols]

            if hallucinated and not known_refs:
                # Step references ONLY non-existent symbols — drop it entirely
                print(f"   [REJECT STEP] All referenced symbols are unknown: {hallucinated} — dropping hallucinated step")
                print(f"      Step: {step[:120]}")
                # Don't append — this step is dropped
            else:
                if hallucinated:
                    print(f"   [WARN] Step has some unknown symbol(s): {hallucinated} — keeping (has known refs {known_refs})")
                validated_steps.append(step)

        if not validated_steps:
            return None, (
                f"All planner steps referenced non-existent symbols and were rejected. "
                f"Known symbols in codebase: {sorted(known_symbols)[:15]}. "
                "Re-plan using only functions/classes that actually exist."
            )
        steps = validated_steps

    for target in targets:
        candidate = candidate_lookup.get(target["file"], {})
        structure = candidate.get("structure", {}) if isinstance(candidate, dict) else {}
        available_symbols = set()
        if isinstance(structure, dict):
            available_symbols.update(
                normalize_symbol(symbol) for symbol in structure.get("functions", []) if isinstance(symbol, str)
            )
            available_symbols.update(
                normalize_symbol(symbol) for symbol in structure.get("classes", []) if isinstance(symbol, str)
            )
            available_symbols.update(
                normalize_symbol(symbol) for symbol in structure.get("imports", []) if isinstance(symbol, str)
            )
        snippet = str(candidate.get("snippet", "") or "")
        target_exists = (
            symbol_matches(target["symbol"], available_symbols)
            or normalize_symbol(target["symbol"]).lower() in snippet.lower()
        )
        target_steps = [
            step for step in steps
            if target["file"] in step or os.path.basename(target["file"]) in step
        ]
        is_creation_target = any(
            any(verb in step.lower() for verb in _CREATION_VERBS)
            for step in target_steps
        )
        if not target_exists and not is_creation_target:
            return None, (
                f"Grounded target '{target['symbol']}' does not exist in {target['file']} "
                "and the plan does not clearly describe creating it."
            )

        if not any(
            change["file"] == target["file"]
            and normalize_symbol(change["symbol"]).lower() == normalize_symbol(target["symbol"]).lower()
            for change in behavior_changes
        ):
            return None, (
                f"Target '{target['symbol']}' in {target['file']} is missing a before/after behavior change mapping"
            )

    files_to_modify.sort(key=lambda path: candidate_scores.get(path, 0.0), reverse=True)
    files_to_read.sort(key=lambda path: candidate_scores.get(path, 0.0), reverse=True)

    return {
        "files_to_modify": files_to_modify,
        "files_to_read": files_to_read,
        "steps": steps,
        "targets": targets,
        "behavior_changes": behavior_changes,
    }, "OK"


def _repair_json_string(raw: str) -> str:
    """Apply common LLM output cleanups to make JSON parseable."""
    # Strip markdown code fences first
    raw = re.sub(r"```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    
    # Strip smart quotes
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")
    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw


def _extract_json_by_depth(text: str) -> str | None:
    """Use brace-depth tracking to extract the outermost JSON object."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_json_from_text(text: str) -> dict | None:
    """
    Try to extract a JSON object from LLM output that may contain wrapping text.
    Uses multiple strategies: direct parse, markdown fence extraction, and
    brace-depth scanning with auto-repair.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dict or None if no valid JSON found.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Strategy 0: Strip markdown fences first (aggressive approach)
    text_cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    text_cleaned = re.sub(r"\n?```\s*$", "", text_cleaned)
    
    # Strategy 1: Direct parse on original and cleaned versions
    for attempt_text in [text, text_cleaned]:
        try:
            result = json.loads(attempt_text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 2: Markdown fences (with variations)
    for pattern in [
        r"```(?:json)?\s*\n(\{.*?\})\s*\n```",  # With newlines
        r"```(?:json)?\s*(\{.*?\})\s*```",       # Compact format
        r"```\s*(\{.*\})\s*```"                   # Any fence
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass

    # Strategy 3: Brace-depth extraction (handles nested arrays/objects)
    candidate = _extract_json_by_depth(text)
    if candidate:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        # Strategy 4: Auto-repair then re-parse
        repaired = _repair_json_string(candidate)
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                print("   [INFO] JSON auto-repaired successfully.")
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _extract_paths_from_raw(raw: str) -> list[str]:
    """
    Extract file paths from raw LLM output even when JSON parsing fails.
    Looks for patterns like "path/to/file.py" in the text.
    Returns deduplicated list of paths in order of appearance.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    _CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".rb", ".cpp", ".c", ".cs"}

    # Match quoted file paths with known source extensions
    ext_pattern = "|".join(re.escape(e) for e in _CODE_EXTS)
    pattern = r'["\']([a-zA-Z0-9_/\\.-]+(?:' + ext_pattern + r'))["\']'
    matches = re.findall(pattern, raw)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        normalized = m.replace("\\", "/").strip().strip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise code analysis agent. Given a GitHub issue and candidate repository files, produce a JSON fix plan.

OUTPUT: ONLY a single raw JSON object. No prose, no markdown, no explanation.

{
  "files_to_modify": ["path/to/file.py"],
  "files_to_read": ["path/to/dependency.py"],
  "steps": [
    "Step 1: In file.py, function foo(), change X to Y to fix Z"
  ],
  "targets": [
    {
      "file": "path/to/file.py",
      "symbol": "foo",
      "class_name": "MyClass",
      "symbol_type": "method",
      "language": "python",
      "why": "foo contains the logic that currently produces the incorrect result",
      "expected_behavior": "update foo so the reported issue path produces the expected behavior",
      "input_output": "problematic input/state -> corrected output/state"
    }
  ],
  "behavior_changes": [
    {
      "file": "path/to/file.py",
      "symbol": "foo",
      "before": "the issue scenario currently returns/does the wrong thing",
      "after": "the issue scenario returns/does the expected thing"
    }
  ]
}

LANGUAGE DETECTION & GROUNDING (CRITICAL):
- Identify the PRIMARY LANGUAGE of the repository from candidate files (python, typescript, java, golang, etc.)
- Include "language" field in each target with the exact language of the target file (e.g., "python", "typescript", "java")
- For cross-language issues (e.g., Python backend + TypeScript frontend), identify files in each language separately
- Do NOT mix languages in files_to_modify — keep backend changes separate from frontend changes
- If issue affects multiple languages, plan modifications per language with appropriate file grouping

CROSS-LANGUAGE SYMBOL MATCHING:
- Extract from issue: function names, class names, API endpoints, domain keywords
- Match against candidate code:
  * Python: function/class names, module imports
  * TypeScript/JavaScript: function/const names, exported symbols, API handlers
  * Java: method/class names, package structure
  * Other: equivalent constructs in that language
- Validate that selected file CONTAINS the relevant symbol BEFORE planning changes
- Reject plan if no relevant file found OR no matching symbol in correct language

RULES:
- STRICT ALIGNMENT: You MUST ONLY fix the issue described. Do NOT refactor, format, or add unrelated features.
- If a candidate file is not directly related to the core logic of the issue, DO NOT modify it.
- files_to_modify: 1-10 files that MUST be changed. Be accurate — only include files where code actually changes.
- files_to_read: files needed for context only (imports, types, constants). Do NOT include files that need changes.
- steps: 1-10 concise steps. Each step must name the file, exact function/class, and exact change needed.
- targets: 1+ grounded targets. Every target MUST:
  * Name the exact file and function/class/module to modify
  * For methods inside a class: set "class_name" to the class name, "symbol" to the method name, "symbol_type" to "method"
  * For standalone functions: set "class_name" to "", "symbol" to the function name, "symbol_type" to "function"
  * For classes themselves: set "symbol" to the class name, "symbol_type" to "class"
  * Specify the language (python, typescript, java, etc.)
  * Explain why it is relevant to the issue
  * Include an explicit input -> output mapping (e.g., "None input -> default value", "empty list -> empty list")
- behavior_changes: 1+ entries describing the concrete before -> after behavior for each grounded target.
- Do NOT include test files in files_to_modify.
- Prefer modifying existing functions over creating new ones.
- If only 1 file needs changing, list only 1.
- Select the HIGHEST-SCORED candidates — they are ranked by semantic + keyword relevance + language match.
- Copy every file path exactly from the provided candidate list.
- Do not invent paths or rename directories.

CRITICAL — ANTI-HALLUCINATION:
- ONLY reference functions, classes, and variables that appear in the candidate code previews.
- Do NOT assume logic exists that is not shown in the previews.
- Reject vague plans. If you cannot name the exact function/class/module and describe the input -> output behavior change, do not guess.
- If the issue describes logic that does NOT exist in any candidate file, your step MUST say "add" or "create", not "change" or "fix".
- If the existing code already handles the issue correctly, return:
  {"files_to_modify": [], "files_to_read": [], "steps": ["NO_CHANGE_NEEDED: code already handles this correctly"], "targets": [], "behavior_changes": []}

ISSUE GROUNDING (CRITICAL — read before planning):
- Before writing ANY plan, cross-reference the issue text with the candidate code previews.
- If the issue mentions a function/class that does NOT appear in any candidate:
  * Do NOT plan to "fix" or "change" it — it does not exist.
  * Instead, determine what the issue ACTUALLY needs:
    Option A: The function needs to be CREATED (use "add" in your step).
    Option B: The issue is about SIMILAR existing functionality — plan changes to the CLOSEST matching code.
    Option C: The issue cannot be mapped to any real code — return NO_CHANGE_NEEDED.
- If the issue describes a behavior (e.g., "calculate_total returns wrong value") but no such function exists:
  * Search the candidates for the function that DOES handle that behavior.
  * Plan your fix against that REAL function, not the hallucinated name.
- NEVER plan a "refactor" or "rename" unless the issue explicitly asks for it.

IMPROVEMENT MODE (when core functionality already exists):
- First determine: does the requested feature/function already exist in the candidate code?
- If YES — do NOT re-implement it. Instead, plan improvements:
  * Security hardening (e.g., parameterized queries for SQL injection)
  * Input validation (only meaningful checks, NOT trivial ones like isinstance(x, object))
  * Error handling for real edge cases
  * Flexibility (e.g., making hard-coded values configurable)
- If NO — plan the implementation from scratch.
- NEVER add redundant validation that checks for conditions that cannot occur.
- NEVER add isinstance checks against base types (object, type) — they are always true and meaningless.

VALIDATION PHILOSOPHY:
- Prefer GRACEFUL HANDLING over raising exceptions. Skip invalid items or coerce types instead of crashing.
- Only raise exceptions for truly unrecoverable conditions (e.g., missing required config, database connection failure).
- Do NOT add strict type constraints the issue does not require (e.g., do not reject float when int is reasonable).
- Validation should PROTECT the logic, not restrict valid usage.

DOCUMENTATION ISSUES (CRITICAL):
- If the issue relates to documentation, glossary, reference docs, or user-facing descriptions, treat it as a documentation task.
- Modifying documentation means modifying static documentation files (e.g., .rst, .md), NEVER docs/conf.py unless specifically changing Sphinx configuration.
- Do NOT generate Python code or use runtime file generation (e.g., open(...).write(...)) to create documentation. Documentation must be static and version-controlled.
- For glossaries, use proper Sphinx format `.. glossary::` and define terms clearly in .rst files (e.g., glossary.rst).
- Ensure new documentation files (like glossary) are integrated into the documentation structure (referenced in an index or visible in navigation).
- Do NOT generate executable Python logic for documentation tasks.
"""


def plan_issue(
    issue_text: str,
    candidate_files: list[dict],
    qa_feedback: str = None,
    failure_type: str = None,
    failed_tests: int = None,
    repo_path: str = "",
    retry_strategy: str = "normal",
    max_retries: int = 0,
    previous_patch: dict | None = None,
    simulated_results: list | None = None,
) -> dict | None:
    """
    Generate a strict JSON plan for fixing an issue.

    Args:
        issue_text: The issue title + body.
        candidate_files: List of dicts with {path, snippet, structure} from retriever.
        max_retries: Number of retry attempts if JSON parsing fails. Must be >= 0.

    Returns:
        Validated plan dict or None if planning fails entirely.

    Raises:
        TypeError: If issue_text is not a string or candidate_files is not a list.
        ValueError: If issue_text is empty or candidate_files is empty.
    """
    # --- Input validation ---
    if not isinstance(issue_text, str):
        raise TypeError(f"issue_text must be a string, got {type(issue_text).__name__}")
    if not issue_text.strip():
        raise ValueError("issue_text cannot be empty")

    if not isinstance(candidate_files, list):
        raise TypeError(f"candidate_files must be a list, got {type(candidate_files).__name__}")
    if not candidate_files:
        raise ValueError("candidate_files cannot be empty")

    if not isinstance(max_retries, int) or max_retries < 0:
        max_retries = 0

    if not isinstance(repo_path, str):
        repo_path = ""
    if not isinstance(retry_strategy, str) or not retry_strategy.strip():
        retry_strategy = "normal"

    strategy_limits = {
        "normal": 50,
        "function-only": 20,
        "minimal": 12,
        "simplified": 25,
        "force_append": 30,
        "minimal_safe_fix": 15,
        "fallback": 12,
    }
    candidate_limit = strategy_limits.get(retry_strategy, 25)
    candidate_subset = candidate_files[:candidate_limit]
    issue_entities, candidate_grounding = build_candidate_grounding_map(candidate_subset, issue_text)

    # Build candidate summary for the LLM
    # For large candidate sets (>10), use compressed one-line format to stay within prompt limits
    candidate_summary = ""
    use_compressed = len(candidate_subset) > 10
    for i, c in enumerate(candidate_subset, 1):
        if not isinstance(c, dict):
            continue

        path = c.get("path", "(unknown)")
        score = c.get("combined_score", 0.0)
        language = c.get("language", "unknown")
        structure = c.get("structure", {})
        if not isinstance(structure, dict):
            structure = {}

        functions = structure.get("functions", [])
        classes = structure.get("classes", [])
        grounding = candidate_grounding.get(_normalize_path(path), {})
        if not isinstance(functions, list):
            functions = []
        if not isinstance(classes, list):
            classes = []

        if use_compressed:
            # Compressed: one line per candidate — path + language + top function names only
            fns = ", ".join(str(f) for f in functions[:6]) or "(none)"
            grounding_note = ""
            matched_symbols = (
                grounding.get("matched_functions", [])[:2]
                + grounding.get("matched_classes", [])[:2]
                + grounding.get("matched_modules", [])[:2]
            )
            if matched_symbols:
                grounding_note = f" | grounded: {', '.join(matched_symbols[:4])}"
            elif grounding.get("matched_keywords"):
                grounding_note = f" | keywords: {', '.join(grounding.get('matched_keywords', [])[:4])}"
            candidate_summary += f"[{i}] {path} [{language}] | score: {score} | fns: {fns}{grounding_note}\n"
        else:
            fns = ", ".join(str(f) for f in functions[:8]) or "(none)"
            cls = ", ".join(str(cl) for cl in classes[:5]) or "(none)"
            snippet_preview = str(c.get("snippet", ""))[:300]
            grounding_lines = ""
            grounded_symbols = (
                grounding.get("matched_functions", [])[:3]
                + grounding.get("matched_classes", [])[:3]
                + grounding.get("matched_modules", [])[:3]
            )
            if grounded_symbols:
                grounding_lines += f"    Grounded symbols: {', '.join(grounded_symbols[:6])}\n"
            elif grounding.get("matched_keywords"):
                grounding_lines += f"    Grounded keywords: {', '.join(grounding.get('matched_keywords', [])[:6])}\n"
            candidate_summary += (
                f"\n[{i}] Path: {path}\n"
                f"    Language: {language}\n"
                f"    Score: {score}\n"
                f"    Functions: {fns}\n"
                f"    Classes: {cls}\n"
                f"{grounding_lines}"
                f"    Preview:\n{snippet_preview}\n"
            )

    if not candidate_summary.strip():
        print("[ERROR] No valid candidates to plan with")
        return None

    try:
        # Load long-term memory for intelligent planner contextualization
        import sys, os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        from integration.memory_store import search_memory
        similar_cases = search_memory(issue_text, top_k=2)
        if similar_cases:
            candidate_summary += "\n\nPAST SUCCESSFUL RESOLUTIONS (Use as reference for patterns and test cases):\n"
            for k, sc in enumerate(similar_cases):
                candidate_summary += f"[History {k+1}] Issue: {sc.get('issue', '')[:200]}...\n"
                candidate_summary += f"  Files changed in solution: {_extract_solution_files(sc.get('solution'))}\n"
    except Exception as e:
        print(f"Skipping memory integration: {e}")

    user_prompt = (
        f"[ISSUE]\n{issue_text}\n\n"
        f"[EXTRACTED ISSUE ENTITIES]\n{json.dumps(issue_entities, ensure_ascii=False)}\n\n"
        f"[CANDIDATE FILES]\nNote: Each candidate includes a [language] tag. Use this to ensure cross-language correctness.\n{candidate_summary}\n\n"
        f"[LANGUAGE-AWARE PLANNING]\n"
        f"- Identify the primary language(s) in the repository from the candidates\n"
        f"- For each target file, specify its language in the 'language' field (e.g., python, typescript, java)\n"
        f"- Do NOT select files purely by semantic match — ensure the language matches the issue context\n"
        f"- For multi-language issues, group changes by language\n\n"
    )


    if qa_feedback:
        user_prompt += f"[PREVIOUS FAILURE]\n{qa_feedback}\n\n"
    if failure_type:
        user_prompt += f"[FAILURE TYPE]\n{failure_type}\n\n"
    if failed_tests is not None:
        user_prompt += f"[FAILED TESTS COUNT]\n{failed_tests}\n\n"

    if simulated_results and isinstance(simulated_results, list):
        failed_details = []
        for tr in simulated_results:
            if isinstance(tr, dict) and tr.get("status") == "fail":
                failed_details.append(f"- {tr.get('test_name', 'unknown')}: {tr.get('reasoning', '')}")
        
        if failed_details:
            user_prompt += (
                "[SIMULATED TEST FAILURES (CRITICAL EDGE CASES)]\n"
                "The previous patch failed the following behavioral tests. Your plan MUST explicitly include steps to handle these edge cases:\n"
                + "\n".join(failed_details)
                + "\n\n"
            )

    if previous_patch and isinstance(previous_patch, dict):
        file_summaries = []
        for path, diff in previous_patch.items():
            if not isinstance(path, str) or not isinstance(diff, str):
                continue
            line_count = len(diff.splitlines())
            file_summaries.append(f"{path} ({line_count} diff lines)")
        if file_summaries:
            user_prompt += (
                "[PREVIOUS FAILED PATCH SUMMARY]\n"
                "The prior patch attempted changes in the following files and must not be repeated exactly:\n"
                + "\n".join(file_summaries)
                + "\n\n"
            )

    # ── Strategy-aware retry guidance ──
    # Each retry_strategy level injects progressively different instructions
    # so the planner genuinely changes its approach instead of repeating.
    if retry_strategy == "function-only":
        user_prompt += (
            "[RETRY GUIDANCE — FUNCTION ONLY]\n"
            "Use the most precise, narrow plan possible.\n"
            "- Target a single function or method in one file.\n"
            "- Keep files_to_read minimal and only include the function's surrounding context.\n"
            "- Do not propose changes across multiple files.\n"
            "- Output the exact function or method name in the plan step.\n\n"
        )
    elif retry_strategy == "minimal":
        user_prompt += (
            "[RETRY GUIDANCE — MINIMAL PATCH]\n"
            "Produce the smallest reasonable fix.\n"
            "- Only one file in files_to_modify.\n"
            "- Only one concrete step in steps.\n"
            "- Avoid broad refactors or multiple changes.\n"
            "- Use the simplest patch that addresses the issue.\n\n"
        )
    elif retry_strategy == "force_append":
        user_prompt += (
            "[RETRY GUIDANCE — FORCE APPEND]\n"
            "Previous plan was rejected. The issue requires NEW functionality.\n"
            "- Do NOT attempt to modify existing functions in-place.\n"
            "- Instead, explicitly instruct the coder to CREATE / ADD a new function or class.\n"
            "- Example step: 'Add a new function validate_input() to handle edge cases.'\n\n"
        )
    elif retry_strategy == "minimal_safe_fix":
        user_prompt += (
            "[RETRY GUIDANCE — MINIMAL SAFE FIX]\n"
            "Multiple retries failed. Use MINIMAL safe scope:\n"
            "- Target only ONE file — the highest-scored candidate.\n"
            "- Plan only ONE concrete step.\n"
            "- Focus on the single most impactful, non-destructive change.\n"
            "- Do not delete any existing functionality.\n"
            "- If you cannot identify a real code gap, return NO_CHANGE_NEEDED.\n\n"
        )
    elif retry_strategy == "fallback":
        user_prompt += (
            "[RETRY GUIDANCE — FALLBACK / NO-CHANGE]\n"
            "All previous approaches failed. You have two options:\n"
            "Option 1: Plan the absolute simplest fix to ONE function in ONE file.\n"
            "Option 2: If the issue cannot be mapped to any real code gap, return:\n"
            '  {"files_to_modify": [], "files_to_read": [], "steps": ["NO_CHANGE_NEEDED: issue does not map to any actionable code gap"]}\n\n'
        )
    else:
        user_prompt += f"[RETRY STRATEGY]\n{retry_strategy}\n\n"

    if qa_feedback:
        user_prompt += "INSTRUCTION:\nUpdate the plan to fix the previous failure. Do NOT repeat the same approach. Produce the JSON plan now.\n\n"
    else:
        user_prompt += "Produce the JSON plan now.\n\n"
    
    user_prompt += (
        "JSON OUTPUT REQUIREMENTS (CRITICAL):\n"
        "1. Return ONLY valid JSON. No markdown code fences, no prose before/after.\n"
        "2. Each target in 'targets' MUST include: file, symbol, symbol_type, language, why, expected_behavior, input_output\n"
        "3. Ensure language field is set correctly (python, typescript, java, etc.)\n"
        "4. All strings must be properly quoted and escaped.\n"
        "5. No trailing commas in arrays or objects.\n"
        "6. Strip all markdown formatting from the output."
    )

    # Track per-attempt failure reasons so each retry prompt is unique
    attempt_errors: list[str] = []
    # Preserve planner intent: store file paths extracted from raw output
    # even when JSON parsing fails, so the fallback doesn't discard them.
    _planner_identified_files: list[str] = []

    for attempt in range(max_retries + 1):
        attempt_label = f"(attempt {attempt + 1}/{max_retries + 1})"
        print(f"[PLAN] Planning issue fix {attempt_label}...")

        # ── Build progressive prompt with accumulated error context ──
        current_user_prompt = user_prompt

        # Inject previous attempt failures so the LLM never sees the same input twice
        if attempt_errors:
            error_summary = "\n".join(f"  Attempt {i+1}: {e}" for i, e in enumerate(attempt_errors))
            current_user_prompt += (
                f"\n\n[PREVIOUS ATTEMPT FAILURES]\n{error_summary}\n"
                "You MUST avoid these mistakes. Output ONLY raw JSON."
            )

        # Progressive strictness
        if attempt == 1:
            current_user_prompt += (
                "\n\nCRITICAL: You MUST output ONLY valid JSON. "
                "No extra text, no explanation, no markdown backticks. "
                "Start your response with { and end with }."
            )
        elif attempt == 2:
            current_user_prompt += (
                "\n\nCRITICAL: Parsing failed again. Output a MINIMAL plan: "
                "one file in files_to_modify, empty files_to_read, one step. "
                "Choose the HIGHEST-SCORED candidate file. Pure JSON only."
            )
        elif attempt >= 3:
            # Give the LLM one last chance with extreme constraint
            # Prefer planner-identified files over blind retrieval
            top_path = ""
            if _planner_identified_files:
                top_path = _planner_identified_files[0]
            else:
                for c in candidate_subset:
                    if isinstance(c, dict) and c.get("path"):
                        top_path = c["path"]
                        break
            current_user_prompt += (
                f'\n\nLAST CHANCE. Return exactly this structure with real values:\n'
                f'{{"files_to_modify": ["{top_path}"], '
                f'"files_to_read": [], '
                f'"steps": ["Step 1: In {os.path.basename(top_path)}, modify the exact relevant function/class to fix the issue behavior"], '
                f'"targets": [{{"file": "{top_path}", "symbol": "{os.path.splitext(os.path.basename(top_path))[0]}", "symbol_type": "module", "why": "This file is the highest-confidence grounded candidate for the issue", "expected_behavior": "Change the relevant logic so the issue scenario behaves correctly", "input_output": "issue trigger -> corrected result"}}], '
                f'"behavior_changes": [{{"file": "{top_path}", "symbol": "{os.path.splitext(os.path.basename(top_path))[0]}", "before": "The issue scenario behaves incorrectly", "after": "The issue scenario behaves correctly"}}]}}'
            )

        # ── Temperature escalation: avoid deterministic repetition ──
        temp = min(0.1 * attempt, 0.3)

        try:
            import sys, os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, base_dir)
            from utils.llm_utils import safe_invoke
            
            response = safe_invoke(
                client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": current_user_prompt},
                ],
                temperature=temp,
            )
            raw_content = response.choices[0].message.content
            raw = raw_content.strip() if isinstance(raw_content, str) else ""
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "quota" in err_str or "exhausted" in err_str or "token" in err_str:
                print(f"   [WARN] LLM quota exhausted in planner! Skipping retries. Error: {e}")
                # Stop retrying when the LLM is unavailable; do not fall back to heuristic-only planning.
                break
            
            err = f"LLM call failed: {e}"
            print(f"   [WARN] {err}")
            attempt_errors.append(err)
            continue

        if not raw:
            err = "Empty LLM output"
            print(f"   [FAIL] Planner returned empty output {attempt_label}")
            attempt_errors.append(err)
            continue

        # ── Extract planner intent from raw output (even if JSON is broken) ──
        # This preserves the planner's file targets for the fallback.
        raw_paths = _extract_paths_from_raw(raw)
        if raw_paths:
            # Keep unique, in order, prioritizing earlier (higher-confidence) attempts
            for p in raw_paths:
                if p not in _planner_identified_files:
                    _planner_identified_files.append(p)
            print(f"   [INFO] Planner-identified files so far: {_planner_identified_files[:5]}")

        # ── Parse JSON (brace-depth extraction + auto-repair) ──
        plan = _extract_json_from_text(raw)

        if plan is None:
            err = f"JSON parse failed (raw length={len(raw)}, starts with: {raw[:80]!r})"
            print(f"   [FAIL] {err} {attempt_label}")
            attempt_errors.append(err)
            continue

        # ── Validate schema ──
        valid, reason = _validate_plan(plan)
        if not valid:
            err = f"Schema validation: {reason}"
            print(f"   [FAIL] {err} {attempt_label}")
            attempt_errors.append(err)
            continue

        # ── Sanitize against candidate list ──
        sanitized_plan, sanitize_reason = _sanitize_plan(plan, candidate_subset, repo_path=repo_path)
        if sanitized_plan is None:
            err = f"Target validation: {sanitize_reason}"
            print(f"   [FAIL] {err} {attempt_label}")
            attempt_errors.append(err)
            continue

        # ── Validate alignment with issue intent ──
        aligned, alignment_reason = _validate_issue_alignment(
            issue_text,
            sanitized_plan["files_to_modify"],
            candidate_subset,
            sanitized_plan.get("targets", []),
        )
        if not aligned:
            err = f"Issue alignment: {alignment_reason}"
            print(f"   [FAIL] {err} {attempt_label}")
            attempt_errors.append(err)
            continue

        # ── Success ──
        plan = sanitized_plan
        print(f"   [OK] Plan generated successfully")
        print(f"   Files to modify: {plan['files_to_modify']}")
        print(f"   Files to read: {plan['files_to_read']}")
        print(f"   Steps: {len(plan['steps'])}")
        print(f"   Targets: {len(plan.get('targets', []))}")
        for step in plan["steps"]:
            print(f"      - {step}")

        return plan

    print("[ERROR] All planner attempts failed. No valid plan could be generated.")
    return None
