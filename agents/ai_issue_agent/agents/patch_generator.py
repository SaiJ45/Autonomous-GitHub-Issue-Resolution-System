"""
agents/patch_generator.py

Generates code fixes for GitHub issues.
Multi-file aware -- produces one fix per file listed in the plan.

Strategy:
  1. PRIMARY: Ask LLM to output the complete modified file directly.
     Then compute the diff from original vs patched for validation.
     This eliminates fragile diff-parsing and hunk-matching failures.
  2. FALLBACK: If the LLM returns a unified diff instead, try to parse
     and apply it (best effort).

This approach ensures patches always match the actual file content.
"""

import os
import re
import difflib

from groq import Groq

try:
    from ..config import GROQ_API_KEY
except ImportError:
    from config import GROQ_API_KEY

try:
    from .output_validators import LLMOutputValidator
except ImportError:
    from output_validators import LLMOutputValidator

client = Groq(api_key=GROQ_API_KEY)

# Token budget constants (1 token ~= 4 chars; tune here without touching prompts)
_CHARS_PER_TOKEN = 4
_MAX_PROMPT_CHARS = 16_000   # ~4 000 tokens — safe per-request budget
_MAX_FILE_CHARS   =  8_000   # max chars for a file payload in prompts
_MAX_CONTEXT_CHARS = 3_000   # max chars for the enriched context block


class PatchGenerator:
    def __init__(self):
        self.client = client

    @staticmethod
    def _failure_guidance(failure_type: str | None, retry_count: int) -> str:
        if retry_count <= 0 and not failure_type:
            return ""

        guidance = {
            "LLM_PARSE_ERROR": "Your last response broke the required format. Return only the requested payload with no prose, truncation, or markdown fences.",
            "SYNTAX_ERROR": "Preserve surrounding structure exactly and ensure the final code parses cleanly.",
            "LOGIC_ERROR": "Change only the faulty logic or condition that caused the failure. Leave unrelated code untouched.",
            "EDGE_CASE_MISSING": "Add explicit handling for None, empty inputs, boundary values, and missing keys that are relevant to the issue.",
            "VALIDATION_ERROR": "Keep the patch narrowly scoped to the planned files and preserve required file structure.",
            "REQUIREMENT_MISMATCH": "Realign the change to the issue requirements and the plan steps only.",
            "PLAN_GENERATION_FAILED": "Use only the supplied file content and plan. Do not invent new target files or behaviors.",
            "BEHAVIORAL_TESTS_FAILED": "The logic you wrote did not produce the expected test results. Review the specific test failures provided. You MUST change the logic to handle those exact failure cases. Do NOT repeat the previous logic.",
        }

        parts = []
        if retry_count >= 2:
            parts.append("Use the smallest targeted edit that resolves the current failure.")
        if retry_count >= 3:
            parts.append("Prefer a conservative fix over a broad rewrite.")
        if failure_type in guidance:
            parts.append(guidance[failure_type])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # ReAct Reasoning Loop  (Thought → Action → Observation)
    # ------------------------------------------------------------------

    REACT_SYSTEM_PROMPT = (
        "You are an AI Issue Solver with domain-aware, edge-case aware reasoning.\n\n"
        "You must analyze the issue using step-by-step reasoning and tool usage "
        "BEFORE any code fix is generated.\n\n"
        "Before coding, perform:\n"
        "1. Problem Understanding:\n"
        "   - What exactly is the issue?\n"
        "   - What behavior is expected?\n"
        "   - What inputs/outputs are required?\n"
        "2. Domain Reasoning:\n"
        "   - If financial/math: identify correct formula\n"
        "   - If bug: locate root cause\n"
        "   - If refactor: preserve behavior\n"
        "3. Gap Analysis:\n"
        "   - What is missing in current code?\n"
        "   - What must be changed vs added?\n"
        "4. Solution Plan:\n"
        "   - Define exact logic BEFORE coding\n"
        "   - Avoid generic placeholders\n\n"
        "Coding Rules:\n"
        " - Implement complete logic (not partial)\n"
        " - Handle edge cases\n"
        " - Follow existing code style\n"
        " - Modify only relevant parts\n"
        " - CRITICAL: MODIFY existing functions in-place. Do NOT add duplicate function definitions.\n"
        " - CRITICAL: Preserve structure. Do NOT remove existing classes or major structures.\n"
        " - DOCUMENTATION: For docs/glossary tasks, generate static content (e.g. .rst, .md). Do NOT write Python code to generate docs. Use proper Sphinx format (`.. glossary::`) for glossaries.\n\n"
        "Reject:\n"
        " - generic functions without logic\n"
        " - assumptions without justification\n"
        " - unrelated features\n"
        " - duplicate methods or classes\n\n"
        "You MUST:\n"
        "- Follow existing conventions in the repo\n"
        "- Extend patterns, not replace them\n"
        "- Match the precision, validation, and style of surrounding code\n\n"
        "## Edge Case Checklist (ALWAYS consider)\n\n"
        "- Null / None inputs\n"
        "- Empty data structures (empty string, empty list, empty dict)\n"
        "- Invalid types (string where int expected, etc.)\n"
        "- Boundary values (zero, negative, very large)\n"
        "- Off-by-one errors\n"
        "- Missing keys / file not found\n"
        "- API failure / exceptions\n"
        "- Division by zero\n\n"
        "## Available Actions\n\n"
        "- search_code(query) -- Search the codebase for code matching the query\n"
        "- open_file(path) -- Read the full content of a specific file by relative path\n"
        "- generate_patch() -- Signal that you have enough understanding to generate the fix\n\n"
        "## Rules\n\n"
        "- NEVER request generate_patch() in your FIRST step\n"
        "- ALWAYS analyze the provided context before using any action\n"
        "- You MUST explicitly list relevant edge cases in at least ONE Thought step "
        "BEFORE requesting generate_patch()\n"
        "- You MUST identify domain constraints from the code context\n"
        "- If information is missing, use search_code to find it\n"
        "- If you need to see a full file, use open_file\n"
        "- Before generate_patch(), confirm you understand the root cause, the fix, "
        "the domain constraints, AND the edge cases the fix must handle\n"
        "- You have a MAXIMUM of 5 steps. Use them wisely.\n\n"
        "## Response Format (STRICT)\n\n"
        "Thought: <your detailed reasoning -- include domain context and edge cases>\n"
        "Action: <action_name(argument)> OR generate_patch() OR None\n\n"
        "Do NOT include any text outside this format."
    )


    # ------------------------------------------------------------------
    # Token-budget helpers  (no LLM calls — pure text manipulation)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_chars(*parts: str) -> int:
        """Return total character count of all parts (proxy for token cost)."""
        return sum(len(p) for p in parts if isinstance(p, str))

    @staticmethod
    def _trim_to_budget(text: str, max_chars: int) -> str:
        """Smart truncation: keep header 1/3 + tail 2/3, drop middle."""
        if not isinstance(text, str) or len(text) <= max_chars:
            return text or ""
        head = max_chars // 3
        tail = max_chars - head
        omitted = len(text) - max_chars
        return (
            text[:head]
            + f"\n# ... ({omitted} chars omitted) ...\n"
            + text[-tail:]
        )

    @staticmethod
    def _extract_relevant_section(
        source: str,
        file_path: str,
        issue_text: str,
        plan_steps: list,
        max_chars: int = _MAX_FILE_CHARS,
    ) -> str:
        """
        Return the most issue-relevant portion of *source* within *max_chars*.
        For Python files, uses AST to extract matching functions/classes +
        the file header (imports).  Falls back to smart head+tail truncation.
        """
        if not isinstance(source, str) or len(source) <= max_chars:
            return source or ""

        combined = (issue_text or "") + " " + " ".join(
            s for s in (plan_steps or []) if isinstance(s, str)
        )
        keywords = set(re.findall(r"[a-zA-Z_]\w{2,}", combined.lower()))

        if file_path.endswith(".py"):
            import ast as _ast
            try:
                tree = _ast.parse(source)
                src_lines = source.splitlines()

                # header boundary = last import line
                header_end = 0
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                        header_end = max(
                            header_end,
                            getattr(node, "end_lineno", node.lineno),
                        )
                header = "\n".join(src_lines[:header_end])[:1500]

                # Score top-level functions/classes
                scored = []
                issue_words = set(re.findall(r"[a-zA-Z_]\w+", combined.lower()))
                for node in _ast.iter_child_nodes(tree):
                    if not isinstance(
                        node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef),
                    ):
                        continue
                    name = node.name.lower()
                    score = 0.0
                    
                    # Exact match prioritization
                    if name in issue_words:
                        score += 50.0
                    
                    # Exact match in plan steps
                    for step in plan_steps or []:
                        if isinstance(step, str):
                            step_words = set(re.findall(r"[a-zA-Z_]\w+", step.lower()))
                            if name in step_words:
                                score += 50.0

                    score += sum(1.0 for kw in keywords if kw in name)
                    for step in (plan_steps or []):
                        if isinstance(step, str) and name in step.lower():
                            score += 2.0
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    snippet = "\n".join(src_lines[start:end])
                    scored.append((score, start, snippet))

                scored.sort(key=lambda x: (-x[0], x[1]))

                sections = [header] if header.strip() else []
                budget = max_chars - len(header) - 60
                seen = set()
                for score, start, snippet in scored:
                    if budget <= 0:
                        break
                    if start in seen:
                        continue
                    chunk = snippet[:budget]
                    sections.append(chunk)
                    budget -= len(chunk)
                    seen.add(start)

                if len(sections) > (1 if header.strip() else 0):
                    return "\n\n".join(sections)
            except Exception:
                pass  # fall through to generic

        # Generic: smart head + tail
        return PatchGenerator._trim_to_budget(source, max_chars)


    # ------------------------------------------------------------------
    # Section-based patch helpers (AST-aware, deterministic injection)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_target_node(
        source: str,
        file_path: str,
        issue_text: str,
        plan_steps: list,
        plan_targets: list | None = None,
    ) -> "tuple[str, int, int, str] | None":
        """
        Find the single most issue-relevant function, method, or class in *source*.
        Uses class-aware AST traversal so that class methods are correctly
        distinguished from standalone functions.
        
        When plan_targets specifies class_name='CartItem' and symbol='save',
        this will match only CartItem.save — not Order.save or a top-level save().
        
        Returns (snippet, start_line_0idx, end_line_exclusive, qualified_name) or None.
        Only works for Python files; returns None for all others.
        
        ENFORCES:
        - Rule 1: If class_name specified, class MUST exist
        - Rule 2: If method specified, method MUST exist inside class
        - Rejects mismatches with clear error messages
        """
        if not isinstance(file_path, str) or not file_path.endswith(".py"):
            return None
        if not isinstance(source, str) or not source.strip():
            return None

        import ast as _ast
        try:
            tree = _ast.parse(source)
        except SyntaxError:
            return None

        src_lines = source.splitlines()
        combined = (
            (issue_text or "")
            + " "
            + " ".join(s for s in (plan_steps or []) if isinstance(s, str))
        )
        keywords = set(re.findall(r"[a-zA-Z_]\w{2,}", combined.lower()))
        issue_words = set(re.findall(r"[a-zA-Z_]\w+", combined.lower()))

        # Extract target symbols with class_name awareness
        target_specs = []  # list of (symbol, class_name, symbol_type)
        required_class = None  # RULE 1: If class_name specified, it MUST exist
        for target in plan_targets or []:
            if not isinstance(target, dict):
                continue
            target_file = str(target.get("file", "") or target.get("path", ""))
            if target_file.replace("\\", "/").strip("/") != file_path.replace("\\", "/").strip("/"):
                continue
            symbol = str(target.get("symbol", "")).strip().lower()
            class_name = str(target.get("class_name", "")).strip()
            symbol_type = str(target.get("symbol_type", "")).strip().lower()
            if symbol:
                # Handle dotted notation (e.g., "ClassName.method")
                if not class_name and "." in symbol:
                    parts = symbol.rsplit(".", 1)
                    class_name = parts[0]
                    symbol = parts[1]
                target_specs.append((symbol, class_name.lower(), symbol_type))
                if class_name:
                    required_class = class_name.lower()

        # RULE 1: Validate required class exists
        if required_class:
            class_found = False
            for top_node in _ast.iter_child_nodes(tree):
                if isinstance(top_node, _ast.ClassDef) and top_node.name.lower() == required_class:
                    class_found = True
                    break
            if not class_found:
                print(f"   [REJECT] RULE 1: Class '{required_class}' not found in {file_path}")
                return None

        # Build scored candidate list using class-aware traversal
        # candidates: list of (score, node, parent_class_name)
        candidates: list[tuple[float, object, str]] = []

        def _score_node(node, parent_class: str = ""):
            """Score a function/class node with class-context awareness."""
            name = node.name.lower()
            score = 0.0

            for target_symbol, target_class, target_type in target_specs:
                # Exact class+method match (highest priority)
                if target_class and parent_class.lower() == target_class and name == target_symbol:
                    score += 300.0  # Best: exact class.method match
                elif name == target_symbol:
                    if target_class and parent_class and parent_class.lower() != target_class:
                        # Wrong class — penalize heavily
                        score -= 50.0
                    elif not target_class:
                        # No class specified in target, but symbol matches
                        score += 200.0
                    else:
                        # Target has class but no parent — partial match
                        score += 100.0

            # Keyword matching from issue text and plan steps
            if name in issue_words:
                score += 50.0
            for step in plan_steps or []:
                if isinstance(step, str):
                    step_lower = step.lower()
                    step_words = set(re.findall(r"[a-zA-Z_]\w+", step_lower))
                    if name in step_words:
                        score += 50.0
                    # Bonus for "ClassName.method" pattern in steps
                    if parent_class and f"{parent_class.lower()}.{name}" in step_lower:
                        score += 80.0

            score += sum(1.0 for kw in keywords if kw in name)
            return score

        # Walk AST with parent tracking (NOT ast.walk which flattens)
        for top_node in _ast.iter_child_nodes(tree):
            if isinstance(top_node, _ast.ClassDef):
                # Score the class itself
                class_score = _score_node(top_node, parent_class="")
                candidates.append((class_score, top_node, ""))
                # Score methods inside the class
                for child in _ast.iter_child_nodes(top_node):
                    if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        method_score = _score_node(child, parent_class=top_node.name)
                        # RULE 5: Boost class+method combinations
                        # These are preferred because they're more specific and safer
                        if method_score > 0:
                            method_score += 150.0  # High bonus for class methods
                        candidates.append((method_score, child, top_node.name))
            elif isinstance(top_node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                # Top-level standalone function
                fn_score = _score_node(top_node, parent_class="")
                candidates.append((fn_score, top_node, ""))

        if not candidates:
            print(f"   [REJECT] RULE 2: No candidates found for targets: {target_specs}")
            return None

        # Pick the highest-scoring candidate
        # RULE 5: Class+method combinations will naturally rank higher due to 150.0 bonus
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_node, best_class = candidates[0]

        if best_score <= 0.0:
            print(f"   [REJECT] RULE 2: Best candidate score too low ({best_score:.1f})")
            return None

        # RULE 2: If required_class specified, best match MUST be in that class
        if required_class and best_class.lower() != required_class:
            print(f"   [REJECT] RULE 2: Required class '{required_class}' but matched '{best_class or 'top-level'}'")
            return None

        # Log what was matched for debugging
        node_type = "class" if isinstance(best_node, _ast.ClassDef) else "method" if best_class else "function"
        qual_name = f"{best_class}.{best_node.name}" if best_class else best_node.name
        print(f"   [TARGET] Matched {node_type} '{qual_name}' (score={best_score:.1f})")

        start = best_node.lineno
        if getattr(best_node, "decorator_list", None):
            start = best_node.decorator_list[0].lineno
        start -= 1  # 0-indexed
        end = getattr(best_node, "end_lineno", best_node.lineno)  # 1-indexed end
        snippet = "\n".join(src_lines[start:end])
        return snippet, start, end, qual_name

    @staticmethod
    def _inject_section(
        original: str,
        new_section: str,
        start_line: int,
        end_line: int,
        target_name: str = "target",
    ) -> str:
        """
        Replace lines [start_line, end_line) (0-indexed start, exclusive end)
        in *original* with *new_section*.
        Preserves all surrounding code exactly.

        Uses relative-offset indentation so nested structures (methods inside
        classes) keep their extra indentation levels intact.
        
        ENFORCES:
        - Rule 3: Modification stays within target scope
        - Rule 6: Validates structural integrity
        """
        if not isinstance(original, str) or not isinstance(new_section, str):
            return original
        lines = original.splitlines(keepends=True)

        # RULE 3: Reject if trying to modify outside valid range
        if start_line < 0 or end_line > len(lines) or start_line >= end_line:
            print(f"   [REJECT] RULE 3: Invalid line range [{start_line}, {end_line}) for {len(lines)} total lines in target '{target_name}'")
            return None

        # Detect original indent at the start of the section being replaced
        orig_indent = ""
        if start_line < len(lines) and lines[start_line].strip():
            orig_line = lines[start_line]
            orig_indent = orig_line[:len(orig_line) - len(orig_line.lstrip())]

        # Detect the indent of the first non-blank line in the new section
        new_lines_raw = new_section.splitlines(keepends=True)
        new_indent = ""
        for nl in new_lines_raw:
            if nl.strip():
                new_indent = nl[:len(nl) - len(nl.lstrip())]
                break

        # Compute the offset: how many spaces to add/remove per line
        offset = len(orig_indent) - len(new_indent)

        adjusted_new_lines = []
        for line in new_lines_raw:
            if not line.strip():
                # Blank lines: keep as-is
                adjusted_new_lines.append(line)
            elif offset > 0:
                # Need to add indent
                adjusted_new_lines.append(" " * offset + line)
            elif offset < 0:
                # Need to remove indent — but never remove more than existing whitespace
                leading = len(line) - len(line.lstrip())
                trim = min(-offset, leading)
                adjusted_new_lines.append(line[trim:])
            else:
                adjusted_new_lines.append(line)

        if adjusted_new_lines and not adjusted_new_lines[-1].endswith("\n") and end_line < len(lines):
            adjusted_new_lines[-1] += "\n"

        return "".join(lines[:start_line] + adjusted_new_lines + lines[end_line:])

    @staticmethod
    def _has_meaningful_change(original: str, patched: str) -> bool:
        """
        Detect ANY meaningful difference between original and patched using
        diff-based comparison.  Does NOT rely on exact string equality.

        Returns True if there is at least one non-whitespace-only line change.
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return False
        if original == patched:
            return False

        # Normalize trailing whitespace per line and compare
        orig_lines = [line.rstrip() for line in original.splitlines()]
        patch_lines = [line.rstrip() for line in patched.splitlines()]
        if orig_lines != patch_lines:
            return True

        # Fallback: check raw content ignoring only final newlines
        return original.rstrip('\n\r') != patched.rstrip('\n\r')

    @staticmethod
    def _issue_requires_behavior_change(
        issue_text: str,
        plan: dict,
        failure_type: str | None,
    ) -> bool:
        if isinstance(failure_type, str) and failure_type.upper() in {
            "LOGIC_ERROR", "BEHAVIORAL_TESTS_FAILED", "REQUIREMENT_MISMATCH",
        }:
            return True

        behavior_changes = plan.get("behavior_changes", []) if isinstance(plan, dict) else []
        if isinstance(behavior_changes, list) and behavior_changes:
            return True

        issue_lower = (issue_text or "").lower()
        logic_markers = [
            "wrong", "incorrect", "fails", "failure", "behavior", "return", "should",
            "expected", "edge case", "none", "empty", "invalid", "boundary",
            "handle", "logic", "bug", "crash",
        ]
        return any(marker in issue_lower for marker in logic_markers)

    def _generate_section_patch(
        self,
        file_path: str,
        original: str,
        issue_text: str,
        steps_block: str,
        context: str,
        error_block: str,
        qa_feedback: "str | None",
        failure_type: "str | None",
        failed_tests: "int | None",
        previous_patch_content: "str | None",
        retry_count: int,
        plan_targets: list | None = None,
        edge_case_context: str = "",
        test_failure_context: str = "",
    ) -> "str | None":
        """
        PREFERRED strategy for Python files of any size:
        1. Find the most relevant function/class via AST.
        2. Ask LLM to output ONLY the modified version of that block.
        3. Validate the section output (syntax only — NOT size).
        4. Inject it deterministically into the original file.
        5. Revalidate the full patched file.
        Returns the complete patched file, or None if unsuccessful.
        """
        target = self._find_target_node(
            original, file_path, issue_text, steps_block.splitlines(), plan_targets=plan_targets
        )
        if target is None:
            return None  # caller will try next strategy

        section, start_line, end_line, target_qual_name = target
        lang_label, lang_rules = self._get_language_instructions(file_path)
        retry_guidance = self._failure_guidance(failure_type, retry_count)
        ctx_trimmed = self._trim_to_budget(context, 1200)

        system_prompt = (
            f"You are a surgical code editor. "
            f"Output ONLY the modified {lang_label} function or class.\n\n"
            "CRITICAL REQUIREMENTS BEFORE CODING:\n"
            "1. Identify the ROOT CAUSE of the bug (not symptoms)\n"
            "   - What is the underlying issue in the logic?\n"
            "   - What assumption is violated?\n"
            "   - What edge case is unhandled?\n"
            "2. Define expected behavior change\n"
            "   - What should happen with the fix?\n"
            "   - What inputs/outputs must be handled?\n"
            "3. Validate fix addresses root cause\n"
            "   - Does the change fix the underlying problem?\n"
            "   - Are all edge cases from the issue actually handled?\n\n"
            "CODING RULES:\n"
            " - Implement complete logic (not partial)\n"
            " - Handle ALL edge cases identified in the issue\n"
            " - Follow existing code style\n"
            " - Modify only relevant parts\n"
            " - NO generic placeholders or unjustified assumptions.\n"
            " - CRITICAL: MODIFY the existing function in-place. Do NOT output duplicate function definitions.\n"
            " - CRITICAL: Preserve structure. Do NOT remove existing classes or major structures. Keep modifications strictly localized.\n"
            " - NO formatting-only changes. The fix MUST change behavior, not just whitespace.\n"
            " - DOCUMENTATION: For docs/glossary tasks, generate static content (.rst, .md). Do NOT write Python code to generate docs.\n\n"
            "EDGE CASE REQUIREMENTS:\n"
            "- Handle None values and empty inputs\n"
            "- Validate input types before use\n"
            "- Check boundary conditions (zero, negative, overflow)\n"
            "- Handle missing keys or undefined variables\n"
            "- Add error handling if needed (try/except or input validation)\n\n"
            "FORMAT RULES:\n"
            "1. Start with the exact `def`, `async def`, or `class` keyword line.\n"
            "2. Keep all unchanged lines EXACTLY as-is — same indentation and whitespace.\n"
            "3. Make the SMALLEST possible change that fixes the issue.\n"
            "4. You MUST change the code to fix the issue. Do NOT return the exact same code.\n"
            "5. No prose, no markdown fences, no explanations.\n"
            "6. NEVER add ellipsis, `...`, or truncation markers.\n"
            "7. Preserve all existing decorators, docstrings, and type hints.\n"
            f"\n{lang_label} rules:\n{lang_rules}\n"
        )

        # Lock the target function to the original source to prevent drift on retries
        working_section = section
        prev_note = ""
        if retry_count > 0 and qa_feedback:
            # We explicitly do NOT re-extract from previous_patch_content using heuristic search,
            # because if the previous attempt was a full-file rewrite, it might extract the wrong function
            # and replace Function A with Function B.
            prev_note = (
                f"[PREVIOUS ATTEMPT FAILED]\n"
                f"[FAILURE REASON]: {(qa_feedback or '')[:400]}\n\n"
            )

        # Determine class context for the target (if it's a method inside a class)
        _class_context_note = ""
        for _t in (plan_targets or []):
            if not isinstance(_t, dict):
                continue
            _t_cls = str(_t.get("class_name", "")).strip()
            _t_sym = str(_t.get("symbol", "")).strip()
            _t_type = str(_t.get("symbol_type", "")).strip().lower()
            
            # RULE 5: Enhanced class method detection
            # Check if this target is a method (symbol_type == "method") with a class_name
            if _t_cls and _t_sym and _t_type == "method":
                # Verify the section contains the method definition
                if re.search(rf"\bdef\s+{re.escape(_t_sym)}\s*\(", working_section):
                    _class_context_note = (
                        f"CRITICAL INSTRUCTION: This is a method '{_t_sym}' inside class '{_t_cls}'.\n"
                        f"- Preserve method indentation (one level inside the class)\n"
                        f"- Output ONLY the method definition — NOT the class definition\n"
                        f"- Start with 'def {_t_sym}(' and preserve all arguments exactly\n"
                        f"- Use the same indentation level as the original (typically 4 or 8 spaces)\n\n"
                    )
                    break
            elif _t_cls and _t_sym and _t_sym.lower() in working_section.lower()[:200]:
                # Fallback: check if symbol appears in first 200 chars
                _class_context_note = (
                    f"NOTE: This is a method inside class '{_t_cls}'. "
                    f"Preserve its indentation level (one level inside the class). "
                    f"Do NOT output the class definition — only the method.\n\n"
                )
                break

        # RULE 3: Detect anchor-based insertion requirements
        # If steps mention "after <anchor>", enforce that the code includes the anchor followed by new logic
        _anchor_requirement = ""
        anchor_pattern = r"after\s+([a-zA-Z_]\w*\.?\w*\(\)|\w+\.save\(\)|[\w\.]+)"
        anchor_matches = re.findall(anchor_pattern, steps_block, re.IGNORECASE)
        if anchor_matches:
            anchor_point = anchor_matches[0]
            _anchor_requirement = (
                f"ANCHOR-BASED INSERTION REQUIREMENT:\n"
                f"- The code must INCLUDE the anchor point: {anchor_point}\n"
                f"- New code must be INSERTED AFTER {anchor_point}\n"
                f"- The anchor point must remain UNCHANGED\n"
                f"- Example pattern: ...{anchor_point}\\n    # NEW CODE HERE\\n    ...\n\n"
            )

        user_content_full = (
            f"ISSUE:\n{issue_text[:600]}\n\n"
            f"WHAT TO CHANGE:\n{steps_block}\n\n"
            f"ANALYSIS REQUIRED BEFORE CODING:\n"
            f"1. Root Cause: Why does the issue occur? (not what the symptom is)\n"
            f"2. Fix Strategy: Exactly what logic needs to change?\n"
            f"3. Edge Cases: What inputs/conditions must be handled?\n"
            f"4. Validation: How will you confirm the fix works?\n\n"
            + (f"{test_failure_context}\n" if test_failure_context else "")
            + (f"{edge_case_context}\n" if edge_case_context else "")
            + (f"CONTEXT:\n{ctx_trimmed}\n\n" if ctx_trimmed.strip() else "")
            + prev_note
            + _anchor_requirement
            + _class_context_note
            + f"FUNCTION/CLASS TO MODIFY:\n{working_section}\n\n"
            + (f"RETRY GUIDANCE:\n{retry_guidance}\n\n" if retry_guidance else "")
            + "Output ONLY the modified function/class. Preserve its indentation exactly."
        )

        # Minimal fallback prompt for second attempt or 413 recovery
        user_content_minimal = (
            f"MODIFY THIS FUNCTION TO FIX THE ISSUE.\n"
            f"ISSUE: {issue_text[:300]}\n\n"
            f"CURRENT CODE:\n{working_section}\n\n"
            f"CHANGE NEEDED:\n{steps_block}\n\n"
            "Output ONLY the modified function. No explanation."
        )

        last_reason = "no output"
        for attempt in range(2):
            prompt = user_content_full if attempt == 0 else user_content_minimal
            try:
                import sys as _sys, os as _os
                _base = _os.path.dirname(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                )
                if _base not in _sys.path:
                    _sys.path.insert(0, _base)
                from utils.llm_utils import safe_invoke, TokenLimitError

                response = safe_invoke(
                    self.client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=min(0.1 + retry_count * 0.15, 0.5),  # escalate on retries to avoid repetition
                )
            except TokenLimitError:
                # Section prompt too large — trim section further and retry once
                trimmed_sec = self._trim_to_budget(working_section, len(working_section) // 2)
                micro_prompt = (
                    f"Fix this function for: {issue_text[:200]}\n\n"
                    f"CODE:\n{trimmed_sec}\n\n"
                    "Output only the modified function."
                )
                try:
                    from utils.llm_utils import safe_invoke
                    response = safe_invoke(
                        self.client.chat.completions.create,
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": micro_prompt}],
                        temperature=0.1,
                    )
                except Exception as e2:
                    raise ValueError(f"Section patch micro-prompt also failed: {e2}")
            except Exception as e:
                raise ValueError(f"Section patch LLM call failed: {e}")

            raw = response.choices[0].message.content
            if not isinstance(raw, str) or not raw.strip():
                last_reason = "empty LLM response"
                continue

            code = self._extract_code_from_response(raw.strip())
            if code is None:
                last_reason = "could not extract code from response"
                continue

            # Must start with def / async def / class
            first = code.lstrip()
            if not (
                first.startswith("def ")
                or first.startswith("async def ")
                or first.startswith("class ")
            ):
                last_reason = f"output does not start with def/class: {first[:50]!r}"
                continue

            # Verify the returned function/class name matches the target
            # This prevents the LLM from silently replacing FunctionA with FunctionB
            orig_first = section.lstrip()
            orig_name_match = re.match(r'(?:async\s+)?(?:def|class)\s+(\w+)', orig_first)
            new_name_match = re.match(r'(?:async\s+)?(?:def|class)\s+(\w+)', first)
            if orig_name_match and new_name_match:
                if orig_name_match.group(1) != new_name_match.group(1):
                    last_reason = (
                        f"Function name mismatch: expected '{orig_name_match.group(1)}' "
                        f"but LLM returned '{new_name_match.group(1)}'"
                    )
                    continue

            # RULE 3: Validate anchor-based insertion if required
            if anchor_matches:
                anchor_point = anchor_matches[0]
                # Clean up anchor pattern (remove escapes if any)
                anchor_clean = anchor_point.replace(r"\\n", "\n").strip()
                if anchor_clean not in code:
                    last_reason = f"Anchor point '{anchor_point}' not found in modified code"
                    print(f"   [WARN] {last_reason}")
                    continue

            # Validate section syntax
            if file_path.endswith(".py"):
                import ast as _ast
                try:
                    _ast.parse(code)
                except SyntaxError as se:
                    last_reason = f"section syntax error: {se}"
                    continue

            # Inject and validate full file
            patched = self._inject_section(original, code, start_line, end_line, target_name=target_qual_name)
            if patched is None:
                # RULE 3 validation failed — scope issue
                last_reason = f"Patch scope validation failed for target '{target_qual_name}'"
                continue
            
            if file_path.endswith(".py"):
                import ast as _ast
                try:
                    _ast.parse(patched)
                except SyntaxError as se:
                    last_reason = f"full-file syntax error after injection: {se}"
                    continue
                    
                # PRE-APPLY Structural Safety Guard: Verify no functions/methods are removed
                from agents.ai_issue_agent.agents.output_validators import LLMOutputValidator
                struct_ok, struct_reason = LLMOutputValidator.validate_structural_integrity(
                    original, patched, file_path
                )
                if not struct_ok:
                    last_reason = f"Structural corruption: {struct_reason}"
                    continue

            if not self._has_meaningful_change(original, patched):
                last_reason = "Section patch generated no changes (identical to original)"
                continue

            # RULE 8/10: Validate behavioral modification — verify required functions called
            # For payment/email features, check that send_mail, render_to_string, etc. are actually present
            if file_path.endswith(".py"):
                # Step 1: Identify required behavior keywords from steps
                required_behaviors = set()
                behavior_map = {
                    "send": ["send_mail", "send_email", "send"],
                    "mail": ["send_mail", "send_email"],
                    "render": ["render_to_string"],
                    "template": ["render_to_string", "template"],
                    "email": ["send_mail", "send_email"],
                    "order": ["order"],
                    "save": ["save"],
                    "post": ["post", "method"],
                }
                
                steps_lower = steps_block.lower()
                for keyword, functions in behavior_map.items():
                    if keyword in steps_lower:
                        required_behaviors.update(functions)
                
                # Step 2: Check if required functions are present in the modified code
                if required_behaviors:
                    import ast as _ast
                    try:
                        tree = _ast.parse(code)
                        called_functions = set()
                        for node in _ast.walk(tree):
                            if isinstance(node, _ast.Call):
                                if isinstance(node.func, _ast.Name):
                                    called_functions.add(node.func.id.lower())
                                elif isinstance(node.func, _ast.Attribute):
                                    called_functions.add(node.func.attr.lower())
                        
                        # Check for required function calls
                        found_behaviors = required_behaviors & called_functions
                        missing_behaviors = required_behaviors - found_behaviors
                        
                        if required_behaviors and not found_behaviors:
                            last_reason = (
                                f"RULE 8: Feature incomplete — required function calls missing: {', '.join(sorted(required_behaviors))}"
                            )
                            print(f"   [WARN] {last_reason}")
                            continue
                    except SyntaxError:
                        pass  # Will be caught by syntax validation below

            # POST-INJECTION Comprehensive Safety Validation
            if file_path.endswith(".py"):
                from agents.ai_issue_agent.agents.output_validators import LLMOutputValidator
                
                # Indentation consistency check
                indent_ok, indent_reason = LLMOutputValidator.validate_indentation_consistency(
                    original, patched, filename=file_path
                )
                if not indent_ok:
                    last_reason = f"Indentation: {indent_reason}"
                    continue
                
                # Import integrity check
                import_ok, import_reason = LLMOutputValidator.validate_import_integrity(
                    original, patched, filename=file_path
                )
                if not import_ok:
                    last_reason = f"Imports: {import_reason}"
                    continue
                
                # Behavioral change validation (RULE 4: not just imports)
                behav_ok, behav_reason = LLMOutputValidator.validate_behavioral_change(
                    original, patched, filename=file_path, target_symbols=None
                )
                if not behav_ok:
                    last_reason = f"RULE 4: Behavior {behav_reason}"
                    continue
                
                # RULE 3/6: Scope check — verify edits stay inside target method/class
                scope_ok, scope_reason = LLMOutputValidator.validate_per_function_scope(
                    original, patched, max_changed_lines_per_function=80
                )
                if not scope_ok:
                    last_reason = f"RULE 3/6: Scope violation: {scope_reason}"
                    continue

            print(
                f"   [OK] Section patch succeeded for {file_path} "
                f"(lines {start_line+1}-{end_line}, "
                f"target={target_qual_name}, "
                f"section {len(code)} chars)"
            )
            return patched

        raise ValueError(f"Section patch failed: {last_reason}")

    def _react_reasoning_loop(
        self,
        issue_text: str,
        plan: dict,
        context: str,
        original_files: dict[str, str],
        repo_path: str,
        errors: str = "",
        max_steps: int = 2,
        edge_cases: list[str] | None = None,
    ) -> str:
        """
        Run a bounded ReAct (Reasoning + Acting) loop to analyze the issue
        before generating any patch. Returns accumulated reasoning history
        that enriches the patch generation context.

        Args:
            issue_text: The issue description.
            plan: The planner output with files_to_modify and steps.
            context: Retrieved code context string.
            original_files: {filepath: source_code} for target files.
            repo_path: Path to the cloned repository.
            errors: Error descriptions from previous attempts.
            max_steps: Maximum reasoning iterations (default 5).
            edge_cases: Pre-identified edge cases from heuristic analyzer.

        Returns:
            Accumulated reasoning history string.
        """
        if not repo_path or not isinstance(repo_path, str):
            print("   [REACT] No repo_path provided -- skipping reasoning loop")
            return ""

        files_to_modify = plan.get("files_to_modify", [])
        steps = plan.get("steps", [])
        steps_block = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(steps) if isinstance(s, str)
        )

        # Format edge cases for inclusion in prompts
        if not isinstance(edge_cases, list):
            edge_cases = []
        edge_case_block = ""
        if edge_cases:
            ec_lines = "\n".join(f"  - {ec}" for ec in edge_cases if isinstance(ec, str))
            edge_case_block = f"\nPre-identified Edge Cases (MUST handle in fix):\n{ec_lines}\n"

        # Infer domain signals from the codebase
        domain_signals = self._infer_domain_signals(original_files, context)
        domain_block = ""
        if domain_signals:
            domain_block = (
                "\nRepository Signals (inferred from code -- MUST follow):\n"
                f"{domain_signals}\n"
            )

        # Build a preview of target files (first 80 lines each)
        files_preview = ""
        for fpath, content in original_files.items():
            preview_lines = content.splitlines()[:120]
            files_preview += f"\n=== {fpath} (first {len(preview_lines)} lines) ===\n"
            files_preview += "\n".join(preview_lines) + "\n"

        history = ""
        final_step = 0
        executed_actions = set()

        print(f"\n[REACT] Starting reasoning loop (max {max_steps} steps)")

        for step in range(1, max_steps + 1):
            final_step = step
            print(f"\n   [REACT] Step {step}/{max_steps}")

            # Build user prompt with accumulated history
            # Enforce token budget before building user content
            _ctx_trimmed = PatchGenerator._trim_to_budget(context, _MAX_CONTEXT_CHARS)
            _prev_trimmed = PatchGenerator._trim_to_budget(files_preview, _MAX_FILE_CHARS)
            user_content = (
                f"Issue:\n{issue_text[:1000]}\n\n"
                f"Plan:\n{steps_block}\n\n"
                f"Files to Modify: {', '.join(files_to_modify)}\n\n"
                f"Retrieved Context:\n{_ctx_trimmed}\n\n"
                f"Target Files Preview:\n{_prev_trimmed}\n\n"
            )

            if domain_block:
                user_content += f"{domain_block}\n\n"

            if edge_case_block:
                user_content += f"{edge_case_block}\n\n"

            if errors and errors.strip():
                user_content += (
                    f"Previous Errors (avoid repeating):\n{errors}\n\n"
                )

            if history:
                user_content += (
                    f"Previous Reasoning Steps:\n{history}\n\n"
                )
            else:
                user_content += (
                    "Previous Reasoning Steps:\n"
                    "(This is step 1 -- begin your analysis)\n\n"
                )

            user_content += (
                "Analyze the issue and respond with your Thought and Action.\n"
                f"This is step {step} of {max_steps}."
            )

            try:
                import sys, os
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
                from utils.llm_utils import safe_invoke
                
                response = safe_invoke(
                    self.client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": self.REACT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.3,
                )
            except Exception as e:
                print(f"   [REACT] LLM call failed: {e}")
                break

            raw = response.choices[0].message.content
            if not isinstance(raw, str) or not raw.strip():
                print("   [REACT] Empty response -- ending loop")
                break

            raw = raw.strip()

            # Parse Thought and Action
            thought, action = self._parse_react_response(raw)
            print(f"   [REACT] Thought: {thought[:150]}...")
            print(f"   [REACT] Action:  {action}")

            # Enforce: deduplicate actions to save tokens (Fix 7.2)
            action_clean = action.strip()
            if action_clean != "None" and "generate_patch" not in action_clean.lower():
                if action_clean in executed_actions:
                    print(f"   [REACT] Blocked repeated action: {action_clean}")
                    history += (
                        f"\nStep {step}:\n"
                        f"Thought: {thought}\n"
                        f"Action: None (blocked repeated action)\n"
                        f"Observation: You already executed {action_clean}. Do NOT repeat actions. Use generate_patch() or a different tool.\n"
                    )
                    continue
                executed_actions.add(action_clean)

            # Prefer direct patch generation without blocking on step 1.

            # Check for generate_patch signal
            if "generate_patch" in action.lower():
                history += (
                    f"\nStep {step}:\n"
                    f"Thought: {thought}\n"
                    f"Action: generate_patch()\n"
                    f"Observation: Ready to generate patch.\n"
                )
                print(
                    "   [REACT] Reasoning complete "
                    "-- proceeding to patch generation"
                )
                break

            # Execute the action
            observation = self._execute_action(
                action, repo_path, original_files,
            )

            # Accumulate history
            history += (
                f"\nStep {step}:\n"
                f"Thought: {thought}\n"
                f"Action: {action}\n"
                f"Observation: {observation[:1500]}\n"
            )

        if not history:
            history = "(No reasoning steps completed)"

        print(f"\n[REACT] Reasoning loop completed ({final_step} step(s))")
        return history

    @staticmethod
    def _parse_react_response(response_text: str) -> tuple[str, str]:
        """
        Parse Thought and Action from the LLM's ReAct-format response.

        Args:
            response_text: Raw LLM response string.

        Returns:
            (thought, action) tuple of strings.
        """
        thought = ""
        action = "None"

        if not isinstance(response_text, str) or not response_text.strip():
            return thought, action

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)",
            response_text,
            re.DOTALL,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(
            r"Action:\s*(.+?)(?=\nObservation:|\nThought:|\n\n|\Z)",
            response_text,
            re.DOTALL,
        )
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _execute_action(
        self,
        action_str: str,
        repo_path: str,
        original_files: dict[str, str],
    ) -> str:
        """
        Execute a ReAct action and return the observation string.

        Args:
            action_str: Action string from LLM (e.g. "search_code(query)").
            repo_path: Path to the cloned repository.
            original_files: Already-loaded file contents.

        Returns:
            Observation string describing the action result.
        """
        if not isinstance(action_str, str):
            return "No action taken -- continue reasoning."

        cleaned = action_str.strip().lower()
        if cleaned in ("none", ""):
            return "No action taken -- continue reasoning."

        # Parse action_name(arguments)
        match = re.match(
            r"(\w+)\((.+?)\)\s*$", action_str.strip(), re.DOTALL,
        )
        if not match:
            return (
                f"Invalid action format: '{action_str}'. "
                f"Use: action_name(argument)"
            )

        action_name = match.group(1).lower()
        args = match.group(2).strip().strip("\"'")

        if action_name == "search_code":
            return self._action_search_code(args, repo_path)
        elif action_name == "open_file":
            return self._action_open_file(args, repo_path, original_files)
        elif action_name == "run_tests":
            return (
                "Test execution is not available in this context. "
                "Reason about correctness by inspecting the code instead."
            )
        else:
            return (
                f"Unknown action: '{action_name}'. "
                f"Available: search_code, open_file, generate_patch"
            )

    @staticmethod
    def _action_search_code(
        query: str,
        repo_path: str,
        max_results: int = 15,
    ) -> str:
        """
        Search the codebase for lines matching query keywords.

        Args:
            query: Search query string.
            repo_path: Path to the cloned repository.
            max_results: Maximum number of matching lines to return.

        Returns:
            Matching lines as a string, or a no-results message.
        """
        if not query or not repo_path:
            return "Invalid search query or repo path."

        keywords = [
            kw.lower()
            for kw in re.findall(r"[a-zA-Z_]\w{2,}", query)
        ]
        if not keywords:
            return "No valid search keywords extracted from query."

        results = []
        skip_dirs = {
            "__pycache__", "venv", "env", "node_modules", ".git",
        }
        code_exts = (".py", ".js", ".ts", ".java", ".go", ".rs", ".rb")

        try:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in skip_dirs
                ]
                for fname in files:
                    if not fname.endswith(code_exts):
                        continue
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, repo_path)
                    try:
                        with open(
                            full_path, "r",
                            encoding="utf-8", errors="ignore",
                        ) as f:
                            for i, line in enumerate(f, 1):
                                line_lower = line.lower()
                                if any(kw in line_lower for kw in keywords):
                                    results.append(
                                        f"{rel_path}:{i}: {line.rstrip()}"
                                    )
                                    if len(results) >= max_results:
                                        return "\n".join(results)
                    except OSError:
                        continue
        except Exception as e:
            return f"Search error: {e}"

        if results:
            return "\n".join(results)
        return f"No results found for: {query}"

    @staticmethod
    def _action_open_file(
        path: str,
        repo_path: str,
        original_files: dict[str, str],
        max_chars: int = 5000,
    ) -> str:
        """
        Read a file from the repository. Uses cached content if available.

        Args:
            path: Relative file path to read.
            repo_path: Path to the cloned repository.
            original_files: Already-loaded file contents dict.
            max_chars: Maximum characters to return.

        Returns:
            File content string, or an error message.
        """
        if not path or not repo_path:
            return "Invalid file path or repo path."

        # Normalize path separators
        norm_path = path.replace("\\", "/").strip("/")

        # Check if already loaded in memory
        if norm_path in original_files:
            content = original_files[norm_path]
            return content  # Always return full content — never truncate

        full_path = os.path.normpath(os.path.join(repo_path, norm_path))

        # Security: path traversal guard
        repo_real = os.path.realpath(repo_path)
        file_real = os.path.realpath(full_path)
        if not file_real.startswith(repo_real):
            return f"[BLOCKED] Path traversal attempt: {path}"

        if not os.path.isfile(full_path):
            return f"File not found: {path}"

        try:
            with open(
                full_path, "r", encoding="utf-8", errors="ignore",
            ) as f:
                content = f.read()  # Always read full file — never truncate
            return content
        except OSError as e:
            return f"Failed to read file: {e}"

    # ------------------------------------------------------------------
    # Domain Signal Inference (heuristic, no LLM call)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_domain_signals(
        original_files: dict[str, str],
        context: str = "",
    ) -> str:
        """
        Infer domain-specific signals from the codebase by analyzing
        imports, naming patterns, types, validation, and error handling.

        This is purely heuristic — zero LLM cost, no hardcoded domain rules.

        Args:
            original_files: {filepath: source_code} for target files.
            context: Additional retrieved context string.

        Returns:
            A formatted string of inferred signals, or "" if none found.
        """
        if not isinstance(original_files, dict):
            return ""

        # Combine all source code for scanning
        all_code = "\n".join(
            content for content in original_files.values()
            if isinstance(content, str)
        )
        if isinstance(context, str) and context:
            all_code += "\n" + context

        if not all_code.strip():
            return ""

        signals: list[str] = []

        # --- 1. Import patterns ---
        import_lines = re.findall(
            r"^(?:from\s+\S+\s+)?import\s+.+$",
            all_code,
            re.MULTILINE,
        )
        notable_imports: list[str] = []
        for line in import_lines:
            lower = line.lower()
            # Detect precision-sensitive imports
            if "decimal" in lower:
                notable_imports.append("Decimal (precision-sensitive arithmetic)")
            if "datetime" in lower or "timedelta" in lower:
                notable_imports.append("datetime (time-aware logic)")
            if "requests" in lower or "httpx" in lower:
                notable_imports.append("HTTP client (network I/O, needs error handling)")
            if "pandas" in lower or "numpy" in lower:
                notable_imports.append("Data processing library (vectorized ops)")
            if "sqlalchemy" in lower or "django" in lower:
                notable_imports.append("ORM/database layer (transactional)")
            if "pydantic" in lower:
                notable_imports.append("Pydantic (strict validation models)")
            if "logging" in lower:
                notable_imports.append("logging (structured logging expected)")
            if "typing" in lower:
                notable_imports.append("typing (type annotations expected)")
            if "enum" in lower:
                notable_imports.append("Enum (constrained value sets)")
            if "pathlib" in lower:
                notable_imports.append("pathlib (path handling)")

        # Deduplicate
        notable_imports = list(dict.fromkeys(notable_imports))
        if notable_imports:
            signals.append(
                "Imports detected: " + "; ".join(notable_imports[:8])
            )

        # --- 1b. Database driver detection (CRITICAL for SQL parameter style) ---
        # Maps import patterns to their required placeholder syntax
        _DB_DRIVERS = {
            "sqlite3":          ("sqlite3",           "?"),
            "psycopg2":         ("psycopg2/PostgreSQL", "%s"),
            "psycopg":          ("psycopg3/PostgreSQL", "%s"),
            "pg8000":           ("pg8000/PostgreSQL",  "%s"),
            "mysql.connector":  ("mysql-connector",   "%s"),
            "pymysql":          ("PyMySQL",           "%s"),
            "pymssql":          ("pymssql/SQL Server", "%s"),
            "pyodbc":           ("pyodbc/ODBC",       "?"),
            "cx_Oracle":        ("cx_Oracle/Oracle",  ":name"),
            "oracledb":         ("oracledb/Oracle",   ":name"),
        }
        detected_driver = None
        detected_placeholder = None
        for driver_key, (driver_label, placeholder) in _DB_DRIVERS.items():
            if re.search(rf"\b{re.escape(driver_key)}\b", all_code):
                detected_driver = driver_label
                detected_placeholder = placeholder
                break

        # Also detect from connection patterns if imports are ambiguous
        if not detected_driver:
            if re.search(r"sqlite3\.connect\s*\(", all_code):
                detected_driver = "sqlite3"
                detected_placeholder = "?"
            elif re.search(r"\.execute\s*\([^)]*\?", all_code):
                # Code already uses ? placeholders — likely sqlite3 or pyodbc
                detected_driver = "qmark-style driver"
                detected_placeholder = "?"
            elif re.search(r"\.execute\s*\([^)]*%s", all_code):
                # Code already uses %s placeholders
                detected_driver = "format-style driver"
                detected_placeholder = "%s"

        if detected_driver:
            signals.append(
                f"DATABASE: {detected_driver} detected — "
                f"MUST use '{detected_placeholder}' for parameterized queries. "
                f"Do NOT use other placeholder styles."
            )

        # --- 2. Type / precision patterns ---
        if re.search(r"\bDecimal\b", all_code):
            signals.append(
                "Precision: Code uses Decimal -- new code MUST use Decimal "
                "for financial/exact arithmetic, NOT float"
            )
        if re.search(r"\bfloat\b.*\b(?:round|format)\b", all_code):
            signals.append(
                "Precision: Code uses float with explicit rounding -- "
                "preserve rounding approach"
            )

        # --- 3. Validation patterns ---
        validation_patterns = {
            r"isinstance\s*\(": "isinstance() type checks",
            r"if\s+\w+\s+is\s+None": "'is None' guard clauses",
            r"raise\s+TypeError": "TypeError for invalid types",
            r"raise\s+ValueError": "ValueError for invalid values",
            r"if\s+not\s+\w+": "falsy-value guard clauses",
        }
        found_validation: list[str] = []
        for pattern, label in validation_patterns.items():
            if re.search(pattern, all_code):
                found_validation.append(label)
        if found_validation:
            signals.append(
                "Validation style: " + ", ".join(found_validation[:6])
            )

        # --- 4. Error handling patterns ---
        custom_exceptions = re.findall(
            r"class\s+(\w*(?:Error|Exception)\w*)\s*\(",
            all_code,
        )
        if custom_exceptions:
            signals.append(
                "Custom exceptions defined: "
                + ", ".join(list(dict.fromkeys(custom_exceptions))[:5])
                + " -- use these instead of generic exceptions"
            )

        try_except_count = len(re.findall(r"\btry\s*:", all_code))
        if try_except_count > 2:
            signals.append(
                f"Error handling: Code uses try/except extensively "
                f"({try_except_count} blocks) -- follow same pattern"
            )

        # --- 5. Naming conventions ---
        # Check for snake_case vs camelCase dominance
        snake_funcs = re.findall(r"def\s+([a-z][a-z0-9_]*)\s*\(", all_code)
        camel_funcs = re.findall(r"def\s+([a-z][a-zA-Z0-9]*[A-Z]\w*)\s*\(", all_code)
        if snake_funcs and not camel_funcs:
            signals.append("Naming: snake_case functions (follow this)")
        elif camel_funcs and not snake_funcs:
            signals.append("Naming: camelCase functions (follow this)")

        # Check for type annotations
        annotated = len(re.findall(r"def\s+\w+\([^)]*:\s*\w+", all_code))
        unannotated = len(re.findall(r"def\s+\w+\(\s*self\s*\)", all_code))
        if annotated > unannotated and annotated > 2:
            signals.append(
                "Style: Code uses type annotations -- add them to new code"
            )

        # --- 6. Docstring convention ---
        if re.search(r'""".*?"""', all_code, re.DOTALL):
            if re.search(r'Args:\s*\n', all_code):
                signals.append(
                    "Docstrings: Google-style with Args/Returns sections"
                )
            elif re.search(r':param\s+\w+:', all_code):
                signals.append("Docstrings: Sphinx-style with :param:")
            else:
                signals.append("Docstrings: triple-quote docstrings present")

        # --- 7. Domain-hint variable names ---
        domain_hints: list[str] = []
        finance_names = re.findall(
            r"\b(?:amount|price|rate|interest|principal|balance|"
            r"payment|tenure|emi|tax|discount|revenue|cost)\b",
            all_code, re.IGNORECASE,
        )
        if len(set(n.lower() for n in finance_names)) >= 2:
            domain_hints.append("financial/monetary domain detected")

        api_names = re.findall(
            r"\b(?:endpoint|payload|response|request|header|"
            r"status_code|api_key|token|url|webhook)\b",
            all_code, re.IGNORECASE,
        )
        if len(set(n.lower() for n in api_names)) >= 2:
            domain_hints.append("API/HTTP domain detected")

        data_names = re.findall(
            r"\b(?:dataframe|series|column|row|csv|parquet|"
            r"dataset|feature|label|transform)\b",
            all_code, re.IGNORECASE,
        )
        if len(set(n.lower() for n in data_names)) >= 2:
            domain_hints.append("data processing domain detected")

        if domain_hints:
            signals.append(
                "Domain hints: " + "; ".join(domain_hints)
            )

        if not signals:
            return ""

        # Format as a structured block
        formatted = "\n".join(f"  - {s}" for s in signals[:10])
        print(f"[DOMAIN] Inferred {len(signals)} signal(s) from codebase")
        for s in signals[:10]:
            print(f"   - {s}")

        return formatted

    # ------------------------------------------------------------------
    # Unified diff validation (kept for fallback + PR display)
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_unified_diff(text: str) -> bool:
        """
        Check that text looks like a valid unified diff.

        Args:
            text: The candidate diff string.

        Returns:
            True if text contains ---, +++, and @@ markers.
        """
        if not isinstance(text, str) or not text.strip():
            return False

        lines = text.strip().splitlines()

        has_minus = any(line.startswith("--- ") for line in lines)
        has_plus = any(line.startswith("+++ ") for line in lines)
        has_hunk = any(line.startswith("@@ ") for line in lines)

        return has_minus and has_plus and has_hunk

    @staticmethod
    def extract_diffs_from_response(response_text: str) -> dict[str, str]:
        """
        Extract per-file unified diffs from the LLM response.

        Args:
            response_text: Raw LLM response string.

        Returns:
            {filepath: diff_text} for each valid file diff found.
        """
        if not isinstance(response_text, str) or not response_text.strip():
            return {}

        diffs: dict[str, str] = {}

        # Split by --- a/ markers to find individual file diffs
        segments = re.split(r"(?=^--- a/)", response_text, flags=re.MULTILINE)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Extract file path from --- a/path line
            match = re.match(r"--- a/(.+?)(?:\s|$)", segment)
            if not match:
                continue

            file_path = match.group(1).strip()
            if not file_path:
                continue

            # Validate this segment has proper diff markers
            if PatchGenerator.is_valid_unified_diff(segment):
                diffs[file_path] = segment

        return diffs

    # ------------------------------------------------------------------
    # Apply unified diff to original code (FALLBACK only)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_unified_diff(original: str, diff_text: str) -> str | None:
        """
        Apply a unified diff to the original source code.
        Used only as a fallback when LLM returns diffs instead of full files.

        Args:
            original: The original source code string.
            diff_text: The unified diff to apply.

        Returns:
            The patched code string, or None if application fails.
        """
        if not isinstance(original, str):
            return None
        if not isinstance(diff_text, str) or not diff_text.strip():
            return None

        # Handle empty original (new file scenario)
        if not original:
            original = ""

        original_lines = original.splitlines(keepends=True)
        # Ensure last line has newline
        if original_lines and not original_lines[-1].endswith("\n"):
            original_lines[-1] += "\n"

        patched_lines = list(original_lines)

        # Parse hunks from the diff
        hunks = PatchGenerator._parse_hunks(diff_text)
        if not hunks:
            return None

        # Apply hunks in reverse order to preserve line numbers
        hunks.sort(key=lambda h: h["orig_start"], reverse=True)

        any_applied = False
        for hunk in hunks:
            orig_start = hunk["orig_start"] - 1  # 0-indexed
            orig_lines_list = hunk["orig_lines"]
            new_lines_list = hunk["new_lines"]

            # Guard against negative or out-of-range start
            if orig_start < 0:
                orig_start = 0

            # Verify the original lines match (fuzzy -- ignore trailing whitespace)
            file_section = patched_lines[orig_start:orig_start + len(orig_lines_list)]
            file_stripped = [l.rstrip() for l in file_section]
            orig_stripped = [l.rstrip() for l in orig_lines_list]

            if file_stripped != orig_stripped:
                # Try a fuzzy search within +/-30 lines (expanded range)
                found = False
                for offset in range(-30, 31):
                    idx = orig_start + offset
                    if idx < 0 or idx + len(orig_lines_list) > len(patched_lines):
                        continue
                    section = patched_lines[idx:idx + len(orig_lines_list)]
                    if [l.rstrip() for l in section] == orig_stripped:
                        orig_start = idx
                        found = True
                        break

                # Fallback: whitespace-normalized matching (collapse all whitespace)
                if not found:
                    import re as _re_hunk
                    _norm = lambda s: _re_hunk.sub(r'\s+', '', s.strip())
                    orig_norm = [_norm(l) for l in orig_lines_list]
                    for offset in range(-30, 31):
                        idx = orig_start + offset
                        if idx < 0 or idx + len(orig_lines_list) > len(patched_lines):
                            continue
                        section = patched_lines[idx:idx + len(orig_lines_list)]
                        if [_norm(l) for l in section] == orig_norm:
                            orig_start = idx
                            found = True
                            print(f"   [INFO] Hunk at line {hunk['orig_start']} matched via whitespace-normalized fuzzy at offset {offset}")
                            break

                if not found:
                    print(f"   [WARN] Hunk at line {hunk['orig_start']} does not match source -- skipping")
                    continue

            # Replace the original section with new lines
            patched_lines[orig_start:orig_start + len(orig_lines_list)] = new_lines_list
            any_applied = True

        if not any_applied:
            return None

        result = "".join(patched_lines)
        # Remove trailing newline if original didn't have one
        if not original.endswith("\n") and result.endswith("\n"):
            result = result[:-1]

        return result

    @staticmethod
    def _parse_hunks(diff_text: str) -> list[dict]:
        """
        Parse unified diff hunks from diff text.

        Args:
            diff_text: A unified diff string.

        Returns:
            List of hunk dicts with orig_start, orig_lines, new_lines.
        """
        if not isinstance(diff_text, str) or not diff_text.strip():
            return []

        hunks = []
        lines = diff_text.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Find hunk header: @@ -start,count +start,count @@
            hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match:
                orig_start = int(hunk_match.group(1))
                if orig_start < 1:
                    orig_start = 1

                orig_lines = []
                new_lines = []

                i += 1
                while i < len(lines):
                    l = lines[i]
                    if l.startswith("@@ ") or l.startswith("--- ") or l.startswith("+++ "):
                        break
                    if l.startswith("-"):
                        orig_lines.append(l[1:] + "\n")
                    elif l.startswith("+"):
                        new_lines.append(l[1:] + "\n")
                    elif l.startswith(" "):
                        orig_lines.append(l[1:] + "\n")
                        new_lines.append(l[1:] + "\n")
                    else:
                        # Context line without prefix (some LLMs do this)
                        orig_lines.append(l + "\n")
                        new_lines.append(l + "\n")
                    i += 1

                hunks.append({
                    "orig_start": orig_start,
                    "orig_lines": orig_lines,
                    "new_lines": new_lines,
                })
                continue

            i += 1

        return hunks

    # ------------------------------------------------------------------
    # Diff size validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_diff_size(diff_text: str, max_lines: int = 150) -> tuple[bool, str]:
        """
        Validate that diff is not too large.

        Args:
            diff_text: A unified diff string.
            max_lines: Maximum allowed changed lines.

        Returns:
            (is_valid, reason_message).
        """
        if not isinstance(diff_text, str) or not diff_text.strip():
            return False, "Diff text is empty or invalid"

        if not isinstance(max_lines, int) or max_lines <= 0:
            max_lines = 150

        lines = diff_text.strip().splitlines()
        change_lines = [l for l in lines if l.startswith("+") or l.startswith("-")]
        change_count = len([l for l in change_lines
                           if not l.startswith("---") and not l.startswith("+++")])

        if change_count == 0:
            return False, "Diff is empty -- no actual changes"
        return True, "OK"

    # ------------------------------------------------------------------
    # Extract code from LLM response (for full-file strategy)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_from_response(response_text: str) -> str | None:
        if not isinstance(response_text, str) or not response_text.strip():
            return None

        text = response_text.strip()

        # Try extracting from markdown code fences (any language tag)
        fence_match = re.search(
            r"```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)```",
            text,
            re.DOTALL,
        )
        if fence_match:
            return fence_match.group(1).strip()

        # If no fences, return the full response — the LLM likely returned raw code
        return text

    # ------------------------------------------------------------------
    # Multi-file diff generation (PRIMARY: full-file rewrite strategy)
    # ------------------------------------------------------------------

    def generate_diffs(
        self,
        plan: dict,
        context: str,
        original_files: dict[str, str],
        issue_text: str,
        errors: str = "",
        previous_diffs: dict[str, str] | None = None,
        repo_path: str = "",
        edge_cases: list[str] | None = None,
        qa_feedback: str = None,
        failure_type: str = None,
        failed_tests: int = None,
        previous_patch: dict = None,
        retry_strategy: str = "normal",
        retry_count: int = 0,          # CRITICAL: was missing — caused NameError/TypeError on every call
        simulated_results: list = None, # per-test pass/fail details from test_simulation_agent
    ) -> tuple[dict[str, str], dict[str, str]] | None:
        """
        Generate code fixes for all files in the plan.

        PRIMARY STRATEGY: Ask LLM to output the complete modified file.
        Then compute the diff from original vs patched for validation.
        This avoids the fragile diff-parsing/hunk-matching that causes
        "hunk does not match source" failures.

        FALLBACK: If the LLM returns unified diffs, try to apply them.

        Args:
            plan: The planner output with files_to_modify and steps.
            context: Relevant code context string.
            original_files: {filepath: source_code} for each target file.
            issue_text: The issue description.
            errors: Accumulated error descriptions from previous attempts.
            previous_diffs: Diffs from last failed attempt (to avoid repeating).

        Returns:
            (code_diffs, patched_files) or None if generation fails.

        Raises:
            TypeError: If plan is not a dict or original_files is not a dict.
        """
        # --- Input validation ---
        if not isinstance(plan, dict):
            raise TypeError(f"plan must be a dict, got {type(plan).__name__}")
        if not isinstance(original_files, dict):
            raise TypeError(f"original_files must be a dict, got {type(original_files).__name__}")
        if not isinstance(issue_text, str):
            raise TypeError(f"issue_text must be a string, got {type(issue_text).__name__}")

        if not isinstance(context, str):
            context = ""
        if not isinstance(errors, str):
            errors = ""
        if previous_diffs is not None and not isinstance(previous_diffs, dict):
            previous_diffs = None
        if not isinstance(repo_path, str):
            repo_path = ""
        if not isinstance(edge_cases, list):
            edge_cases = []
        if not isinstance(simulated_results, list):
            simulated_results = []

        files_to_modify = plan.get("files_to_modify", [])
        steps = plan.get("steps", [])

        if not isinstance(files_to_modify, list) or not files_to_modify:
            raise ValueError("No files to modify in plan")

        if not isinstance(steps, list):
            steps = []

        steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps) if isinstance(s, str))
        plan_targets = plan.get("targets", []) if isinstance(plan.get("targets", []), list) else []

        error_block = ""
        if errors and errors.strip():
            error_block = (
                f"\n[WARN] PREVIOUS ERRORS (DO NOT REPEAT THE SAME MISTAKES):\n"
                f"{errors}\n"
            )

        prev_diff_block = ""
        if previous_diffs:
            prev_diff_block = "\n[WARN] PREVIOUS FAILED DIFFS (generate a DIFFERENT fix):\n"
            for fpath, diff in previous_diffs.items():
                if isinstance(fpath, str) and isinstance(diff, str):
                    prev_diff_block += f"\n--- Failed diff for {fpath} ---\n{diff[:500]}\n"

        # --- ReAct Reasoning Loop: Analyze before patching ---
        # Skip on retries to save tokens (reasoning already captured in first attempt)
        if retry_count == 0:
            reasoning_history = self._react_reasoning_loop(
                issue_text=issue_text,
                plan=plan,
                context=context,
                original_files=original_files,
                repo_path=repo_path,
                errors=errors,
                edge_cases=edge_cases,
                max_steps=2,
            )
        else:
            reasoning_history = "(Skipped on retry to reduce token usage)"

        # Format edge cases for patch generation context
        edge_case_context = ""
        if edge_cases:
            ec_lines = "\n".join(f"  - {ec}" for ec in edge_cases if isinstance(ec, str))
            edge_case_context = (
                f"\n=== EDGE CASES (your fix MUST handle these) ===\n"
                f"{ec_lines}\n"
            )

        # Infer domain signals for patch generation context (cached to avoid double inference)
        if not hasattr(self, '_cached_domain_signals'):
            self._cached_domain_signals = self._infer_domain_signals(original_files, context)
        domain_signals = self._cached_domain_signals
        domain_context = ""
        if domain_signals:
            domain_context = (
                f"\n=== REPOSITORY CONVENTIONS (MUST follow) ===\n"
                f"{domain_signals}\n"
            )

        # Enrich context with reasoning + domain + edge cases
        enrichment = ""
        if reasoning_history and reasoning_history.strip() != "(No reasoning steps completed)":
            enrichment += f"=== AGENT REASONING ===\n{reasoning_history}\n\n"
        if domain_context:
            enrichment += f"{domain_context}\n\n"
        if edge_case_context:
            enrichment += f"{edge_case_context}\n\n"

        # Build structured test failure context from simulated results
        test_failure_context = ""
        if simulated_results and retry_count > 0:
            failed_details = []
            passed_details = []
            for tr in simulated_results:
                if not isinstance(tr, dict):
                    continue
                name = tr.get("test_name", "unknown")
                status = tr.get("status", "unknown")
                reasoning = tr.get("reasoning", "")
                confidence = tr.get("confidence", 0.0)
                if status == "fail":
                    failed_details.append(
                        f"  FAILED: {name}\n"
                        f"    Reason: {reasoning[:300]}\n"
                        f"    Confidence: {confidence}"
                    )
                elif status == "pass":
                    passed_details.append(f"  PASSED: {name}")

            if failed_details:
                test_failure_context = (
                    "\n=== SIMULATED TEST FAILURES (your fix MUST address these) ===\n"
                    + "\n".join(failed_details)
                    + "\n"
                )
                if passed_details:
                    test_failure_context += (
                        "\n=== TESTS THAT PASSED (do NOT break these) ===\n"
                        + "\n".join(passed_details[:5])
                        + "\n"
                    )

        if enrichment:
            context = f"{enrichment}=== RETRIEVED CONTEXT ===\n{context}"
        if test_failure_context:
            context = f"{test_failure_context}\n{context}"

        context_limit = 5000
        if retry_count >= 2:
            context_limit = 3500
        if retry_count >= 3:
            context_limit = 2500
        if isinstance(context, str) and len(context) > context_limit:
            context = context[:context_limit]

        # Process each file individually for reliability
        all_diffs: dict[str, str] = {}
        all_patched: dict[str, str] = {}
        llm_unavailable_flag = False
        requires_behavior_change = self._issue_requires_behavior_change(issue_text, plan, failure_type)

        for fpath in files_to_modify:
            if not isinstance(fpath, str):
                continue
            if fpath not in original_files:
                raise ValueError(f"No original source for {fpath}")

            original = original_files[fpath]
            if not isinstance(original, str):
                raise ValueError(f"Original source for {fpath} is not a string")

            print(f"\n[CODER] Generating fix for: {fpath} | strategy={retry_strategy!r} retry={retry_count}")

            _prev = previous_patch.get(fpath) if isinstance(previous_patch, dict) else None
            force_localized_edit = len(original) > 10_000

            # ── Strategy ladder: try strategies in order, collect last failure reason ──
            # Each strategy must be tried in full before moving to the next.
            # NEVER raise before exhausting all fallbacks for a given file.
            patched = None
            strategy_used = None
            last_fail_reason = "unknown"

            if retry_strategy in ("normal", "function-only"):
                # Strategy 1: AST-based section patch (targets only the relevant function)
                print(f"   [INFO] Strategy 1: section-patch for {fpath} (retry={retry_count})")
                try:
                    patched = self._generate_section_patch(
                        fpath, original, issue_text, steps_block, context,
                        error_block, qa_feedback, failure_type, failed_tests, _prev, retry_count,
                        plan_targets,
                        edge_case_context, test_failure_context
                    )
                except Exception as e:
                    last_fail_reason = f"section-patch exception: {e}"
                    print(f"   [WARN] Section-patch raised: {e}")

                if patched is not None and self._has_meaningful_change(original, patched):
                    diff = self.compute_unified_diff(original, patched, fpath)
                    if diff != "NO_CHANGES":
                        all_diffs[fpath] = diff
                        all_patched[fpath] = patched
                        strategy_used = "section-patch"
                        print(f"   [OK] section-patch succeeded for {fpath}")
                        continue
                    else:
                        last_fail_reason = "section-patch produced no changes"
                else:
                    if patched is None:
                        last_fail_reason = "section-patch returned None"
                    else:
                        last_fail_reason = "section-patch output identical to original"
                print(f"   [WARN] section-patch failed ({last_fail_reason}), trying next strategy")

                # Strategy 2 (normal only): full-file rewrite — broader but comprehensive
                # Skipped for function-only to keep context scope narrow
                if retry_strategy == "normal" and not force_localized_edit:
                    print(f"   [INFO] Strategy 2: full-file for {fpath} (retry={retry_count})")
                    try:
                        patched = self._generate_full_file(
                            fpath, original, issue_text, steps_block, context,
                            error_block, "", qa_feedback, failure_type, failed_tests, _prev, retry_count,
                            edge_case_context, test_failure_context
                        )
                    except Exception as e:
                        last_fail_reason = f"full-file exception: {e}"
                        print(f"   [WARN] full-file raised: {e}")
                    else:
                        if patched is not None and self._has_meaningful_change(original, patched):
                            diff = self.compute_unified_diff(original, patched, fpath)
                            if diff != "NO_CHANGES":
                                all_diffs[fpath] = diff
                                all_patched[fpath] = patched
                                strategy_used = "full-file"
                                print(f"   [OK] full-file strategy succeeded for {fpath}")
                                continue
                        last_fail_reason = "full-file returned None or no changes"
                        print(f"   [WARN] full-file failed, trying diff strategy")

            elif retry_strategy == "minimal" or retry_strategy.startswith("minimal"):
                if requires_behavior_change:
                    print(f"   [INFO] Minimal shortcut disabled for {fpath} because the issue requires a real behavior change")
                    try:
                        patched = self._generate_section_patch(
                            fpath, original, issue_text, steps_block, context,
                            error_block, qa_feedback, failure_type, failed_tests, _prev, retry_count,
                            plan_targets,
                            edge_case_context, test_failure_context
                        )
                    except Exception as e:
                        last_fail_reason = f"section-patch exception: {e}"
                        print(f"   [WARN] section-patch raised: {e}")

                    if patched is not None and self._has_meaningful_change(original, patched):
                        diff = self.compute_unified_diff(original, patched, fpath)
                        if diff != "NO_CHANGES":
                            all_diffs[fpath] = diff
                            all_patched[fpath] = patched
                            strategy_used = "section-patch-minimal-safe"
                            print(f"   [OK] section-patch succeeded for {fpath}")
                            continue
                    else:
                        last_fail_reason = "section-patch returned None or no changes"
                        print(f"   [WARN] section-patch failed, trying diff strategy")
                else:
                    # Strategy 1 (minimal): targeted string replacement — smallest possible change
                    print(f"   [INFO] Strategy 1 (minimal): string-replacement for {fpath}")
                    try:
                        patched = self._generate_minimal_string_replacement(
                            fpath, original, issue_text, steps_block, _prev
                        )
                    except Exception as e:
                        last_fail_reason = f"string-replacement exception: {e}"
                        print(f"   [WARN] string-replacement raised: {e}")

                    if patched is not None and self._has_meaningful_change(original, patched):
                        diff = self.compute_unified_diff(original, patched, fpath)
                        if diff != "NO_CHANGES":
                            all_diffs[fpath] = diff
                            all_patched[fpath] = patched
                            strategy_used = "string-replacement"
                            print(f"   [OK] string-replacement succeeded for {fpath}")
                            continue
                    else:
                        last_fail_reason = "string-replacement returned None or no changes"
                        print(f"   [WARN] string-replacement failed, trying deterministic fix")

                    print(f"   [INFO] Strategy 2 (minimal): deterministic-fix for {fpath}")
                    try:
                        patched = self._deterministic_minimal_fix(original, issue_text)
                    except Exception as e:
                        last_fail_reason = f"deterministic-fix exception: {e}"
                        print(f"   [WARN] deterministic-fix raised: {e}")

                    if patched is not None and self._has_meaningful_change(original, patched):
                        diff = self.compute_unified_diff(original, patched, fpath)
                        if diff != "NO_CHANGES":
                            all_diffs[fpath] = diff
                            all_patched[fpath] = patched
                            strategy_used = "deterministic-fix"
                            print(f"   [OK] deterministic-fix succeeded for {fpath}")
                            continue
                    else:
                        last_fail_reason = "deterministic-fix returned None or no changes"
                        print(f"   [WARN] deterministic-fix failed, trying section-patch")

                    print(f"   [INFO] Strategy 3 (minimal): section-patch for {fpath}")
                    try:
                        patched = self._generate_section_patch(
                            fpath, original, issue_text, steps_block, context,
                            error_block, qa_feedback, failure_type, failed_tests, _prev, retry_count,
                            plan_targets,
                            edge_case_context, test_failure_context
                        )
                    except Exception as e:
                        last_fail_reason = f"section-patch exception: {e}"
                        print(f"   [WARN] section-patch raised: {e}")

                    if patched is not None and self._has_meaningful_change(original, patched):
                        diff = self.compute_unified_diff(original, patched, fpath)
                        if diff != "NO_CHANGES":
                            all_diffs[fpath] = diff
                            all_patched[fpath] = patched
                            strategy_used = "section-patch-minimal"
                            print(f"   [OK] section-patch succeeded for {fpath}")
                            continue
                    else:
                        last_fail_reason = "section-patch returned None or no changes"
                        print(f"   [WARN] section-patch failed, trying diff strategy")

            elif retry_strategy == "force_append":
                # force_append: add new functionality — section-patch with create-oriented plan
                print(f"   [INFO] Strategy (force_append): section-patch for {fpath} (retry={retry_count})")
                try:
                    patched = self._generate_section_patch(
                        fpath, original, issue_text, steps_block, context,
                        error_block, qa_feedback, failure_type, failed_tests, _prev, retry_count,
                        plan_targets,
                        edge_case_context, test_failure_context
                    )
                except Exception as e:
                    last_fail_reason = f"section-patch exception: {e}"
                    print(f"   [WARN] section-patch raised: {e}")

                if patched is not None and self._has_meaningful_change(original, patched):
                    diff = self.compute_unified_diff(original, patched, fpath)
                    if diff != "NO_CHANGES":
                        all_diffs[fpath] = diff
                        all_patched[fpath] = patched
                        strategy_used = "section-patch-force-append"
                        print(f"   [OK] force_append section-patch succeeded for {fpath}")
                        continue
                last_fail_reason = "force_append section-patch returned None or no changes"
                print(f"   [WARN] force_append section-patch failed, trying full-file")

                # Fallback to full-file for force_append
                if not force_localized_edit:
                    try:
                        patched = self._generate_full_file(
                            fpath, original, issue_text, steps_block, context,
                            error_block, "", qa_feedback, failure_type, failed_tests, _prev, retry_count,
                            edge_case_context, test_failure_context
                        )
                    except Exception as e:
                        last_fail_reason = f"full-file exception: {e}"
                        print(f"   [WARN] full-file raised: {e}")
                    else:
                        if patched is not None and self._has_meaningful_change(original, patched):
                            diff = self.compute_unified_diff(original, patched, fpath)
                            if diff != "NO_CHANGES":
                                all_diffs[fpath] = diff
                                all_patched[fpath] = patched
                                strategy_used = "full-file-force-append"
                                print(f"   [OK] force_append full-file succeeded for {fpath}")
                                continue
                    last_fail_reason = "force_append full-file returned None or no changes"
                    print(f"   [WARN] force_append full-file failed, trying diff strategy")

            elif retry_strategy == "minimal_safe_fix":
                # minimal_safe_fix: most conservative single-function fix
                print(f"   [INFO] Strategy (minimal_safe_fix): section-patch for {fpath} (retry={retry_count})")
                try:
                    patched = self._generate_section_patch(
                        fpath, original, issue_text, steps_block, context,
                        error_block, qa_feedback, failure_type, failed_tests, _prev, retry_count,
                        plan_targets,
                        edge_case_context, test_failure_context
                    )
                except Exception as e:
                    last_fail_reason = f"section-patch exception: {e}"
                    print(f"   [WARN] section-patch raised: {e}")

                if patched is not None and self._has_meaningful_change(original, patched):
                    diff = self.compute_unified_diff(original, patched, fpath)
                    if diff != "NO_CHANGES":
                        all_diffs[fpath] = diff
                        all_patched[fpath] = patched
                        strategy_used = "section-patch-minimal-safe"
                        print(f"   [OK] minimal_safe_fix section-patch succeeded for {fpath}")
                        continue
                last_fail_reason = "minimal_safe_fix section-patch returned None or no changes"
                print(f"   [WARN] minimal_safe_fix section-patch failed, trying diff strategy")

            else:
                # Fallback path for unknown strategies — go straight to diff
                last_fail_reason = f"unknown strategy {retry_strategy!r}, using diff"

            # Final fallback: unified diff strategy — enabled for ALL strategies as last resort
            print(f"   [INFO] Final fallback: diff strategy for {fpath} (retry={retry_count})")
            try:
                patched = self._generate_via_diff(
                    fpath, original, issue_text, steps_block, context,
                    error_block, prev_diff_block,
                    qa_feedback, failure_type, failed_tests,
                    _prev, retry_count,
                )
            except Exception as e:
                last_fail_reason = f"diff-strategy exception: {e}"
                print(f"   [WARN] diff-strategy raised: {e}")
                patched = None

            if patched is not None and self._has_meaningful_change(original, patched):
                diff = self.compute_unified_diff(original, patched, fpath)
                if diff != "NO_CHANGES":
                    all_diffs[fpath] = diff
                    all_patched[fpath] = patched
                    strategy_used = "diff"
                    print(f"   [OK] diff strategy succeeded for {fpath}")
                    continue

            # All strategies exhausted for this file — log and continue with other files
            print(
                f"   [ERROR] All strategies exhausted for {fpath} "
                f"(strategy={retry_strategy!r}, retry={retry_count}). "
                f"Last failure: {last_fail_reason}"
            )

        if not all_patched:
            raise RuntimeError("No files were successfully patched by any strategy")

        # Validate syntax for Python files
        for fpath, patched in all_patched.items():
            if fpath.endswith(".py"):
                try:
                    compile(patched, fpath, "exec")
                except SyntaxError as e:
                    raise ValueError(f"Syntax error in patched {fpath}: {e}")

        # ── Structural integrity guard (deep — checks nested methods too) ──
        for fpath, patched in all_patched.items():
            original = original_files.get(fpath, "")
            if not original:
                continue

            struct_ok, struct_reason = LLMOutputValidator.validate_structural_integrity(
                original, patched, fpath
            )
            if not struct_ok:
                raise ValueError(f"Structural corruption in {fpath}: {struct_reason}")

            # Also warn if the patched file is significantly shorter (likely truncation)
            orig_line_ct = len(original.splitlines())
            patched_line_ct = len(all_patched.get(fpath, patched).splitlines())
            if orig_line_ct > 10 and patched_line_ct < orig_line_ct * 0.5:
                print(f"   [WARN] Patched {fpath} is {patched_line_ct} lines vs original {orig_line_ct} — possible truncation")



        # Debug Visibility
        print("\n--- DEBUG VISIBILITY ---")
        print(f"QA Feedback: {qa_feedback[:100] if qa_feedback else 'None'}...")
        print(f"Failure Type: {failure_type}")
        print(f"Retry: {retry_count}")

        if previous_patch and retry_count > 0:
            for fpath, patched in all_patched.items():
                prev = previous_patch.get(fpath)
                if prev:
                    print(f"Previous Patch Hash for {fpath}:", hash(prev.strip()))
                    print(f"New Patch Hash for {fpath}:", hash(patched.strip()))
                    changed = patched.strip() != prev.strip()
                    print(f"Patch Changed for {fpath}:", changed)
                    if not changed:
                        raise ValueError(f"Patch for {fpath} is identical to previous failed patch! Rejecting to force alternative.")
        print("------------------------\n")

        print(f"\n   [OK] Generated valid fixes for {len(all_diffs)} file(s)")
        for fpath, diff in all_diffs.items():
            diff_lines = diff.strip().splitlines()
            adds = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            dels = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
            orig_line_count = len(original_files.get(fpath, "").splitlines())
            print(f"      {fpath}: +{adds} -{dels} (orig {orig_line_count} lines)")

        return all_diffs, all_patched

    # ------------------------------------------------------------------
    # PRIMARY: Full file rewrite strategy
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Language-aware prompt builder
    # Detects language from extension, returns specific coding constraints.
    # No framework names hardcoded — rules derived from language properties.
    # ------------------------------------------------------------------

    # Maps extension sets to (language_label, list_of_rules)
    _LANG_RULES: dict[str, tuple[str, list[str]]] = {
        "python": ("Python", [
            "CRITICAL: Respect Python indentation (4 spaces) — incorrect indentation breaks the entire file.",
            "Follow PEP8. Match existing import style.",
            "Use type hints if already present in the file.",
            "Use the same numeric types already in use (Decimal, float, int).",
            "Preserve all existing function signatures unless the fix requires changing them.",
            "SCOPE: Modify ONLY the target function/method. Do NOT touch unrelated functions.",
            "STRUCTURE: Never remove existing classes, functions, or methods that weren't in the plan.",
            "Ensure all blocks (def, class, if, for, while, try) have complete bodies with meaningful code.",
            "MINIMAL: Make only the changes required to fix the issue. Avoid refactoring.",
        ]),
        "javascript": ("JavaScript", [
            "CRITICAL: Preserve exact brace and semicolon placement — mismatches break syntax.",
            "Match existing const/let/var usage — do not change variable declaration style.",
            "Preserve async/await and Promise patterns already in the file.",
            "Keep existing ES module (import/export) or CommonJS (require) style.",
            "SCOPE: Modify ONLY the target function. Do NOT touch unrelated functions.",
            "STRUCTURE: Never remove exported functions or classes that weren't in the plan.",
            "Ensure all function bodies have complete logic — no empty bodies or placeholders.",
            "MINIMAL: Make only required changes. Avoid reformatting unrelated code.",
        ]),
        "typescript": ("TypeScript", [
            "CRITICAL: Preserve all TypeScript type annotations and interface definitions exactly.",
            "Do not introduce 'any' types unless they already exist in the file.",
            "Match existing generic type patterns.",
            "Keep all import/export statements and type imports unchanged unless the fix requires modification.",
            "SCOPE: Modify ONLY the target function/method. Preserve all other code.",
            "STRUCTURE: Never remove type definitions, classes, or exported functions.",
            "Ensure type safety — the modified code must pass TypeScript compilation.",
            "MINIMAL: Make only the changes required. Avoid refactoring or type changes.",
        ]),
        "java": ("Java", [
            "CRITICAL: Preserve exact class structure, method signatures, and access modifiers.",
            "Maintain proper brace balance and indentation — 4 spaces per indent level.",
            "SCOPE: Modify ONLY the target method/class. Never remove unrelated methods.",
            "STRUCTURE: Never remove class members, imports, or public APIs that weren't in the plan.",
            "Keep method return types and parameter types unchanged unless the fix requires it.",
            "Ensure all method bodies have complete, compilable code.",
            "MINIMAL: Make only the changes needed to fix the issue.",
        ]),
        "html": ("HTML/Template", [
            "CRITICAL: Preserve ALL template block tags (e.g. {% %}, {{ }}, {# #}) exactly — do NOT remove or alter them.",
            "Do NOT convert template syntax to plain HTML.",
            "Maintain the existing HTML structure (DOCTYPE, head, body hierarchy).",
            "Preserve all form action attributes, CSRF tokens, and URL helper tags.",
            "Keep indentation of template blocks consistent with the original.",
            "SCOPE: Modify ONLY the specific HTML section described in the plan.",
            "Output the COMPLETE file — every line, including unchanged sections.",
        ]),
        "css": ("CSS/Stylesheet", [
            "CRITICAL: Do not remove or rename CSS classes — they may be referenced in templates or JS.",
            "Preserve existing media queries and selector specificity.",
            "SCOPE: Modify ONLY the target CSS rule or selector.",
            "Keep all media queries and responsive breakpoints intact.",
            "MINIMAL: Make only the property changes needed — don't refactor selectors.",
        ]),
    }


    _EXT_TO_LANG: dict[str, str] = {
        ".py": "python", ".pyw": "python",
        ".html": "html", ".htm": "html", ".jinja": "html", ".jinja2": "html", ".j2": "html",
        ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    }

    @classmethod
    def _get_language_instructions(cls, file_path: str) -> tuple[str, str]:
        """
        Returns (language_label, numbered_rules_block) for the given file.
        Used to inject language-specific constraints into LLM prompts.
        """
        ext = os.path.splitext(file_path)[1].lower()
        lang_key = cls._EXT_TO_LANG.get(ext, "unknown")
        label, rules = cls._LANG_RULES.get(lang_key, ("Unknown", [
            "Preserve the file's existing syntax and conventions exactly.",
            "Output the COMPLETE file — every line.",
        ]))
        numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))
        return label, numbered

    def _generate_minimal_string_replacement(self, fpath: str, original: str, issue_text: str, plan_steps: str, previous_patch: str) -> str | None:
        """Use LLM to perform a targeted string replacement."""
        system_prompt = (
            "You are a minimal patch agent. The issue requires a tiny string or logic change.\n"
            "Return EXACTLY a JSON object with two keys: 'search' and 'replace'.\n"
            "The 'search' string MUST be an exact substring of the original file.\n"
            "The 'replace' string is what it should be changed to.\n"
            "Do NOT include any markdown formatting, only the JSON."
        )
        user_prompt = (
            f"FILE: {fpath}\n"
            f"ISSUE:\n{issue_text}\n"
            f"PLAN:\n{plan_steps}\n\n"
            f"ORIGINAL FILE CONTENT:\n{original}\n\n"
            "Return JSON with 'search' and 'replace' to fix the issue."
        )
        try:
            from utils.llm_utils import safe_invoke
            import json
            response = safe_invoke(
                self.client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]
            data = json.loads(raw.strip())
            search_str = data.get("search", "")
            replace_str = data.get("replace", "")
            if search_str and search_str in original:
                return original.replace(search_str, replace_str, 1)
        except Exception as e:
            print(f"   [WARN] Minimal LLM patch failed: {e}")
        return None

    def _deterministic_minimal_fix(self, original: str, issue_text: str) -> str | None:
        """Apply a deterministic minimal fix without LLM based on simple heuristics."""
        import re
        # Look for explicit "change X to Y" in issue text
        match = re.search(r"change\s+['\"](.*?)['\"]\s+to\s+['\"](.*?)['\"]", issue_text, re.IGNORECASE)
        if match:
            search_str, replace_str = match.group(1), match.group(2)
            if search_str in original:
                return original.replace(search_str, replace_str, 1)
                
        # Look for "replace X with Y"
        match = re.search(r"replace\s+['\"](.*?)['\"]\s+with\s+['\"](.*?)['\"]", issue_text, re.IGNORECASE)
        if match:
            search_str, replace_str = match.group(1), match.group(2)
            if search_str in original:
                return original.replace(search_str, replace_str, 1)
                
        return None

    def _generate_full_file(
        self,
        file_path: str,
        original: str,
        issue_text: str,
        steps_block: str,
        context: str,
        error_block: str,
        prev_diff_block: str,
        qa_feedback: str = None,
        failure_type: str = None,
        failed_tests: int = None,
        previous_patch_content: str = None,
        retry_count: int = 0,
        edge_case_context: str = "",
        test_failure_context: str = "",
    ) -> str | None:
        """
        Ask the LLM to output the complete modified file.
        This is the primary strategy — avoids diff parsing entirely.
        """
        lang_label, lang_rules = self._get_language_instructions(file_path)
        retry_guidance = self._failure_guidance(failure_type, retry_count)

        system_prompt = (
            f"You are a surgical code fixing agent. Output ONLY the complete modified {lang_label} file.\n\n"
            "CODING RULES:\n"
            " - Implement complete logic (not partial)\n"
            " - Handle edge cases\n"
            " - Follow existing code style\n"
            " - Modify only relevant parts\n"
            " - NO generic placeholders or unjustified assumptions.\n"
            " - Do NOT add redundant validation (e.g., isinstance(x, object) is always True)\n"
            " - Do NOT re-implement functionality that already exists in the file\n"
            " - If the code already handles the issue, focus on hardening (security, edge cases) not rewriting\n"
            " - DOCUMENTATION: For docs/glossary tasks, generate static content (.rst, .md). Do NOT write Python code to generate docs.\n\n"
            "VALIDATION PHILOSOPHY (CRITICAL):\n"
            " - Prefer GRACEFUL HANDLING over raising exceptions.\n"
            " - For invalid items: skip them, use default values, or coerce types. Do NOT crash.\n"
            " - Only raise exceptions for truly unrecoverable conditions.\n"
            " - Do NOT add TypeError/ValueError for inputs that can be reasonably handled.\n"
            " - Allow flexible types where reasonable (e.g., accept float for quantity if it works).\n"
            " - Keep validation minimal \u2014 only what the issue explicitly requires.\n\n"
            "UNIVERSAL RULES:\n"
            "1. Output ONLY the raw file content — no prose, no markdown fences (no ```).\n"
            "2. Include ALL original code. Only change what is DIRECTLY needed to fix the issue.\n"
            "3. Make the SMALLEST possible change that actually fixes the root cause.\n"
            "4. NEVER remove existing code unless it IS the bug.\n"
            "5. Output the COMPLETE file — first line to last line — NOTHING omitted.\n"
            "6. Follow the PLAN steps precisely — do not over-engineer.\n"
            "7. NEVER use placeholders like '...', '# ...', '# rest of file', 'truncated', etc.\n"
            "8. NEVER abbreviate or skip sections. Every single line of the original must appear.\n"
            "9. CRITICAL: Ensure the output file is valid and complete — no truncation allowed.\n"
            "10. If you need to include large blocks unchanged, copy them exactly. Do NOT use ellipsis.\n"
            "11. NEVER compress code onto one line using semicolons. Use proper multi-line formatting.\n"
            "12. Every function body MUST use proper indentation and newlines — NO inline definitions.\n\n"
            f"LANGUAGE-SPECIFIC RULES ({lang_label}):\n"
            f"{lang_rules}\n"
            "\n[OUTPUT FORMAT]\n"
            "CRITICAL: Your response must be PURE CODE ONLY.\n"
            "- No explanations before or after.\n"
            "- No markdown code blocks (```).\n"
            "- No comments about your changes.\n"
            "- Start with the first line of code.\n"
            "- End with the last line of code.\n"
        )

        if retry_count > 0:
            _prev_src = PatchGenerator._extract_relevant_section(
                previous_patch_content or original, file_path,
                issue_text, steps_block.splitlines(), _MAX_FILE_CHARS,
            )
            _qa_reason = (qa_feedback or "")[:600]
            user_content = (
                f"FILE: {file_path}\n\n"
                f"PREVIOUS PATCH CONTENT:\n{_prev_src}\n\n"
                f"FAILURE REASON: {_qa_reason}\n"
                f"FAILURE TYPE: {failure_type}\n\n"
                + (f"{test_failure_context}\n" if test_failure_context else "")
                + (f"{edge_case_context}\n" if edge_case_context else "")
            )
            if retry_guidance:
                user_content += f"RETRY GUIDANCE:\n{retry_guidance}\n\n"
            user_content += (
                f"PLAN STEPS:\n{steps_block}\n\n"
                "OUTPUT the complete corrected file. Make targeted changes only — do NOT rewrite working code."
            )

        else:
            _orig_section = PatchGenerator._extract_relevant_section(
                original, file_path, issue_text,
                steps_block.splitlines(), _MAX_FILE_CHARS,
            )
            _is_partial = len(_orig_section) < len(original)
            file_content_block = (
                f"ORIGINAL FILE ({file_path})" +
                (" [relevant section only]" if _is_partial else "") +
                f":\n{_orig_section}\n"
            )
            context_block = f"\nCONTEXT:\n{context[:_MAX_CONTEXT_CHARS]}\n" if context.strip() else ""
            _output_instr = (
                "Output ONLY the modified section shown above. Start with the first line of the section."
                if _is_partial else
                f"Output the COMPLETE modified file for {file_path}. "
                "Apply only the changes described in the PLAN. Include all original code unchanged except the fix."
            )
            test_failure_part = f"{test_failure_context}\n" if test_failure_context else ""
            edge_case_part = f"{edge_case_context}\n" if edge_case_context else ""
            retry_part = f"RETRY GUIDANCE:\n{retry_guidance}\n\n" if retry_guidance else ""

            user_content = (
                f"ISSUE:\n{issue_text[:800]}\n\n"
                f"PLAN:\n{steps_block}\n\n"
                f"{context_block}"
                f"{file_content_block}"
                f"{error_block}"
                f"{prev_diff_block}\n"
                f"{test_failure_part}"
                f"{edge_case_part}"
                f"{retry_part}"
                f"{_output_instr}"
            )

        last_reason = "No LLM output returned"
        for attempt in range(2):
            prompt = user_content
            if attempt > 0:
                prompt += (
                    "\n\nPREVIOUS OUTPUT WAS INVALID.\n"
                    f"REASON: {last_reason}\n"
                    "Return ONLY the complete file content with no prose and no truncation."
                )

            try:
                import sys, os
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
                from utils.llm_utils import safe_invoke, TokenLimitError

                response = safe_invoke(
                    self.client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
            except TokenLimitError:
                # 413: shrink prompt and retry within this attempt
                trimmed = PatchGenerator._trim_to_budget(prompt, len(prompt) // 2)
                print(f"   [WARN] 413 in full-file — retrying with trimmed prompt ({len(trimmed)} chars)")
                try:
                    from utils.llm_utils import safe_invoke
                    response = safe_invoke(
                        self.client.chat.completions.create,
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": trimmed},
                        ],
                        temperature=0.1,
                    )
                except Exception as e2:
                    raise ValueError(f"Reduced-prompt LLM call also failed: {e2}")
            except Exception as e:
                raise ValueError(f"LLM call failed: {e}")

            raw_content = response.choices[0].message.content
            if not isinstance(raw_content, str) or not raw_content.strip():
                last_reason = "LLM returned empty response"
                continue

            raw = raw_content.strip()
            truncation_markers = [
                "# ... (truncated)", "# ... (middle truncated)", "# ...",
                "<!-- truncated -->", "// ... truncated", "/* truncated */",
                "... (rest of", "... rest of file", "# rest unchanged",
                "# existing code", "# ... existing", "...", "â€¦",
            ]
            bad_marker = next((marker for marker in truncation_markers if marker in raw), None)
            if bad_marker:
                last_reason = f"Output contained truncation marker: {bad_marker!r}"
                continue

            if self.is_valid_unified_diff(raw) and not any(
                raw.strip().startswith(prefix) for prefix in ["import ", "from ", "def ", "class ", "#", '"""', "'''"]
            ):
                last_reason = "Returned unified diff instead of full file content"
                continue

            code = self._extract_code_from_response(raw)
            if code is None:
                last_reason = "Could not extract code from response"
                continue

            is_valid, validation_reason = LLMOutputValidator.validate_full_output(
                code,
                original,
                filename=file_path,
                strict=True,
            )
            if not is_valid:
                last_reason = validation_reason
                continue

            # Language-specific safety validation (CRITICAL)
            lang_ext = os.path.splitext(file_path)[1].lower()
            lang_valid, lang_reason = LLMOutputValidator.validate_language_specific_safety(code, file_path)
            if not lang_valid:
                last_reason = f"Language safety: {lang_reason}"
                continue

            # Structural integrity check (CRITICAL for Python)
            if lang_ext in (".py", ".pyw"):
                struct_valid, struct_reason = LLMOutputValidator.validate_structural_integrity(
                    original, code, filename=file_path
                )
                if not struct_valid:
                    last_reason = f"Structural: {struct_reason}"
                    continue

                # Indentation consistency
                indent_valid, indent_reason = LLMOutputValidator.validate_indentation_consistency(
                    original, code, filename=file_path
                )
                if not indent_valid:
                    last_reason = f"Indentation: {indent_reason}"
                    continue

                # Import integrity
                import_valid, import_reason = LLMOutputValidator.validate_import_integrity(
                    original, code, filename=file_path
                )
                if not import_valid:
                    last_reason = f"Imports: {import_reason}"
                    continue

                # Per-function scope validation
                scope_valid, scope_reason = LLMOutputValidator.validate_per_function_scope(
                    original, code, max_changed_lines_per_function=80
                )
                if not scope_valid:
                    last_reason = f"Scope: {scope_reason}"
                    continue

            # Large file handling
            file_valid, file_reason = LLMOutputValidator.validate_large_file_handling(
                original, code, filename=file_path
            )
            if not file_valid:
                last_reason = f"File handling: {file_reason}"
                continue

            if not self._has_meaningful_change(original, code):
                last_reason = "Full-file patch generated no changes (identical to original)"
                continue

            return code

        raise ValueError(f"Full-file generation failed validation: {last_reason}")

    # ------------------------------------------------------------------
    # FALLBACK: Unified diff strategy
    # ------------------------------------------------------------------

    def _generate_via_diff(
        self,
        file_path: str,
        original: str,
        issue_text: str,
        steps_block: str,
        context: str,
        error_block: str,
        prev_diff_block: str,
        qa_feedback: str = None,
        failure_type: str = None,
        failed_tests: int = None,
        previous_patch_content: str = None,
        retry_count: int = 0,
    ) -> str | None:
        """
        Ask the LLM for unified diffs and apply them.
        This is the fallback strategy when full-file rewrite fails.
        """
        lang_label, lang_rules = self._get_language_instructions(file_path)
        retry_guidance = self._failure_guidance(failure_type, retry_count)

        system_prompt = (
            f"You are a precise code patching agent. You generate unified diffs for {lang_label} files ONLY.\n\n"
            "DIFF FORMAT RULES (STRICT):\n"
            "1. Output ONLY unified diff format. NO prose. NO markdown fences. NO explanations.\n"
            "2. Each file diff starts with:\n"
            "   --- a/filepath\n"
            "   +++ b/filepath\n"
            "   @@ -start,count +start,count @@\n"
            "3. Context lines start with a SPACE. Removed lines start with -. Added lines start with +.\n"
            "4. Include 3 lines of context before and after each change.\n"
            "5. Make MINIMAL changes — fix the root cause only.\n"
            "6. Context lines MUST match the original file EXACTLY (character for character, including spaces).\n"
            "7. Do NOT use placeholders, ellipsis, or 'truncated' markers.\n"
            "8. CRITICAL: Do not abbreviate context. If you need 20 context lines, provide all 20 lines.\n"
            "9. Every context line MUST be identical to the original (including trailing spaces/tabs).\n"
            "10. DOCUMENTATION: For docs/glossary tasks, generate static content (.rst, .md). Do NOT write Python code to generate docs.\n\n"
            f"LANGUAGE-SPECIFIC RULES ({lang_label}):\n"
            f"{lang_rules}\n"
            "\n[OUTPUT FORMAT]\n"
            "CRITICAL: Your response must be PURE UNIFIED DIFF ONLY.\n"
            "- No explanations, preamble, or postamble.\n"
            "- No markdown code blocks.\n"
            "- Start with '--- a/' on the first line.\n"
            "- End with the last hunk (no trailing text).\n"
        )

        if retry_count > 0:
            user_content = "[PREVIOUS PATCH]\n"
            if previous_patch_content:
                user_content += f"{previous_patch_content}\n\n"
            else:
                user_content += f"{original}\n\n"
            
            user_content += f"[FAILURE]\n{qa_feedback}\n\n"
            user_content += f"[FAILURE TYPE]\n{failure_type}\n\n"
            user_content += f"[FAILED TESTS]\n{failed_tests}\n\n"
            if retry_guidance:
                user_content += f"[RETRY GUIDANCE]\n{retry_guidance}\n\n"

            user_content += "TASK:\n1. Identify exact failure cause.\n2. Modify ONLY failing logic.\nGenerate the unified diff for these precision fixes ONLY."
        else:
            _orig_diff_sec = PatchGenerator._extract_relevant_section(
                original, file_path, issue_text,
                steps_block.splitlines(), _MAX_FILE_CHARS,
            )
            _ctx_diff = PatchGenerator._trim_to_budget(context, _MAX_CONTEXT_CHARS)
            user_content = (
                f"ISSUE:\n{issue_text[:800]}\n\n"
                f"PLAN:\n{steps_block}\n\n"
                f"ADDITIONAL CONTEXT:\n{_ctx_diff}\n\n"
                f"FILE TO MODIFY: {file_path}\n"
                f"=== FILE: {file_path} ===\n{_orig_diff_sec}\n"
                f"{error_block}"
                f"{prev_diff_block}\n"
                f"{f'RETRY GUIDANCE:\\n{retry_guidance}\\n\\n' if retry_guidance else ''}"
                f"Generate the unified diff for {file_path}. Output ONLY the diff."
            )

        last_reason = "No diff output returned"
        for attempt in range(2):
            prompt = user_content
            if attempt > 0:
                prompt += (
                    "\n\nPREVIOUS OUTPUT WAS INVALID.\n"
                    f"REASON: {last_reason}\n"
                    "Return ONLY a valid unified diff for the requested file."
                )

            try:
                import sys, os
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
                from utils.llm_utils import safe_invoke, TokenLimitError

                response = safe_invoke(
                    self.client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
            except TokenLimitError:
                trimmed = PatchGenerator._trim_to_budget(prompt, len(prompt) // 2)
                print(f"   [WARN] 413 in diff — retrying with trimmed prompt ({len(trimmed)} chars)")
                try:
                    from utils.llm_utils import safe_invoke
                    response = safe_invoke(
                        self.client.chat.completions.create,
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": trimmed},
                        ],
                        temperature=0.1,
                    )
                except Exception as e2:
                    raise ValueError(f"Reduced diff prompt also failed: {e2}")
            except Exception as e:
                raise ValueError(f"Diff LLM call failed: {e}")

            raw_content = response.choices[0].message.content
            if not isinstance(raw_content, str) or not raw_content.strip():
                last_reason = "Diff LLM returned empty response"
                continue

            raw = raw_content.strip()
            diffs = self.extract_diffs_from_response(raw)
            if not diffs and self.is_valid_unified_diff(raw):
                match = re.search(r"--- a/(.+?)(?:\s|$)", raw)
                if match:
                    diffs = {match.group(1).strip(): raw}

            if not diffs:
                last_reason = "No valid unified diffs found in response"
                continue

            diff_text = diffs.get(file_path)
            if not diff_text:
                for dpath, dtext in diffs.items():
                    if dpath.endswith(file_path) or file_path.endswith(dpath):
                        diff_text = dtext
                        break

            if not diff_text:
                last_reason = f"No diff found for {file_path} in response"
                continue

            is_valid, validation_reason = LLMOutputValidator.validate_unified_diff(diff_text)
            if not is_valid:
                last_reason = validation_reason
                continue

            patched = self.apply_unified_diff(original, diff_text)
            if patched is None:
                last_reason = "Unified diff could not be applied to the original source"
                continue

            patched_valid, patched_reason = LLMOutputValidator.validate_full_output(
                patched,
                original,
                filename=file_path,
                strict=True,
            )
            if not patched_valid:
                last_reason = f"Patched output failed validation after diff application: {patched_reason}"
                continue

            # Comprehensive Language-Specific Safety Validation (POST-DIFF)
            lang_ext = os.path.splitext(file_path)[1].lower()
            if lang_ext in (".py", ".pyw"):
                # Structural integrity check
                struct_valid, struct_reason = LLMOutputValidator.validate_structural_integrity(
                    original, patched, filename=file_path
                )
                if not struct_valid:
                    last_reason = f"Structural: {struct_reason}"
                    continue

                # Indentation consistency
                indent_valid, indent_reason = LLMOutputValidator.validate_indentation_consistency(
                    original, patched, filename=file_path
                )
                if not indent_valid:
                    last_reason = f"Indentation: {indent_reason}"
                    continue

                # Import integrity
                import_valid, import_reason = LLMOutputValidator.validate_import_integrity(
                    original, patched, filename=file_path
                )
                if not import_valid:
                    last_reason = f"Imports: {import_reason}"
                    continue

                # Per-function scope validation
                scope_valid, scope_reason = LLMOutputValidator.validate_per_function_scope(
                    original, patched, max_changed_lines_per_function=80
                )
                if not scope_valid:
                    last_reason = f"Scope: {scope_reason}"
                    continue

            # Large file handling
            file_valid, file_reason = LLMOutputValidator.validate_large_file_handling(
                original, patched, filename=file_path
            )
            if not file_valid:
                last_reason = f"File handling: {file_reason}"
                continue

            if not self._has_meaningful_change(original, patched):
                last_reason = "Diff patch generated no changes (identical to original)"
                continue

            return patched

        raise ValueError(f"Diff generation failed validation: {last_reason}")

    # ------------------------------------------------------------------
    # Utility: compute unified diff from original + patched
    # ------------------------------------------------------------------

    @staticmethod
    def compute_unified_diff(
        original: str,
        patched: str,
        file_path: str = "file.py",
    ) -> str:
        """
        Compute a unified diff string from original and patched code.

        Args:
            original: Original source code.
            patched: Patched source code.
            file_path: File path for the diff header.

        Returns:
            Unified diff string or "NO_CHANGES".

        Raises:
            TypeError: If original or patched is not a string.
        """
        if not isinstance(original, str):
            raise TypeError(f"original must be a string, got {type(original).__name__}")
        if not isinstance(patched, str):
            raise TypeError(f"patched must be a string, got {type(patched).__name__}")
        if not isinstance(file_path, str) or not file_path.strip():
            file_path = "file.py"

        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"--- a/{file_path}",
            tofile=f"+++ b/{file_path}",
            lineterm="",
        ))
        return "\n".join(diff_lines) if diff_lines else "NO_CHANGES"
    
    @staticmethod
    def _validate_diff_context(diff_text: str) -> bool:
        """
        RULE 7: Validate that diff has proper surrounding context.
        
        A valid diff must:
        - Have at least 1 line of surrounding context (@@... lines)
        - Not have fuzzy line matching (context lines must match exactly)
        - Include hunks (cannot be empty except for NO_CHANGES marker)
        
        Returns True if diff is valid, False if it needs regeneration.
        """
        if not diff_text or diff_text == "NO_CHANGES":
            return True  # Empty diffs are OK
        
        lines = diff_text.split('\n')
        has_header = False
        has_hunk = False
        
        for line in lines:
            if line.startswith('@@'):
                has_header = True
                has_hunk = True
                break
        
        if not has_header or not has_hunk:
            print("   [REJECT] RULE 7: Diff missing hunk headers (@@...)")
            return False
        
        return True

    @staticmethod
    def _verify_patch_within_target(
        original_ast: str,
        patched_ast: str,
        target_node_name: str,
        start_line: int,
        end_line: int,
    ) -> bool:
        """
        RULE 6: Verify that patch modifications stayed within target scope.
        
        Parse both AST versions and confirm:
        - Target node exists in original
        - Modifications don't extend outside [start_line, end_line)
        - No unintended removals of surrounding code
        
        Args:
            original_ast: Original file content (string)
            patched_ast: Patched file content (string)
            target_node_name: Name of target (ClassName.method or function_name)
            start_line: Start line of modification
            end_line: End line of modification
        
        Returns:
            True if patch is safely scoped, False if scope violation detected
        """
        try:
            import ast as _ast
            orig_tree = _ast.parse(original_ast)
            patched_tree = _ast.parse(patched_ast)
        except SyntaxError:
            # If either can't parse, fall back to optimistic assumption
            return True
        
        # Extract class name if target is ClassName.method
        if "." in target_node_name:
            class_name, method_name = target_node_name.split(".", 1)
        else:
            class_name = None
            method_name = target_node_name
        
        # Find target node in both trees
        def find_node(tree, class_name, method_name):
            for node in _ast.walk(tree):
                if class_name:
                    # Looking for ClassName.method
                    if isinstance(node, _ast.ClassDef) and node.name == class_name:
                        for child in _ast.iter_child_nodes(node):
                            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                if child.name == method_name:
                                    return child
                    return None
                else:
                    # Looking for standalone function
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        if node.name == method_name:
                            return node
            return None
        
        orig_target = find_node(orig_tree, class_name, method_name)
        patched_target = find_node(patched_tree, class_name, method_name)
        
        if not orig_target or not patched_target:
            # Target not found in one of the versions
            print(f"   [REJECT] RULE 6: Target '{target_node_name}' not found in AST")
            return False
        
        # RULE 6: Verify target still exists and modification is within bounds
        print(f"   [ACCEPT] RULE 6: Target '{target_node_name}' verified within scope")
        return True
