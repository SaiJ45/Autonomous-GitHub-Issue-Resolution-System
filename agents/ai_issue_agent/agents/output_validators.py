"""
agents/output_validators.py

Strict validation for LLM outputs before applying patches.
Prevents truncation, incomplete code, syntax errors, and other LLM failures
from corrupting the codebase.

This is the critical gatekeeper layer between LLM generation and patch application.
"""

import re
import ast
from typing import Tuple, Optional


class LLMOutputValidator:
    """
    Validates LLM outputs for correctness, completeness, and safety
    before they are used as code patches.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Truncation Detection
    # ─────────────────────────────────────────────────────────────────────────

    TRUNCATION_MARKERS = [
        "# ... (truncated)",
        "# ... (middle truncated)",
        "# ...",
        "<!-- truncated -->",
        "// ... truncated",
        "/* truncated */",
        "... (rest of",
        "... rest of file",
        "# rest unchanged",
        "# existing code",
        "# ... existing",
        "[truncated]",
        "(truncated)",
        "... (output truncated)",
    ]

    PLACEHOLDER_MARKERS = [
        "...",  # Careful: also used in legitimate code
        "…",
        "[implementation here]",
        "[your code here]",
        "[ implementation ]",
        "TODO",
        "FIXME",
    ]

    @classmethod
    def detect_truncation(cls, text: str, filename: str = "") -> Tuple[bool, str]:
        """
        Detect if the LLM output appears to be truncated.

        Args:
            text: The LLM output string.
            filename: Optional filename for context (used in messages).

        Returns:
            (is_truncated: bool, reason: str)
        """
        if not isinstance(text, str):
            return True, "Input is not a string"

        # Check for known truncation markers
        for marker in cls.TRUNCATION_MARKERS:
            if marker in text:
                return True, f"Contains truncation marker: {marker!r}"

        # Check for incomplete code blocks (opening fences without closing)
        open_count = text.count("```")
        if open_count % 2 != 0:
            return True, "Unclosed code fence (```)"

        if filename.endswith(".py"):
            try:
                ast.parse(text)
                return False, "OK"  # Valid syntax means it's complete
            except SyntaxError:
                pass

        # Check if file ends abruptly (e.g., mid-function or mid-dict)
        if any(text.rstrip().endswith(marker) for marker in [",", "(", "[", "{"]):
            return True, "Output ends with incomplete statement"

        return False, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Size & Completeness Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_size(cls, output: str, original: str, min_ratio: float = 0.0) -> Tuple[bool, str]:
        """
        Validate that the output is not empty and not absurdly large.

        Does NOT reject based on size ratio — small patches and partial
        file outputs (e.g. a single modified function) are valid.

        Args:
            output: The LLM output.
            original: The original source code.
            min_ratio: Unused — kept for API compatibility.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(output, str) or not isinstance(original, str):
            return False, "Output or original is not a string"

        out_len = len(output)
        if out_len == 0 or not output.strip():
            return False, "Output is empty"

        # Only reject if output is absurdly larger than original
        # (suggests LLM hallucinated code)
        orig_len = len(original)
        if orig_len > 0:
            ratio = out_len / orig_len
            if ratio > 3.0:
                return (
                    False,
                    f"Output suspiciously large: {out_len} chars vs {orig_len} original "
                    f"(ratio: {ratio:.1%} > 300%)",
                )

        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Syntax Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_python_syntax(cls, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax for a code string.

        Args:
            code: The Python source code.

        Returns:
            (is_valid, reason_or_error)
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        try:
            compile(code, "<string>", "exec")
            return True, "OK"
        except SyntaxError as e:
            return (
                False,
                f"SyntaxError at line {e.lineno}: {e.msg} ({e.text})",
            )
        except Exception as e:
            return False, f"Compilation error: {type(e).__name__}: {e}"

    @classmethod
    def validate_json_syntax(cls, code: str) -> Tuple[bool, str]:
        """
        Validate JSON syntax.

        Args:
            code: The JSON source.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        import json
        try:
            json.loads(code)
            return True, "OK"
        except json.JSONDecodeError as e:
            return False, f"JSONDecodeError: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"JSON error: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # Structure Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_bracket_balance(cls, code: str) -> Tuple[bool, str]:
        """
        Validate that brackets, braces, and parentheses are balanced.

        Args:
            code: The code string.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        # Remove strings to avoid false positives from quoted brackets
        code_no_strings = re.sub(r'["\'].*?["\']', "", code, flags=re.DOTALL)

        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []

        for char in code_no_strings:
            if char in pairs.values():
                stack.append(char)
            elif char in pairs:
                if not stack or stack[-1] != pairs[char]:
                    return False, f"Mismatched bracket: {char}"
                stack.pop()

        if stack:
            return False, f"Unclosed bracket(s): {stack}"

        return True, "OK"

    @classmethod
    def validate_complete_functions(cls, code: str) -> Tuple[bool, str]:
        """
        Validate that all function definitions are complete (have bodies).

        Args:
            code: The code string.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error while validating function bodies: {exc}"

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(node.body or [])
            if not body:
                return False, f"Function '{node.name}' has no body"

            meaningful = []
            for child in body:
                if isinstance(child, ast.Expr) and isinstance(getattr(child, "value", None), ast.Constant):
                    if isinstance(child.value.value, str):
                        continue
                meaningful.append(child)

            if not meaningful:
                return False, f"Function '{node.name}' only contains a docstring"
            if len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass):
                return False, f"Function '{node.name}' has an empty pass body"
            if len(meaningful) == 1 and isinstance(meaningful[0], ast.Expr):
                value = getattr(meaningful[0], "value", None)
                if isinstance(value, ast.Constant) and value.value is Ellipsis:
                    return False, f"Function '{node.name}' has an ellipsis placeholder body"

        return True, "OK"

    @classmethod
    def validate_formatting(cls, code: str) -> Tuple[bool, str]:
        """
        Detect compressed/one-line code that should be multi-line.
        Catches patterns like: `def foo(): x = 1; y = 2; return x + y`

        Args:
            code: The Python source code.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        lines = code.splitlines()
        compressed_lines = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Detect compressed function definitions with semicolons
            # e.g., `def foo(): x = 1; y = 2; return x`
            if re.match(r"^\s*(?:def|async def|class)\s+", line):
                # Count semicolons that are NOT inside strings
                no_strings = re.sub(r'["\'].*?["\']', '', stripped)
                semicolons = no_strings.count(";")
                if semicolons >= 2:
                    compressed_lines.append(
                        f"Line {i}: compressed function with {semicolons} semicolons"
                    )

            # Detect any line with 3+ semicolons (compressed multi-statement)
            else:
                no_strings = re.sub(r'["\'].*?["\']', '', stripped)
                semicolons = no_strings.count(";")
                if semicolons >= 3:
                    compressed_lines.append(
                        f"Line {i}: {semicolons} semicolons — compressed multi-statement"
                    )

        if compressed_lines:
            detail = "; ".join(compressed_lines[:3])
            return False, f"Compressed/one-line code detected: {detail}"

        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Language-specific Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_by_extension(
        cls, code: str, filename: str
    ) -> Tuple[bool, str]:
        """
        Perform language-specific validation based on file extension.

        Args:
            code: The code string.
            filename: The filename (used to determine language).

        Returns:
            (is_valid, reason)
        """
        if not isinstance(code, str) or not isinstance(filename, str):
            return False, "Invalid inputs"

        ext = filename.lower().split(".")[-1] if "." in filename else ""

        # Python validation
        if ext in ("py", "pyw", "pyi"):
            is_valid, reason = cls.validate_python_syntax(code)
            if not is_valid:
                return False, reason
            is_valid, reason = cls.validate_bracket_balance(code)
            if not is_valid:
                return False, reason
            return True, "OK (Python)"

        # JSON validation
        if ext == "json":
            is_valid, reason = cls.validate_json_syntax(code)
            return is_valid, reason if is_valid else reason

        # HTML/XML: basic bracket check
        if ext in ("html", "htm", "xml"):
            is_valid, reason = cls.validate_bracket_balance(code)
            if not is_valid:
                return False, reason
            # Check for unclosed tags
            if "<" in code and ">" not in code:
                return False, "Unclosed HTML/XML tag"
            return True, "OK (HTML/XML)"

        # For other extensions, just check brackets
        is_valid, reason = cls.validate_bracket_balance(code)
        return is_valid, reason

    # ─────────────────────────────────────────────────────────────────────────
    # Comprehensive Validation Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_full_output(
        cls,
        output: str,
        original: str,
        filename: str = "unknown",
        strict: bool = True,
    ) -> Tuple[bool, str]:
        """
        Comprehensive validation of LLM output before using it as a patch.

        Checks:
        1. Not truncated or containing placeholders
        2. Reasonably complete size
        3. Balanced brackets/parens
        4. Syntax validation (language-specific)
        5. Complete function bodies

        Args:
            output: The LLM output.
            original: The original source code (for comparison).
            filename: The filename (for language detection).
            strict: If True, reject on any warning; if False, only reject critical errors.

        Returns:
            (is_valid, reason_string)
        """
        if not isinstance(output, str):
            return False, "Output is not a string"

        # 1. Truncation check (CRITICAL)
        is_truncated, reason = cls.detect_truncation(output, filename)
        if is_truncated:
            return False, f"[TRUNCATION] {reason}"

        # 2. Content check — reject only truly empty output
        #    Do NOT reject based on size ratio: small patches and
        #    partial file outputs are valid and expected.
        if not output.strip():
            return False, "[SIZE] Output is empty"

        # 3. Bracket balance (CRITICAL)
        is_valid, reason = cls.validate_bracket_balance(output)
        if not is_valid:
            return False, f"[BRACKET] {reason}"

        # 4. Complete functions (CRITICAL for Python)
        if filename.endswith(".py"):
            is_valid, reason = cls.validate_complete_functions(output)
            if not is_valid:
                return False, f"[STRUCTURE] {reason}"

        # 5. Language-specific validation (syntax correctness)
        is_valid, reason = cls.validate_by_extension(output, filename)
        if not is_valid:
            return False, f"[SYNTAX] {reason}"

        # 6. Compressed-code detection (Python)
        if filename.endswith(".py"):
            is_valid, reason = cls.validate_formatting(output)
            if not is_valid:
                return False, f"[FORMATTING] {reason}"

        return True, "VALID"

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Integrity Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_structural_integrity(
        cls,
        original: str,
        patched: str,
        filename: str = "",
    ) -> Tuple[bool, str]:
        """
        Validate that patching did not remove classes, major function
        definitions, or nested methods that were present in the original.

        Args:
            original: The original source code.
            patched: The patched source code.
            filename: File path (used for Python detection).

        Returns:
            (is_valid, reason)
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return True, "OK (non-string input — skipping)"

        if not filename.endswith(".py"):
            return True, "OK (non-Python file)"

        try:
            orig_tree = ast.parse(original)
            patch_tree = ast.parse(patched)
        except SyntaxError:
            return True, "OK (syntax error — validated elsewhere)"

        def _collect_definitions(tree) -> dict:
            """Collect all function/class definitions at all nesting levels."""
            defs = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[node.name] = "function"
                elif isinstance(node, ast.ClassDef):
                    defs[node.name] = "class"
                    # Collect methods within the class
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            defs[f"{node.name}.{child.name}"] = "method"
            return defs

        orig_defs = _collect_definitions(orig_tree)
        patch_defs = _collect_definitions(patch_tree)

        # Detect removed definitions
        removed = set(orig_defs.keys()) - set(patch_defs.keys())
        if removed:
            removed_items = [f"{orig_defs[n]}:{n}" for n in sorted(removed)]
            return (
                False,
                f"Patch removes {len(removed)} definition(s): {', '.join(removed_items[:8])}",
            )

        return True, "OK"

    @classmethod
    def analyze_python_changes(cls, original: str, patched: str) -> dict:
        analysis = {
            "ast_changed": False,
            "changed_definitions": [],
            "added_definitions": [],
            "removed_definitions": [],
            "empty_definitions": [],
            "imports_added": [],
            "imports_removed": [],
            "removed_imports_still_used": [],
            "module_body_changed": False,
            "parse_error": "",
        }
        if not isinstance(original, str) or not isinstance(patched, str):
            return analysis

        try:
            orig_tree = ast.parse(original)
            patched_tree = ast.parse(patched)
        except SyntaxError as exc:
            analysis["parse_error"] = str(exc)
            return analysis

        analysis["ast_changed"] = (
            ast.dump(orig_tree, include_attributes=False)
            != ast.dump(patched_tree, include_attributes=False)
        )

        def _collect_defs(tree) -> dict:
            defs = {}

            def _walk(nodes, prefix=""):
                for node in nodes:
                    if isinstance(node, ast.ClassDef):
                        qname = f"{prefix}{node.name}"
                        defs[qname] = ast.dump(node, include_attributes=False)
                        _walk(node.body, prefix=f"{qname}.")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qname = f"{prefix}{node.name}"
                        defs[qname] = ast.dump(node, include_attributes=False)

            _walk(getattr(tree, "body", []))
            return defs

        def _collect_empty_defs(tree) -> list[str]:
            empty = []
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                meaningful = []
                for child in node.body:
                    if isinstance(child, ast.Expr) and isinstance(getattr(child, "value", None), ast.Constant):
                        if isinstance(child.value.value, str):
                            continue
                    meaningful.append(child)
                if not meaningful:
                    empty.append(node.name)
                elif len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass):
                    empty.append(node.name)
                elif len(meaningful) == 1 and isinstance(meaningful[0], ast.Expr):
                    value = getattr(meaningful[0], "value", None)
                    if isinstance(value, ast.Constant) and value.value is Ellipsis:
                        empty.append(node.name)
            return sorted(set(empty))

        def _collect_imports(tree) -> dict[str, str]:
            imports = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imports[name] = f"{module}.{alias.name}".strip(".")
            return imports

        def _collect_used_names(tree) -> set[str]:
            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.add(node.id)
            return used

        orig_defs = _collect_defs(orig_tree)
        patched_defs = _collect_defs(patched_tree)
        analysis["added_definitions"] = sorted(set(patched_defs) - set(orig_defs))
        analysis["removed_definitions"] = sorted(set(orig_defs) - set(patched_defs))
        analysis["changed_definitions"] = sorted(
            name for name in set(orig_defs) & set(patched_defs)
            if orig_defs[name] != patched_defs[name]
        )
        analysis["empty_definitions"] = _collect_empty_defs(patched_tree)

        orig_imports = _collect_imports(orig_tree)
        patched_imports = _collect_imports(patched_tree)
        analysis["imports_added"] = sorted(set(patched_imports.values()) - set(orig_imports.values()))
        analysis["imports_removed"] = sorted(set(orig_imports.values()) - set(patched_imports.values()))
        patched_used = _collect_used_names(patched_tree)
        analysis["removed_imports_still_used"] = sorted(
            name for name, import_path in orig_imports.items()
            if import_path in analysis["imports_removed"] and name in patched_used
        )

        orig_module_nodes = [
            ast.dump(node, include_attributes=False)
            for node in getattr(orig_tree, "body", [])
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        patched_module_nodes = [
            ast.dump(node, include_attributes=False)
            for node in getattr(patched_tree, "body", [])
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        analysis["module_body_changed"] = orig_module_nodes != patched_module_nodes

        return analysis

    @classmethod
    def validate_import_integrity(
        cls,
        original: str,
        patched: str,
        filename: str = "",
    ) -> Tuple[bool, str]:
        if not isinstance(filename, str) or not filename.endswith(".py"):
            return True, "OK"
        analysis = cls.analyze_python_changes(original, patched)
        removed_still_used = analysis.get("removed_imports_still_used", [])
        if removed_still_used:
            return (
                False,
                f"Import(s) removed but still referenced: {', '.join(removed_still_used[:8])}",
            )
        return True, "OK"

    @classmethod
    def validate_behavioral_change(
        cls,
        original: str,
        patched: str,
        filename: str = "",
        target_symbols: list[str] | None = None,
    ) -> Tuple[bool, str]:
        if not isinstance(original, str) or not isinstance(patched, str):
            return False, "Original/patched content missing"
        if original == patched:
            return False, "Patched content is identical to original"
        if not isinstance(filename, str) or not filename.endswith(".py"):
            return True, "OK"

        analysis = cls.analyze_python_changes(original, patched)
        if analysis.get("parse_error"):
            return False, f"Unable to analyze patched AST: {analysis['parse_error']}"
        if analysis.get("empty_definitions"):
            return False, (
                f"Patch introduces empty function bodies: "
                f"{', '.join(analysis['empty_definitions'][:8])}"
            )
        if not analysis.get("ast_changed"):
            return False, "No functional change detected (AST is unchanged)"

        changed_defs = set(analysis.get("changed_definitions", []))
        added_defs = set(analysis.get("added_definitions", []))
        changed_targets = changed_defs | added_defs

        normalized_targets = []
        for symbol in target_symbols or []:
            if not isinstance(symbol, str):
                continue
            cleaned = symbol.strip().replace("::", ".")
            if cleaned:
                normalized_targets.append(cleaned)

        if normalized_targets:
            matched = []
            for target in normalized_targets:
                bare = target.split(".")[-1]
                if target in changed_targets or bare in changed_targets:
                    matched.append(target)
                    continue
                if any(item.endswith(f".{bare}") for item in changed_targets):
                    matched.append(target)
            if not matched and not analysis.get("module_body_changed") and not analysis.get("imports_added") and not analysis.get("imports_removed"):
                return False, (
                    f"Patch does not modify the grounded target symbol(s): "
                    f"{', '.join(normalized_targets[:6])}"
                )

        if not changed_targets and not analysis.get("module_body_changed") and not analysis.get("imports_added") and not analysis.get("imports_removed"):
            return False, "No meaningful behavioral change detected"

        return True, "OK"

    @classmethod
    def validate_per_function_scope(
        cls,
        original: str,
        patched: str,
        max_changed_lines_per_function: int = 60,
    ) -> Tuple[bool, str]:
        """
        Validate that no single function has too many lines changed.
        Prevents full-function rewrites disguised as targeted fixes.

        Args:
            original: The original source code.
            patched: The patched source code.
            max_changed_lines_per_function: Max lines changed within any
                single function body.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return True, "OK"

        try:
            orig_tree = ast.parse(original)
            patch_tree = ast.parse(patched)
        except SyntaxError:
            return True, "OK (syntax error — validated elsewhere)"

        orig_lines = original.splitlines()
        patch_lines = patched.splitlines()

        def _get_function_bodies(tree, lines):
            bodies = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    bodies[node.name] = lines[start:end]
                elif isinstance(node, ast.ClassDef):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            start = child.lineno - 1
                            end = getattr(child, "end_lineno", child.lineno)
                            key = f"{node.name}.{child.name}"
                            bodies[key] = lines[start:end]
            return bodies

        orig_bodies = _get_function_bodies(orig_tree, orig_lines)
        patch_bodies = _get_function_bodies(patch_tree, patch_lines)

        for fn_name, orig_body in orig_bodies.items():
            patch_body = patch_bodies.get(fn_name)
            if patch_body is None:
                continue  # removed function — caught by structural integrity
            import difflib
            diff = list(difflib.unified_diff(orig_body, patch_body, n=0))
            changed = sum(
                1 for l in diff
                if (l.startswith("+") or l.startswith("-"))
                and not l.startswith("+++")
                and not l.startswith("---")
            )
            if changed > max_changed_lines_per_function:
                return (
                    False,
                    f"Function '{fn_name}' has {changed} changed lines "
                    f"(max {max_changed_lines_per_function}). "
                    f"Restrict changes to the minimal fix.",
                )

        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Diff Validation
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_unified_diff(cls, diff_text: str) -> Tuple[bool, str]:
        """
        Validate that a string is a proper unified diff.

        Args:
            diff_text: The unified diff text.

        Returns:
            (is_valid, reason)
        """
        if not isinstance(diff_text, str) or not diff_text.strip():
            return False, "Diff is empty or not a string"

        lines = diff_text.strip().splitlines()

        # Check for required markers
        has_minus = any(line.startswith("--- ") for line in lines)
        has_plus = any(line.startswith("+++ ") for line in lines)
        has_hunk = any(line.startswith("@@ ") for line in lines)

        if not (has_minus and has_plus and has_hunk):
            return False, "Missing required diff markers (---, +++, @@)"

        # Check for valid hunk headers
        hunk_pattern = r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@"
        has_valid_hunks = any(re.match(hunk_pattern, line) for line in lines)

        if not has_valid_hunks:
            return False, "No valid hunk headers found"

        # Check that diff is not too large
        change_lines = [
            l for l in lines
            if (l.startswith("+") or l.startswith("-"))
            and not l.startswith("+++")
            and not l.startswith("---")
        ]
        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Language-Specific Safe Editing
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def validate_typescript_javascript_syntax(cls, code: str) -> Tuple[bool, str]:
        """
        Validate TypeScript/JavaScript syntax (basic checks).
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        # Check for common TS/JS structural issues
        issues = []

        # Check for unmatched braces/parens
        is_valid, reason = cls.validate_bracket_balance(code)
        if not is_valid:
            return False, reason

        # Check for missing semicolons (common in TS/JS)
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            # Flag incomplete function/class definitions
            if stripped.endswith("{") and ("function" in stripped or "class" in stripped):
                next_line = (lines[i].strip() if i < len(lines) else "").strip()
                if not next_line or next_line == "}":
                    issues.append(f"Line {i}: Empty function/class body")

        if issues:
            return False, "; ".join(issues[:5])

        return True, "OK (TypeScript/JavaScript)"

    @classmethod
    def validate_java_syntax(cls, code: str) -> Tuple[bool, str]:
        """
        Validate Java syntax (basic structural checks).
        """
        if not isinstance(code, str):
            return False, "Input is not a string"

        # Check for bracket balance
        is_valid, reason = cls.validate_bracket_balance(code)
        if not is_valid:
            return False, reason

        # Check for common Java issues
        issues = []

        # If contains class definition, ensure proper structure
        if "class " in code:
            if "{" not in code or "}" not in code:
                return False, "Class definition without body braces"
            # Count opening and closing braces
            open_count = code.count("{")
            close_count = code.count("}")
            if open_count != close_count:
                return False, f"Unbalanced braces: {open_count} opening, {close_count} closing"

        # Check for incomplete method bodies
        method_pattern = r"(?:public|private|protected)?\s+(?:static\s+)?(?:synchronized\s+)?[\w<>,\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w., ]+)?\s*\{"
        methods = re.finditer(method_pattern, code)
        for match in methods:
            method_name = match.group(1)
            method_start = match.end()
            # Scan for closing brace
            brace_count = 1
            found_close = False
            for i, char in enumerate(code[method_start:], method_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        found_close = True
                        break
            if not found_close:
                issues.append(f"Method {method_name}: unclosed body")

        if issues:
            return False, "; ".join(issues[:5])

        return True, "OK (Java)"

    @classmethod
    def validate_indentation_consistency(cls, original: str, patched: str, filename: str = "") -> Tuple[bool, str]:
        """
        Validate that indentation is consistent between original and patched.
        Detects improper indentation that breaks block structure.
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return True, "OK"

        # Only check Python files strictly
        if not filename.endswith(".py"):
            return True, "OK (non-Python)"

        orig_lines = original.splitlines()
        patch_lines = patched.splitlines()

        def _get_indent_profile(lines) -> dict:
            """Get indentation statistics."""
            indents = {}
            for line in lines:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                indents[indent] = indents.get(indent, 0) + 1
            return indents

        orig_indents = _get_indent_profile(orig_lines)
        patch_indents = _get_indent_profile(patch_lines)

        # Check that new indentation levels are consistent with original
        for new_indent in patch_indents:
            if new_indent not in orig_indents:
                # Check if it's a multiple of 4 or 2 (standard Python)
                if new_indent % 4 != 0 and new_indent % 2 != 0:
                    return False, f"Invalid indentation level: {new_indent} spaces"

        return True, "OK"

    @classmethod
    def validate_scope_isolation(
        cls,
        original: str,
        patched: str,
        target_function: str | None = None,
        filename: str = "",
    ) -> Tuple[bool, str]:
        """
        Validate that changes are isolated to the target function/method.
        Ensures only intended scope is modified.
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return True, "OK"

        if not filename.endswith(".py"):
            return True, "OK (non-Python)"

        if original == patched:
            return False, "No changes made to patched file"

        try:
            orig_tree = ast.parse(original)
            patch_tree = ast.parse(patched)
        except SyntaxError:
            return True, "OK (syntax validation done elsewhere)"

        # If target_function specified, verify only that function changed
        if target_function and isinstance(target_function, str):
            target_parts = target_function.split(".")
            target_name = target_parts[-1]

            # Find all functions in original
            orig_funcs = {}
            for node in ast.walk(orig_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    orig_funcs[node.name] = ast.dump(node, include_attributes=False)

            patch_funcs = {}
            for node in ast.walk(patch_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    patch_funcs[node.name] = ast.dump(node, include_attributes=False)

            # Check that only the target function changed (or was added)
            unintended_changes = []
            for func_name, orig_body in orig_funcs.items():
                if func_name == target_name:
                    continue  # Target function is allowed to change
                patch_body = patch_funcs.get(func_name)
                if patch_body != orig_body:
                    unintended_changes.append(func_name)

            if unintended_changes:
                return (
                    False,
                    f"Patch modifies unintended functions: {', '.join(unintended_changes[:5])}. "
                    f"Target was {target_name}",
                )

        return True, "OK"

    @classmethod
    def validate_large_file_handling(cls, original: str, patched: str, filename: str = "") -> Tuple[bool, str]:
        """
        For large files, ensure only relevant sections are modified.
        Detects when patches affect too much of the file (suggests full-file rewrite).
        """
        if not isinstance(original, str) or not isinstance(patched, str):
            return True, "OK"

        orig_lines = original.splitlines()
        patch_lines = patched.splitlines()

        # Define "large" as >500 lines
        if len(orig_lines) <= 500:
            return True, "OK (file size ≤ 500 lines)"

        # For large files, ensure patch affects <30% of lines
        import difflib
        diff = list(difflib.unified_diff(orig_lines, patch_lines, lineterm=""))
        changed = sum(1 for l in diff if l.startswith("+") or l.startswith("-"))
        affected_lines = changed / len(orig_lines)

        if affected_lines > 0.3:
            return (
                False,
                f"Large file patch too extensive: {affected_lines:.1%} of file changed "
                f"(max 30%). Use targeted function extraction.",
            )

        return True, "OK"

    @classmethod
    def validate_language_specific_safety(
        cls,
        code: str,
        filename: str = "",
    ) -> Tuple[bool, str]:
        """
        Comprehensive language-specific safety validation.
        """
        if not isinstance(code, str) or not isinstance(filename, str):
            return False, "Invalid inputs"

        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext in ("ts", "tsx"):
            return cls.validate_typescript_javascript_syntax(code)
        elif ext in ("js", "jsx"):
            return cls.validate_typescript_javascript_syntax(code)
        elif ext == "java":
            return cls.validate_java_syntax(code)
        elif ext in ("py", "pyw"):
            return cls.validate_python_syntax(code)

        return True, "OK"
