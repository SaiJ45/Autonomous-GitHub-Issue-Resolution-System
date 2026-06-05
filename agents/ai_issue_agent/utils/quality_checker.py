"""
utils/quality_checker.py

Language-aware quality gate. NO LLM calls.

Routes checks by file type detected from extension:
- Python: full AST checks (syntax, imports, classes, bare-except)
- HTML/templates: structural checks (unclosed tags, missing artifacts)
- JS/TS/CSS/others: minimal checks (not empty, not identical, no truncation)
"""

import ast
import os
import re


# ---------------------------------------------------------------------------
# Language detection (pure logic, no hardcoded framework names)
# ---------------------------------------------------------------------------

PYTHON_EXTS = {".py", ".pyw"}
HTML_EXTS   = {".html", ".htm", ".jinja", ".jinja2", ".j2"}
JS_EXTS     = {".js", ".jsx", ".mjs", ".cjs"}
TS_EXTS     = {".ts", ".tsx"}
CSS_EXTS    = {".css", ".scss", ".sass", ".less"}


def detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in PYTHON_EXTS: return "python"
    if ext in HTML_EXTS:   return "html"
    if ext in JS_EXTS:     return "javascript"
    if ext in TS_EXTS:     return "typescript"
    if ext in CSS_EXTS:    return "css"
    return "unknown"


# ---------------------------------------------------------------------------
# Universal checks (all languages)
# ---------------------------------------------------------------------------

def _check_not_identical(original: str, candidate: str, lang: str = "unknown"):
    if original.strip() == candidate.strip():
        return "Code is identical to original — no fix was applied"
    
    if lang == "python":
        try:
            orig_tree = ast.parse(original)
            cand_tree = ast.parse(candidate)
            if ast.dump(orig_tree) == ast.dump(cand_tree):
                return "No behavioral changes detected (AST is identical). Only formatting/comments changed."
        except SyntaxError:
            pass



def _check_no_markdown(candidate: str):
    if "```" in candidate:
        return "Markdown code fences leaked into generated code"

def _check_no_truncation_artifacts(candidate: str):
    artifacts = [
        "# ... (truncated)", "# ... (middle truncated)",
        "<!-- truncated -->", "// ... truncated", "/* truncated */",
    ]
    for a in artifacts:
        if a in candidate:
            return f"Truncation artifact found: '{a}' — LLM output incomplete"


# ---------------------------------------------------------------------------
# Python-only checks
# ---------------------------------------------------------------------------

def _check_valid_python_syntax(candidate: str):
    try:
        ast.parse(candidate)
    except SyntaxError as e:
        return f"Syntax error: {e}"

def _check_no_bare_except_pass(candidate: str):
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                return "except: pass found — silently swallows all errors"

def _check_no_unrelated_imports(original: str, candidate: str):
    def _get_imports(code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    new_imports = _get_imports(candidate) - _get_imports(original)
    if len(new_imports) > 8:
        return f"Too many new imports ({len(new_imports)}): {', '.join(sorted(new_imports))}"

def _check_no_class_additions(original: str, candidate: str):
    def _get_classes(code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    new_classes = _get_classes(candidate) - _get_classes(original)
    if len(new_classes) > 3:
        return f"Too many new classes added ({', '.join(new_classes)})"


# ---------------------------------------------------------------------------
# HTML / template checks
# Detects template syntax generically by delimiter patterns ({%, {{, {#, etc.)
# Works for Django, Jinja2, Handlebars, Liquid, Nunjucks, etc.
# ---------------------------------------------------------------------------

def _check_template_tags_preserved(original: str, candidate: str):
    """
    Count template delimiters in original vs candidate.
    Flags if existing tags drop by >10% (LLM stripped template logic).
    Language-agnostic: matches any {% %} / {{ }} / {# #} style.
    """
    for pattern, name in [
        (r"\{%", "{%"), (r"\{\{", "{{"), (r"\{#", "{#"),
    ]:
        orig_n = len(re.findall(pattern, original))
        cand_n = len(re.findall(pattern, candidate))
        if orig_n > 5 and cand_n < orig_n * 0.9:
            return (
                f"Template tag '{name}' count dropped from {orig_n} to {cand_n} "
                f"— template logic may have been stripped"
            )

def _check_html_structural_tags(original: str, candidate: str):
    """Ensure major structural tags from original are still present."""
    if len(original.strip()) < 50:
        return None
    for tag in set(re.findall(r"<(html|head|body|form|table|script|style)\b", original, re.I)):
        if f"<{tag.lower()}" not in candidate.lower():
            return f"Structural tag <{tag}> was removed"


# ---------------------------------------------------------------------------
# JS/TS checks
# ---------------------------------------------------------------------------

def _check_bracket_balance(candidate: str):
    """Verify balanced brackets/braces/parens in JS/TS code."""
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_string = None
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch in ('"', "'", "`"):
            if in_string == ch:
                in_string = None
            elif in_string is None:
                in_string = ch
            continue
        if in_string:
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return f"Unbalanced bracket: '{ch}' without matching opener"
            stack.pop()
    if stack:
        return f"Unclosed brackets: {''.join(stack)}"


def _check_js_ts_functions_preserved(original: str, candidate: str):
    """Ensure exported functions/classes from the original are preserved."""
    # Match: function name, const name =, class Name, export default
    def _extract_symbols(code):
        symbols = set()
        # function declarations
        for m in re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', code):
            symbols.add(m.group(1))
        # arrow/const declarations
        for m in re.finditer(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=', code):
            symbols.add(m.group(1))
        # class declarations
        for m in re.finditer(r'(?:export\s+)?class\s+(\w+)', code):
            symbols.add(m.group(1))
        return symbols

    orig_syms = _extract_symbols(original)
    cand_syms = _extract_symbols(candidate)
    removed = orig_syms - cand_syms
    if removed:
        return f"Removed symbols: {', '.join(sorted(removed)[:5])}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_checklist(
    original: str,
    candidate: str,
    issue_text: str,
    edge_cases: list | None = None,
    file_path: str = "",
) -> tuple[bool, list[str]]:
    """
    Language-aware quality gate. Routes checks based on file extension.
    Returns (passed: bool, failures: list[str]).
    """
    lang = detect_language(file_path)
    print(f"[CHECK] Language={lang} | {file_path or '(unknown)'}")

    checks = [
        _check_not_identical(original, candidate, lang=lang),
        _check_no_markdown(candidate),
        _check_no_truncation_artifacts(candidate),
    ]

    if lang == "python":
        checks += [
            _check_valid_python_syntax(candidate),
            _check_no_bare_except_pass(candidate),
            _check_no_unrelated_imports(original, candidate),
            _check_no_class_additions(original, candidate),
        ]
    elif lang == "html":
        checks += [
            _check_template_tags_preserved(original, candidate),
            _check_html_structural_tags(original, candidate),
        ]
    elif lang in ("javascript", "typescript"):
        checks += [
            _check_bracket_balance(candidate),
            _check_js_ts_functions_preserved(original, candidate),
        ]

    failures = [msg for msg in checks if msg is not None]

    if failures:
        for f in failures:
            print(f"   [FAIL] {f}")
        print(f"[FAIL] {len(failures)} quality issue(s)")
    else:
        print("   [OK] All quality checks passed")

    return len(failures) == 0, failures


def format_quality_failures_for_prompt(failures: list[str]) -> str:
    if not failures:
        return ""
    return "\nQUALITY ISSUES TO FIX:\n" + "\n".join(f"  - {f}" for f in failures) + "\n"
