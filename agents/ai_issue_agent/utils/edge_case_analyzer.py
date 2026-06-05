"""
utils/edge_case_analyzer.py

Lightweight edge case analysis — capped at 5-7 cases MAX.
No separate LLM call. Uses simple heuristic extraction from the issue text
and source code to identify ONLY relevant edge cases.

Forbidden topics (filtered out):
  - concurrency / thread safety
  - argument ordering
  - recursion
  - system-level concerns
"""

import re


# ---------------------------------------------------------------------------
# Forbidden edge case topics (always filtered out)
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    r"concurren",
    r"thread",
    r"lock",
    r"mutex",
    r"race\s*condition",
    r"argument\s*order",
    r"recursion",
    r"recursive",
    r"system[- ]level",
    r"file\s*system",
    r"network",
    r"socket",
    r"shared\s*resource",
    r"reentran",
    r"signal",
    r"deadlock",
]


def _is_forbidden(case: str) -> bool:
    lower = case.lower()
    return any(re.search(p, lower) for p in _FORBIDDEN_PATTERNS)


# ---------------------------------------------------------------------------
# Heuristic edge case extraction (NO LLM call)
# ---------------------------------------------------------------------------

def _extract_edge_cases_from_context(issue_text: str, source_code: str) -> list[str]:
    """
    Derive 5-7 relevant edge cases from the issue text and code.
    Uses pattern matching — zero LLM cost.
    """
    cases = []
    lower_issue = issue_text.lower()
    lower_code = source_code.lower()

    # --- Numeric / math related ---
    math_keywords = ["divide", "division", "emi", "calculate", "compute",
                     "formula", "rate", "principal", "interest", "average",
                     "sum", "multiply", "percent", "total", "tenure"]
    is_math = any(kw in lower_issue or kw in lower_code for kw in math_keywords)

    if is_math:
        cases.append("zero value passed as divisor or denominator (ZeroDivisionError)")
        cases.append("negative numeric input where only positive is valid")
        cases.append("None passed instead of a number")
        cases.append("non-numeric type (e.g. string) passed as argument")
        cases.append("very large values that may cause overflow")

    # --- String related ---
    string_keywords = ["format", "string", "name", "text", "parse", "user"]
    is_string = any(kw in lower_issue or kw in lower_code for kw in string_keywords)

    if is_string:
        cases.append("empty string passed as input")
        cases.append("None passed instead of a string")
        cases.append("non-string type passed where string expected")

    # --- Collection related ---
    collection_keywords = ["list", "items", "order", "array", "dict"]
    is_collection = any(kw in lower_issue or kw in lower_code for kw in collection_keywords)

    if is_collection:
        cases.append("empty list/dict passed as input")
        cases.append("None passed instead of a list/dict")

    # --- Always include if not covered ---
    if not cases:
        cases.append("None passed as input")
        cases.append("invalid type passed as argument")
        cases.append("empty/zero input value")

    # --- General boundary ---
    if len(cases) < 5:
        cases.append("boundary value at zero")

    # Deduplicate and cap at 7
    seen = set()
    unique = []
    for c in cases:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:7]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_edge_cases(issue_text: str, source_code: str) -> list[str]:
    """
    Return 5-7 relevant edge cases.  NO LLM call — pure heuristic.

    Returns:
        List of edge case description strings (max 7).
    """
    print("\n[ANALYZE] Analyzing edge cases...")

    edge_cases = _extract_edge_cases_from_context(issue_text, source_code)

    # Safety filter
    edge_cases = [ec for ec in edge_cases if not _is_forbidden(ec)][:7]

    print(f"[OK] {len(edge_cases)} edge case(s) identified:")
    for ec in edge_cases:
        print(f"   - {ec}")

    return edge_cases


def format_edge_cases_for_prompt(edge_cases: list[str]) -> str:
    """Format edge cases for insertion into LLM prompts."""
    if not edge_cases:
        return ""
    lines = "\n".join(f"  - {ec}" for ec in edge_cases)
    return f"\nKEY EDGE CASES TO HANDLE:\n{lines}\n"
