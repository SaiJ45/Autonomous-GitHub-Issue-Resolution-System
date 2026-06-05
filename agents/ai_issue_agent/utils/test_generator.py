"""
utils/test_generator.py

Simplified test generator for simulation-first workflows.
Generates candidate tests without executing them.
"""

import os
import ast

from groq import Groq

try:
    from ..config import GROQ_API_KEY, CLONE_PATH
except ImportError:
    from config import GROQ_API_KEY, CLONE_PATH

client = Groq(api_key=GROQ_API_KEY)

AGENT_TEST_FILENAME = "test_agent_generated.py"


# ---------------------------------------------------------------------------
# LLM call (SINGLE attempt)
# ---------------------------------------------------------------------------

def _call_llm_for_tests(
    issue_text: str,
    file_path: str,
    source_code: str,
    edge_cases: list | None = None,
) -> str | None:
    """Generate pytest tests in ONE LLM call."""

    rel = file_path.replace("\\", "/").lstrip("./")
    module = rel.replace("/", ".").removesuffix(".py")

    edge_block = ""
    if edge_cases:
        edge_lines = "\n".join(f"  - {ec}" for ec in edge_cases[:7])
        edge_block = f"\nEDGE CASES TO TEST:\n{edge_lines}\n"

    prompt = f"""Write 3-5 pytest test functions.

ISSUE: {issue_text}

TARGET FILE: {file_path}
MODULE: {module}

SOURCE CODE:
```python
{source_code[:2500]}
```
{edge_block}
RULES:
- Import from: `from {module} import <name>`
- Tests MUST FAIL on the buggy code above
- Tests MUST PASS once the issue is correctly fixed
- Include: 1 normal case, 2-3 edge cases (zero, negative, None)
- Use pytest.raises for expected exceptions
- Output ONLY Python code. No markdown. No explanations.
- Start with import statements.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Output ONLY valid Python pytest code. No markdown. No prose."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Test generation LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    lines = text.splitlines()
    out = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        out.append(line)
    return "\n".join(out).strip()


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _has_test_functions(code: str) -> bool:
    try:
        tree = ast.parse(code)
        return any(
            isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            for n in ast.walk(tree)
        )
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tests(
    issue_text: str,
    file_path: str,
    source_code: str,
    max_regenerate: int = 2,
    edge_cases: list | None = None,
) -> str | None:
    """
    Generate pytest tests — max 2 attempts (not 3).
    Returns path to test file or None.
    """
    test_file_path = os.path.join(CLONE_PATH, AGENT_TEST_FILENAME)
    edge_cases = edge_cases or []

    for attempt in range(1, max_regenerate + 1):
        print(f"\n🧪 Generating tests (attempt {attempt}/{max_regenerate})...")

        raw = _call_llm_for_tests(issue_text, file_path, source_code, edge_cases=edge_cases)

        if raw is None:
            print("❌ LLM call failed — skipping")
            continue

        code = _strip_markdown(raw)

        if not _is_valid_python(code):
            print("❌ Generated tests have syntax errors")
            continue

        if not _has_test_functions(code):
            print("❌ No test_ functions found")
            continue

        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Tests written to {test_file_path}")

        # Runtime execution removed; return generated tests for simulation pipeline.
        print("ℹ️  Runtime execution disabled — forwarding generated tests to simulation layer.")
        return test_file_path

    print("❌ Could not generate failing tests")
    return test_file_path if os.path.exists(test_file_path) else None


def cleanup_generated_tests():
    test_file_path = os.path.join(CLONE_PATH, AGENT_TEST_FILENAME)
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"🗑️  Removed {AGENT_TEST_FILENAME}")
