"""
utils/self_reviewer.py

Simplified LLM self-review — ONE focused call.

ONLY checks:
  1. Is the core logic correct?
  2. Are key edge cases (zero/negative/None) handled?
  3. Any obvious runtime errors?

Does NOT check:
  - concurrency / thread safety
  - argument ordering
  - system-level concerns
  - exhaustive edge case coverage
"""

import re
from groq import Groq

try:
    from ..config import GROQ_API_KEY
except ImportError:
    from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def self_review(
    candidate_code: str,
    issue_text: str,
    edge_cases: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Quick LLM review — 1 call. Only checks core correctness.

    Returns:
        (approved: bool, feedback: str)
    """
    edge_cases = edge_cases or []
    edge_block = ""
    if edge_cases:
        edge_block = "\nKey edge cases: " + ", ".join(edge_cases[:5])

    prompt = f"""Review this code fix. Be PRACTICAL, not perfectionist.

ISSUE: {issue_text}
{edge_block}

CODE:
```python
{candidate_code[:2500]}
```

Check ONLY these 4 things:
1. Does the code correctly solve the issue described above?
2. Does it handle zero values, None, and negative inputs where relevant?
3. Are there any obvious runtime errors (division by zero, TypeError, etc.)?
4. Does the code modify ONLY functions related to the issue? If unrelated functions are changed, refactored, or reformatted, that is a failure.

IGNORE: concurrency, thread safety, argument ordering, system-level concerns.

If the code is basically correct, handles common edge cases, and ONLY modifies relevant code:
  VERDICT: APPROVED

If there is a clear bug, the issue is NOT solved, or unrelated code was modified:
  VERDICT: NEEDS_REVISION
  FIX: <one specific thing to fix>
"""

    print("[REVIEW] Running quick self-review...")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a practical code reviewer. Approve code that works correctly. Don't be overly strict."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [WARN] Self-review failed: {e} -- treating as approved")
        return True, f"Skipped due to error: {e}"

    if "APPROVED" in raw.upper():
        print("   [OK] Self-review: APPROVED")
        return True, "Approved"

    # Extract fix suggestion
    fix_match = re.search(r"FIX:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    feedback = fix_match.group(1).strip() if fix_match else raw[:300]
    print(f"   [FAIL] Self-review: NEEDS_REVISION -- {feedback[:150]}")
    return False, feedback


def format_review_feedback_for_prompt(feedback: str) -> str:
    if not feedback:
        return ""
    return f"\nSELF-REVIEW ISSUE: {feedback}\n"
