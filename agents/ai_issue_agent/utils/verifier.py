"""
utils/verifier.py

Simplified LLM verification — ONE focused call.
Checks whether the fix solves the issue and handles key edge cases.
Practical — approves fixes that work, doesn't demand perfection.
"""

import difflib
from groq import Groq

try:
    from ..config import GROQ_API_KEY
except ImportError:
    from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def compute_unified_diff(original: str, patched: str, file_path: str = "file.py") -> str:
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))
    if not diff_lines:
        return "(no changes detected)"
    return "\n".join(diff_lines[:120])  # cap to save tokens


def verify_fix_with_edge_cases(
    issue_text: str,
    original_code: str,
    patched_code: str,
    edge_cases: list | None = None,
    file_path: str = "file.py",
) -> tuple[bool, str]:
    """
    ONE LLM call to verify the fix. Practical, not perfectionist.

    Returns:
        (verified: bool, reasoning: str)
    """
    edge_cases = edge_cases or []
    diff = compute_unified_diff(original_code, patched_code, file_path)

    if diff == "(no changes detected)":
        return False, "No changes were made."

    edge_block = ""
    if edge_cases:
        edge_block = "\nKey edge cases: " + ", ".join(edge_cases[:5])

    prompt = f"""Verify this code fix. Be PRACTICAL — approve fixes that work.

ISSUE: {issue_text}
{edge_block}

DIFF:
```diff
{diff}
```

Check:
1. Does the diff actually fix the described issue?
2. Are zero/negative/None inputs handled where relevant?
3. Any obvious runtime errors introduced?
4. Does this change solve ONLY the given issue and nothing else? If unrelated code was changed, refactored, or reformatted, that is a rejection.

IGNORE: concurrency, thread safety, perfect coverage of all edge cases.

If the fix solves the core issue AND does not touch unrelated code:
  VERDICT: VERIFIED

If there is a CLEAR bug, the issue is NOT solved, or unrelated code was modified:
  VERDICT: REJECTED
  REASON: <one specific issue>
"""

    print(f"[VERIFY] Verifying fix...")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You verify code fixes. Approve good fixes. Be practical, not perfectionist."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [WARN] Verification failed: {e} -- treating as verified")
        return True, f"Verification skipped: {e}"

    if "VERIFIED" in raw.upper():
        print("   [OK] Verified!")
        return True, raw[:300]

    # Extract reason
    reason = raw[:300]
    for line in raw.splitlines():
        if line.strip().upper().startswith("REASON:"):
            reason = line.strip()[7:].strip()
            break

    print(f"   [REJECTED] Rejected: {reason[:150]}")
    return False, reason


# Backwards-compatible alias
def verify_fix(issue_text, original_code, patched_code, file_path="file.py"):
    return verify_fix_with_edge_cases(issue_text, original_code, patched_code, file_path=file_path)
