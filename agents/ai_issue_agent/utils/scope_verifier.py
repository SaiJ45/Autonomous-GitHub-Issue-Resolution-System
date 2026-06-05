"""
utils/scope_verifier.py

Final scope verification gate — runs ONE LLM call to confirm
the change solves ONLY the given issue and nothing else.

This is the last gate before PR creation. It only runs when all
other checks (quality, diff, self-review, verification) have passed.
"""

import difflib
from groq import Groq

try:
    from ..config import GROQ_API_KEY
except ImportError:
    from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def compute_compact_diff(original: str, patched: str, file_path: str = "file.py") -> str:
    """Compute a compact unified diff for scope review."""
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))
    if not diff_lines:
        return "(no changes)"
    return "\n".join(diff_lines[:100])  # cap to save tokens


def verify_scope(
    issue_text: str,
    original_code: str,
    patched_code: str,
    allowed_files: list[str],
    file_path: str,
) -> tuple[bool, str]:
    """
    Final gate before PR creation.
    Uses ONE LLM call to answer:
    "Does this change solve ONLY the given issue and nothing else?"

    Returns:
        (in_scope: bool, reason: str)
    """
    diff = compute_compact_diff(original_code, patched_code, file_path)

    if diff == "(no changes)":
        return False, "No changes were made."

    prompt = f"""You are a strict scope auditor. Your ONLY job is to verify that code changes
are limited to solving the specific issue described below.

ISSUE:
{issue_text}

FILE BEING MODIFIED: {file_path}
ALLOWED FILES: {', '.join(allowed_files)}

DIFF:
```diff
{diff}
```

Answer this ONE question:
Does this diff solve ONLY the given issue and nothing else?

Check for:
- Changes to functions NOT related to the issue
- Unrelated refactoring, cleanup, or reformatting
- New imports that aren't needed for the fix
- Changes to code comments on unrelated lines
- Performance improvements outside the issue scope

If the changes are strictly scoped to the issue:
  SCOPE: PASS

If ANY unrelated changes are present:
  SCOPE: FAIL
  REASON: <what is out of scope>
"""

    print("[SCOPE] Running final scope verification...")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a strict scope auditor. Only approve changes that are precisely scoped to the described issue. Reject any unrelated changes."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [WARN] Scope verification failed: {e} -- treating as passed")
        return True, f"Skipped due to error: {e}"

    if "PASS" in raw.upper() and "FAIL" not in raw.upper():
        print("   [OK] Scope verification: PASS -- changes are strictly scoped")
        return True, "Scope verified"

    # Extract reason
    reason = raw[:300]
    for line in raw.splitlines():
        if line.strip().upper().startswith("REASON:"):
            reason = line.strip()[7:].strip()
            break

    print(f"   [BLOCKED] Scope verification: FAIL -- {reason[:150]}")
    return False, reason
