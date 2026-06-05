"""
git_tools/commit_push.py

Scope-guarded commit and push.
Before committing, validates that ONLY allowed files were modified
and enforces hard limits on file count and diff size.
"""

import subprocess
import re


def _get_current_branch(cwd: str = "repo_clone") -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_changed_files(cwd: str = "repo_clone") -> list[str]:
    """Return list of files that have been modified/added/deleted."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    changed = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]

    # Also check untracked files
    result2 = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    untracked = [f.strip() for f in result2.stdout.strip().splitlines() if f.strip()]

    return list(set(changed + untracked))


def get_diff_line_count(cwd: str = "repo_clone") -> tuple[int, int]:
    """Return (additions, deletions) from the current diff."""
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    # Count actual diff lines
    diff_result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    additions = 0
    deletions = 0
    for line in diff_result.stdout.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


# Hard limits (raised for multi-file support)
MAX_CHANGED_FILES = 5       # max 5 files
MAX_DIFF_LINES = 500        # reject if total diff exceeds 500 lines


def commit_and_push(message: str, allowed_files: list[str] | None = None):
    """
    Commit and push changes with scope validation.

    Args:
        message: Commit message
        allowed_files: List of file paths that are allowed to be modified.
                      If provided, any changes to other files will cause ABORT.

    """
    current_branch = _get_current_branch()
    if current_branch in {"main", "master", ""}:
        return {
            "success": False,
            "error": f"Unsafe branch for commit: {current_branch or 'unknown'}",
        }

    # Check for changes
    diff_check = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd="repo_clone",
        capture_output=True,
        text=True,
    )

    if not diff_check.stdout.strip():
        print("[WARN] No changes to commit")
        return {"success": False, "error": "No changes to commit"}

    # ── Pre-commit scope guard ────────────────────────────────────────────
    changed = get_changed_files()

    # Hard file-count limit
    if len(changed) > MAX_CHANGED_FILES:
        print(f"[BLOCKED] PRE-COMMIT GUARD: Too many files changed ({len(changed)} > {MAX_CHANGED_FILES})")
        for f in changed:
            print(f"   [FAIL] {f}")
        print("   ABORTING commit — max file limit exceeded")
        return {"success": False, "error": f"Too many files changed: {len(changed)} > {MAX_CHANGED_FILES}"}

    # Diff line-count limit
    additions, deletions = get_diff_line_count()
    total_diff = additions + deletions
    print(f"   [STAT] Diff size: +{additions} -{deletions} = {total_diff} lines")
    if total_diff > MAX_DIFF_LINES:
        print(f"[BLOCKED] PRE-COMMIT GUARD: Diff too large ({total_diff} > {MAX_DIFF_LINES} lines)")
        print("   ABORTING commit — diff size limit exceeded")
        return {"success": False, "error": f"Diff too large: {total_diff} > {MAX_DIFF_LINES}"}

    # Allowed files validation
    if allowed_files:
        # Normalize paths for comparison
        allowed_normalized = set()
        for f in allowed_files:
            norm = f.replace("\\", "/").strip("/")
            allowed_normalized.add(norm)
            # Also allow without repo_clone prefix
            if norm.startswith("repo_clone/"):
                allowed_normalized.add(norm[len("repo_clone/"):])

        violations = []
        for f in changed:
            norm_f = f.replace("\\", "/").strip("/")

            if norm_f not in allowed_normalized:
                violations.append(norm_f)

        if violations:
            print(f"[BLOCKED] PRE-COMMIT GUARD: Unauthorized file changes detected!")
            for v in violations:
                print(f"   [FAIL] {v}")
            print(f"   Allowed: {allowed_normalized}")
            print("   ABORTING commit — only allowed files may be modified")
            return {"success": False, "error": f"Unauthorized changes to: {violations}"}

    print(f"[OK] Pre-commit guard passed -- {len(changed)} file(s) changed, all within scope")

    # ── Commit and push ───────────────────────────────────────────────────
    cmds = [
        ["git", "add", "."],
        ["git", "commit", "-m", message],
        ["git", "push", "-u", "origin", "HEAD"],
    ]

    for cmd in cmds:
        result = subprocess.run(
            cmd,
            cwd="repo_clone",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

    return {"success": True}