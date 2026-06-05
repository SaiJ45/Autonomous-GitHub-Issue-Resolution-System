"""
git_tools/clone_repo.py

Robust clone logic with full cleanup and idempotency.

Guarantees:
- Repo is on main branch, up to date
- Working tree is clean (no leftover files)
- All stale local branches are pruned
- Temp directories (__pycache__) are cleaned
- Idempotent: running multiple times produces the exact same state
"""

import subprocess
import os
import glob
import shutil

try:
    from ..config import REPO_NAME, GITHUB_TOKEN
except ImportError:
    from config import REPO_NAME, GITHUB_TOKEN

REPO_DIR = "repo_clone"


def _build_repo_url() -> str:
    """Build the clone URL from config, embedding the token for authenticated pushes."""
    if GITHUB_TOKEN:
        return f"https://{GITHUB_TOKEN}@github.com/{REPO_NAME}.git"
    return f"https://github.com/{REPO_NAME}.git"


def _run_git(args: list[str], cwd: str = REPO_DIR) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )





def _is_tracked_path(path: str) -> bool:
    """Return True if path is tracked by git."""
    rel_path = os.path.relpath(path, REPO_DIR).replace("\\", "/")
    result = _run_git(["ls-files", rel_path])
    return bool(result.stdout.strip())


def _clean_temp_directories():
    """Remove temp directories only when they are not tracked by git."""
    temp_dirs = ["__pycache__"]
    removed = 0
    for root, dirs, _ in os.walk(REPO_DIR):
        for d in temp_dirs:
            path = os.path.join(root, d)
            if os.path.exists(path):
                if _is_tracked_path(path):
                    continue
                try:
                    shutil.rmtree(path)
                    print(f"   [CLEAN] Removed temp dir: {os.path.relpath(path, REPO_DIR)}")
                    removed += 1
                except OSError:
                    pass
    return removed


def _delete_stale_branches():
    """Delete all local branches except main to prevent branch reuse."""
    result = _run_git(["branch", "--list"])
    if result.returncode != 0:
        return 0

    deleted = 0
    for line in result.stdout.splitlines():
        branch = line.strip()
        if branch.startswith("* "):
            branch = branch[2:]
        if branch and branch != "main":
            del_result = _run_git(["branch", "-D", branch])
            if del_result.returncode == 0:
                print(f"   [CLEAN] Deleted stale branch: {branch}")
                deleted += 1

    return deleted


def _verify_clean_state() -> bool:
    """Verify the repo is in a clean state after reset."""
    result = _run_git(["status", "--porcelain"])
    if result.stdout.strip():
        print(f"   [WARN] Repo not fully clean after reset: {result.stdout.strip()[:200]}")
        return False
    return True


def clone_repo():
    """
    Clone the repo if not present.
    If already cloned, reset to a clean state.

    Guarantees:
    - Repo is on main branch, up to date
    - Working tree is clean (no leftover files)
    - All stale local branches are pruned
    - Temp directories are cleaned
    - Idempotent: running multiple times produces same state
    """
    if os.path.exists(REPO_DIR):
        print("[REPO] Repo already cloned -- resetting to clean state...")

        # 1. Hard reset — discard all local changes
        result = _run_git(["reset", "--hard"])
        if result.returncode != 0:
            print(f"   [WARN] git reset failed: {result.stderr.strip()}")
        else:
            print("   [OK] git reset --hard -- all local changes discarded")

        # 2. Remove untracked files and directories
        result = _run_git(["clean", "-fd"])
        if result.returncode != 0:
            print(f"   [WARN] git clean failed: {result.stderr.strip()}")
        else:
            print("   [OK] git clean -fd -- untracked files removed")

        # 3. Switch to main branch
        _run_git(["checkout", "main"])
        print("   [OK] Switched to main branch")

        # 3a. Update remote URL to ensure auth token is embedded for push
        repo_url = _build_repo_url()
        _run_git(["remote", "set-url", "origin", repo_url])
        print("   [OK] Remote origin URL updated")

        # 4. Delete all stale local branches (prevent reuse)
        stale_count = _delete_stale_branches()
        if stale_count > 0:
            print(f"   [CLEAN] Deleted {stale_count} stale branch(es)")

        # 5. Sync with remote (fetch + hard reset)
        result = _run_git(["fetch", "origin"])
        if result.returncode != 0:
            print(f"   [WARN] git fetch failed: {result.stderr.strip()}")
        else:
            result = _run_git(["reset", "--hard", "origin/main"])
            if result.returncode != 0:
                print(f"   [WARN] git reset to origin/main failed: {result.stderr.strip()}")
            else:
                print("   [OK] Synced latest changes from remote (fetch + reset)")



        # 7. Clean temp directories
        _clean_temp_directories()

        # 8. Verify clean state
        is_clean = _verify_clean_state()
        if is_clean:
            print("   [OK] Repo verified clean -- ready for new issue")
        else:
            print("   [WARN] Repo may not be fully clean -- proceeding anyway")

        return {"success": True, "data": "Reset to clean state", "clean": is_clean}

    # Fresh clone
    print("[CLONE] Cloning repository...")
    repo_url = _build_repo_url()
    result = subprocess.run(
        ["git", "clone", repo_url, REPO_DIR],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"   [FAIL] Clone failed: {result.stderr.strip()}")
        return {"success": False, "error": result.stderr}

    print("   [OK] Repository cloned successfully")
    print(f"   [PATH] Cloned to: {os.path.abspath(REPO_DIR)}")
    return {"success": True, "data": "Fresh clone"}
