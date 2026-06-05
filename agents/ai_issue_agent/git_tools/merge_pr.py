import subprocess

def merge_pull_request(pr_number):
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd="repo_clone",
        capture_output=True,
        text=True,
    )
    current_branch = branch_result.stdout.strip()
    if current_branch in {"main", "master", ""}:
        return {
            "success": False,
            "error": f"Unsafe branch for merge operation: {current_branch or 'unknown'}",
        }

    print("[MERGE] Checking mergeability...")

    # Try normal merge first
    result = subprocess.run(
        ["gh", "pr", "merge", str(pr_number), "--merge"],
        cwd="repo_clone",
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return {"success": True}

    print("[WARN] Initial merge failed. Attempting auto-resolve...")

    # 🔄 Try to resolve by syncing with main
    cmds = [
        ["git", "fetch", "origin"],
        ["git", "merge", "origin/main"],
        ["git", "push"]
    ]

    for cmd in cmds:
        step = subprocess.run(cmd, cwd="repo_clone", capture_output=True, text=True)
        
        if step.returncode != 0:
            return {"success": False, "error": step.stderr.strip() or f"Failed: {' '.join(cmd)}"}

    # 🔁 Retry merge
    retry = subprocess.run(
        ["gh", "pr", "merge", str(pr_number), "--merge"],
        cwd="repo_clone",
        capture_output=True,
        text=True
    )

    if retry.returncode == 0:
        return {"success": True}

    return {
        "success": False,
        "error": retry.stderr
    }