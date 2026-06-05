"""
main.py

Entry point for the AI issue solver.

Pre-graph:   Select issue, clone repo, create branch
Graph:       planner -> retrieval -> coder -> feedback (with retry loop)
Post-graph:  Apply patches, commit, create PR, merge approval
"""

from agents.issue_reader import get_issues
from agents.patch_generator import PatchGenerator

from git_tools.clone_repo import clone_repo
from git_tools.branch_manager import create_branch
from git_tools.commit_push import commit_and_push
from git_tools.create_pr import create_pull_request, generate_pr_content
from git_tools.merge_pr import merge_pull_request


from utils.create_branch import generate_branch_name

from graph import build_graph

import os
try:
    from .config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH


# ======================================================================
# UTILITIES
# ======================================================================

def ask_merge_approval() -> bool:
    """Require explicit yes/no before deciding PR merge."""
    while True:
        try:
            user_choice = input("Merge this PR now? [y/n]: ").strip().lower()
        except EOFError:
            print("[WARN] No input received; merge skipped.")
            return False

        if user_choice in {"y", "yes"}:
            return True
        if user_choice in {"n", "no"}:
            return False

        print("Please respond with 'y' or 'n'.")


def process_patched_files(
    patched_files: dict[str, str],
    allowed_files: list[str],
) -> bool:
    """
    Write all patched files to disk atomically.
    ALL files must pass validation before any are written.
    Returns True if all files were written successfully.

    Raises:
        TypeError: If patched_files is not a dict or allowed_files is not a list.
        ValueError: If any value in patched_files is not a string.
    """
    # --- Input validation ---
    if not isinstance(patched_files, dict):
        raise TypeError(
            f"patched_files must be a dict, got {type(patched_files).__name__}"
        )
    if not isinstance(allowed_files, list):
        raise TypeError(
            f"allowed_files must be a list, got {type(allowed_files).__name__}"
        )
    for fpath, content in patched_files.items():
        if not isinstance(fpath, str) or not isinstance(content, str):
            raise ValueError(
                f"patched_files keys and values must be strings, "
                f"got key={type(fpath).__name__}, value={type(content).__name__}"
            )

    # --- Edge case: empty ---
    if not patched_files:
        print("[ERROR] No patched files to write")
        return False

    if not allowed_files:
        print("[ERROR] allowed_files is empty -- nothing is permitted")
        return False

    # Normalize allowed files for comparison
    allowed_normalized = set()
    for f in allowed_files:
        if not isinstance(f, str):
            continue
        norm = f.replace("\\", "/").strip("/")
        if norm:
            allowed_normalized.add(norm)

    if not allowed_normalized:
        print("[ERROR] No valid entries in allowed_files after normalization")
        return False

    # Validate all paths before writing any
    for file_path in patched_files:
        norm_path = file_path.replace("\\", "/")
        full_path = os.path.normpath(os.path.join(CLONE_PATH, norm_path))
        parent_dir = os.path.dirname(full_path)
        if not os.path.exists(parent_dir):
            print(f"[ERROR] Directory does not exist for: {file_path}")
            return False

    # Check all files are in allowed list
    for file_path in patched_files:
        norm = file_path.replace("\\", "/").strip("/")
        if norm not in allowed_normalized:
            print(f"[BLOCKED] {file_path} not in allowed files: {allowed_normalized}")
            return False

    # All checks passed -- write files
    for file_path, content in patched_files.items():
        norm_path = file_path.replace("\\", "/")
        full_path = os.path.normpath(os.path.join(CLONE_PATH, norm_path))

        print(f"[WRITE] {full_path}")
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   [OK] Updated: {file_path}")
        except OSError as e:
            print(f"   [ERROR] Failed to write {file_path}: {e}")
            return False

    return True


def select_issue() -> dict | None:
    """Interactive issue selection. Returns the selected issue dict or None."""
    issues = get_issues()
    if not issues:
        print("\nNo open GitHub issues were returned for this repository.")
        return None

    print("\nAvailable Issues:\n")
    for issue in issues:
        print(f"#{issue['number']}: {issue['title']}")

    issue_numbers = {issue["number"] for issue in issues}
    while True:
        try:
            raw_issue_number = input("\nWhich issue to solve: ").strip()
        except EOFError:
            print("\nNo issue number was provided.")
            return None

        if not raw_issue_number:
            print("Please enter an issue number from the list above.")
            continue

        if not raw_issue_number.isdigit():
            print("Issue number must be numeric.")
            continue

        issue_number = int(raw_issue_number)
        if issue_number not in issue_numbers:
            print("That issue number is not in the current open-issues list.")
            continue

        break

    return [i for i in issues if i["number"] == issue_number][0]


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_agent():
    # -- PRE-GRAPH: Issue selection ----------------------------------------
    issue = select_issue()
    if issue is None:
        return

    issue_number = issue["number"]
    issue_body = issue.get("body", "") or ""
    issue_text = issue["title"] + " " + issue_body

    print(f"\n{'='*60}")
    print(f"SCOPE CONSTRAINT: Solving ISSUE #{issue_number}: {issue['title']}")
    print(f"{'='*60}")

    # -- PRE-GRAPH: Clone & branch -----------------------------------------
    clone_repo()

    branch_name = generate_branch_name(issue)
    branch_result = create_branch(branch_name)
    if not branch_result or not branch_result.get("success"):
        error_msg = branch_result.get("error") if branch_result else "Unknown error"
        print(f"[ERROR] Cannot continue without a feature branch: {error_msg}")
        return

    # -- GRAPH EXECUTION ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting LangGraph pipeline")
    print(f"{'='*60}")

    graph = build_graph()

    initial_state = {
        "issue": issue_text,
        "issue_meta": {
            "number": issue_number,
            "title": issue["title"],
            "body": issue_body,
        },
        "repo_path": CLONE_PATH,
        "plan": {},
        "context": "",
        "original_files": {},
        "code_diffs": {},
        "patched_files": {},
        "errors": "",
        "previous_diffs": {},
        "retries": 0,
        "status": "running",
        "edge_cases": [],
    }

    result = graph.invoke(initial_state)

    # -- POST-GRAPH: Handle results ----------------------------------------
    status = result.get("status", "failure")
    patched_files = result.get("patched_files", {})
    plan = result.get("plan", {})
    allowed_files = plan.get("files_to_modify", [])

    if status == "planner_failed":
        print("\n[ERROR] Pipeline aborted -- planner could not produce a valid plan")
        return

    if status != "success" or not patched_files:
        print(f"\n[ERROR] Pipeline finished with status: {status}")
        print("   No valid solution was produced after all retries.")
        if result.get("errors"):
            print(f"   Last errors: {result['errors'][:300]}")
        return

    # -- POST-GRAPH: Apply patches to disk ---------------------------------
    print(f"\n{'='*60}")
    print(f"Applying {len(patched_files)} patched file(s) to disk")
    print(f"{'='*60}")

    write_ok = process_patched_files(patched_files, allowed_files)
    if not write_ok:
        print("[BLOCKED] Failed to write patched files -- aborting")
        return

    # -- POST-GRAPH: Commit & PR -------------------------------------------
    print(f"\n{'='*60}")
    print(f"Creating commit and pull request")
    print(f"{'='*60}")

    commit_result = commit_and_push(
        f"Fix #{issue_number}: {issue['title'][:60]}",
        allowed_files=allowed_files,
    )

    if not commit_result.get("success"):
        print(f"[BLOCKED] Commit aborted: {commit_result.get('error')}")
        return

    # Generate PR content
    patch_gen = PatchGenerator()
    title, description = generate_pr_content(
        issue, list(patched_files.keys()), patch_gen,
    )

    pr = create_pull_request(title, description)

    if not pr.get("success"):
        print(f"[ERROR] PR creation failed: {pr.get('error')}")
        return

    pr_number = pr["data"]["pr_number"]
    print(f"\nPR created: #{pr_number}")

    # -- POST-GRAPH: Merge approval ----------------------------------------
    approved = ask_merge_approval()

    if approved:
        merge_result = merge_pull_request(pr_number)
        if merge_result.get("success"):
            print(f"[OK] PR #{pr_number} merged successfully")
        else:
            print(f"[ERROR] PR #{pr_number} merge failed: {merge_result.get('error')}")
    else:
        print("Merge skipped by user")


if __name__ == "__main__":
    run_agent()
