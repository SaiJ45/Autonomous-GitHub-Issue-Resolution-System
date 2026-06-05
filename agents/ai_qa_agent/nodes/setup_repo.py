import os
import subprocess
from state import QAState


def mask_text(text: str) -> str:
    """Masks GitHub tokens in output."""
    if not text:
        return text
    token = os.environ.get("GITHUB_TOKEN")
    if token and token not in ("your_github_token_here", "your_actual_api_key_here", ""):
        if token in text:
            text = text.replace(token, f"{token[:4]}***_MASKED")
    return text


def run_cmd(cmd, cwd=None):
    """Run a shell command, raise RuntimeError on failure."""
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        safe_stdout = mask_text(e.stdout)
        safe_stderr = mask_text(e.stderr)
        safe_cmd = mask_text(" ".join(cmd))
        raise RuntimeError(f"Command '{safe_cmd}' failed.\nstdout: {safe_stdout}\nstderr: {safe_stderr}")


def setup_repo_node(state: QAState) -> dict:
    """
    NODE 1: SETUP_REPO
    Ensures the target repository exists locally at ./<repo_name>.
    If a PR number is provided, checks out the PR branch.
    """
    repo_name = state.get("repo_name", "")
    repo_url = state.get("repo_url", "")
    pr_number = state.get("pr_number")

    if not repo_name or not repo_url:
        return {"status": "FAILED", "error": "Missing repo_name or repo_url in state."}

    # MUST be a relative path — never absolute
    repo_path = f"./{repo_name}"

    # Inject token into clone URL for authenticated access
    token = os.environ.get("GITHUB_TOKEN")
    clone_url = repo_url
    if clone_url.startswith("https://") and token and token not in ("your_github_token_here", "your_actual_api_key_here", ""):
        clone_url = clone_url.replace("https://", f"https://x-access-token:{token}@")

    try:
        if os.path.exists(repo_path):
            print(f"[SETUP_REPO] Repo exists at {repo_path}. Resetting...")
            run_cmd(["git", "reset", "--hard"], cwd=repo_path)
            run_cmd(["git", "clean", "-fd"], cwd=repo_path)

            # Checkout default branch (try main, fallback to master)
            try:
                run_cmd(["git", "checkout", "main"], cwd=repo_path)
                default_branch = "main"
                print("[SETUP_REPO] Checked out 'main'.")
            except RuntimeError:
                try:
                    run_cmd(["git", "checkout", "master"], cwd=repo_path)
                    default_branch = "master"
                    print("[SETUP_REPO] Checked out 'master' (fallback).")
                except RuntimeError:
                    default_branch = "main"  # fallback default
                    print("[SETUP_REPO] WARNING: Could not checkout main or master.")

            # Deterministic sync to avoid unrelated histories merge error
            print(f"[SETUP_REPO] Fetching origin and resetting to origin/{default_branch}...")
            run_cmd(["git", "fetch", "origin"], cwd=repo_path)
            run_cmd(["git", "reset", "--hard", f"origin/{default_branch}"], cwd=repo_path)
            run_cmd(["git", "clean", "-fd"], cwd=repo_path)
            print("[SETUP_REPO] Repo synced with origin successfully.")
            status = "updated"
        else:
            print(f"[SETUP_REPO] Cloning {repo_url} into {repo_path}...")
            run_cmd(["git", "clone", clone_url, repo_path])
            print(f"[SETUP_REPO] Clone complete.")
            status = "cloned"

        # Checkout PR branch if provided
        if pr_number:
            pr_branch = f"pr_{pr_number}"
            print(f"[SETUP_REPO] Fetching PR #{pr_number}...")
            run_cmd(["git", "fetch", "origin", f"pull/{pr_number}/head:{pr_branch}"], cwd=repo_path)
            run_cmd(["git", "checkout", pr_branch], cwd=repo_path)
            print(f"[SETUP_REPO] Checked out PR branch '{pr_branch}'.")

        return {
            "repo_path": repo_path,
            "repo_status": status
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": f"Setup Repo Error: {str(e)}"
        }
