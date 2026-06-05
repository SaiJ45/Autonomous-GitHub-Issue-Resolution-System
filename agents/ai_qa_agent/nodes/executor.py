import os
import time
import subprocess
from datetime import datetime, timezone
from urllib.parse import urlparse
from state import QAState
from tools.github_tools import create_pr, comment_on_pr


def mask_text(text: str) -> str:
    """Masks GitHub tokens in output."""
    if not text:
        return text
    token = os.environ.get("GITHUB_TOKEN")
    if token and token not in ("your_github_token_here", "your_actual_api_key_here", ""):
        # Check if the token is in the text
        if token in text:
            # Mask format: ghp_*** or x-access-token***
            text = text.replace(token, f"{token[:4]}***_MASKED")
    return text


def run_cmd(cmd, cwd=None):
    """Run a shell command, return the result. Raise RuntimeError on failure with masked output."""
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        safe_stdout = mask_text(e.stdout)
        safe_stderr = mask_text(e.stderr)
        safe_cmd = mask_text(" ".join(cmd))
        raise RuntimeError(f"Command '{safe_cmd}' failed.\nstdout: {safe_stdout}\nstderr: {safe_stderr}")


def verify_branch_exists_remote(repo_path: str, branch_name: str) -> bool:
    """Verify a branch exists on the remote using git ls-remote."""
    try:
        result = run_cmd(["git", "ls-remote", "--heads", "origin", branch_name], cwd=repo_path)
        return branch_name in result.stdout
    except Exception:
        return False


def has_staged_changes(repo_path: str) -> bool:
    """Check if there are any changes to commit (staged only)."""
    try:
        # git diff --cached --quiet exits with 1 if there are changes, 0 if no changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_path, capture_output=True, text=True
        )
        return result.returncode == 1
    except Exception:
        return False


def parse_repo_from_remote_url(remote_url: str) -> str | None:
    """Extract owner/repo from a git remote URL."""
    if not remote_url:
        return None
    cleaned = remote_url.strip()
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    # git@host:owner/repo
    if "://" not in cleaned and ":" in cleaned:
        path = cleaned.split(":", 1)[1].strip("/")
    else:
        # https://host/owner/repo (and similar URL forms)
        path = urlparse(cleaned).path.strip("/")

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[-2]}/{parts[-1]}"


def get_repo_from_origin(repo_path: str) -> tuple[str | None, str | None]:
    """Read origin remote URL and derive owner/repo."""
    try:
        remote_v = run_cmd(["git", "remote", "-v"], cwd=repo_path)
        if "origin" not in remote_v.stdout:
            return None, "No git remote 'origin' configured."

        remote_url = run_cmd(["git", "remote", "get-url", "origin"], cwd=repo_path).stdout.strip()
        repo_name = parse_repo_from_remote_url(remote_url)
        if not repo_name:
            return None, f"Unable to derive owner/repo from origin URL: {mask_text(remote_url)}"
        return repo_name, None
    except RuntimeError as e:
        return None, f"Failed to read git origin remote: {mask_text(str(e))}"


def verify_origin_remote(repo_path: str) -> tuple[bool, str | None]:
    """Validate origin exists and is reachable before push."""
    try:
        remote_v = run_cmd(["git", "remote", "-v"], cwd=repo_path)
        if "origin" not in remote_v.stdout:
            return False, "No git remote 'origin' configured."

        run_cmd(["git", "ls-remote", "origin"], cwd=repo_path)
        return True, None
    except RuntimeError as e:
        return False, f"Remote repository not accessible: {mask_text(str(e))}"


def executor_node(state: QAState) -> dict:
    """
    NODE 5: EXECUTOR (REAL ACTIONS ONLY)
    Executes git operations, creates PRs, posts comments.
    Every action is validated — no narration without proof.

    Tracks execution_status separately from qa_decision.
    execution_status = "SUCCESS" only if ALL steps complete successfully.
    """
    if state.get("status") == "FAILED":
        return {}

    repo_path = state.get("repo_path")
    decision = state.get("decision")
    reason = state.get("reason", "No reason provided.")
    pr_number = state.get("pr_number")

    execution_status = "SUCCESS"  # Will be set to FAILED on any step failure
    action_log = f"- Repo: {state.get('repo_status', 'unknown')}\n"

    if not decision:
        return {
            "action_log": action_log,
            "status": "FAILED",
            "error": "No decision found to execute.",
            "execution_status": "FAILED"
        }

    try:
        if decision in ["REJECTED", "REVIEW_REQUIRED"]:
            repo_name, repo_err = get_repo_from_origin(repo_path)
            if not repo_name:
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "FAILED",
                    "error": repo_err or "Unable to resolve repository from git origin."
                }

            # ── Step 1: Create branch ──
            timestamp = int(time.time())
            # Replace rejection with review if REVIEW_REQUIRED
            branch_prefix = "rejection" if decision == "REJECTED" else "review"
            branch_name = f"qa/{branch_prefix}-{timestamp}"
            print(f"[EXECUTOR] Creating branch: {branch_name}")
            try:
                run_cmd(["git", "checkout", "-b", branch_name], cwd=repo_path)
            except RuntimeError as e:
                action_log += f"- Branch creation: FAILURE ({mask_text(str(e))[:100]})\n"
                print(f"[EXECUTOR] STOP: Branch creation failed: {mask_text(str(e))}")
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "FAILED",
                    "error": f"Executor Error: Branch creation failed: {mask_text(str(e))[:200]}"
                }

            # ── Step 2: Create QA report file ──
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            test_cases = "\n".join([f"- {tc}" for tc in state.get("test_cases", [])]) or "None"
            tests_passed = "\n".join([f"- {tp}" for tp in state.get("tests_passed", [])]) or "None"
            tests_failed = "\n".join([f"- {tf}" for tf in state.get("tests_failed", [])]) or "None"
            suggestions = "\n".join([f"- {s}" for s in state.get("suggestions", [])]) or "None"

            report_title = "QA Rejection Report" if decision == "REJECTED" else "QA Review/Fixes Required"

            report_content = (
                f"# {report_title}\n\n"
                f"**Generated**: {now}\n"
                f"**PR**: #{pr_number}\n"
                f"**Decision**: {decision}\n\n"
                f"## 🧠 Reason:\n{reason}\n\n"
                f"## 🧪 Tests Generated:\n{test_cases}\n\n"
                f"## ✅ Tests Passed:\n{tests_passed}\n\n"
                f"## ❌ Tests Failed:\n{tests_failed}\n\n"
                f"## 🔍 Execution Errors:\n{state.get('execution_errors', 'None')}\n\n"
                f"## 🔍 Test Execution Reasoning:\n{state.get('test_execution_reasoning', 'None')}\n\n"
                f"## 💡 Suggestions:\n{suggestions}\n\n"
                f"## 📊 Confidence:\n{state.get('confidence', 'MEDIUM')}\n\n"
                f"## 📝 Final Review:\n{state.get('final_comment', 'None')}\n"
            )

            report_path = os.path.join(repo_path, "qa_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print("[EXECUTOR] Created qa_report.md")

            # ── Step 3: Stage and check for changes ──
            run_cmd(["git", "add", "qa_report.md"], cwd=repo_path)

            if not has_staged_changes(repo_path):
                action_log += "- Commit: SKIPPED (no changes detected — empty commit guard)\n"
                action_log += "- PR: SKIPPED (nothing to commit)\n"
                print("[EXECUTOR] WARNING: No changes to commit. Skipping PR creation.")
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "COMPLETED",
                    "error": "No changes to commit"
                }

            # ── Step 4: Commit ──
            commit_msg = f"QA Rejection: {reason[:50]}"
            try:
                run_cmd(["git", "commit", "-m", commit_msg], cwd=repo_path)
                print(f"[EXECUTOR] Committed: {commit_msg}")
            except RuntimeError as e:
                action_log += f"- Commit: FAILURE ({mask_text(str(e))[:100]})\n"
                print(f"[EXECUTOR] STOP: Commit failed: {mask_text(str(e))}")
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "FAILED",
                    "error": f"Executor Error: Commit failed: {mask_text(str(e))[:200]}"
                }

            # ── Step 5: Verify Remote ──
            print(f"[EXECUTOR] Verifying remote origin...")
            is_remote_ok, remote_err = verify_origin_remote(repo_path)
            if is_remote_ok:
                action_log += "- Remote: verified\n"
            else:
                action_log += "- Remote: failed\n"
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "FAILED",
                    "error": remote_err or "No git remote configured or accessible"
                }

            # ── Step 6: Push to Origin ──
            print(f"[EXECUTOR] Pushing branch {branch_name} to origin...")
            try:
                run_cmd(["git", "push", "origin", branch_name], cwd=repo_path)
                print("[EXECUTOR] Push succeeded.")
                action_log += "- Push: success\n"
            except RuntimeError as e:
                err_str = mask_text(str(e))
                
                # Check for authentication errors
                auth_errors = [
                    "invalid username or password",
                    "authentication failed",
                    "password authentication",
                    "could not read username",
                    "could not read password"
                ]
                is_auth_error = any(ae in err_str.lower() for ae in auth_errors)
                
                if is_auth_error:
                    print("[EXECUTOR] Git authentication failure detected. Attempting to use PAT...")
                    token = os.environ.get("GITHUB_TOKEN")
                    if not token or token in ("your_github_token_here", "your_actual_api_key_here", ""):
                        err_msg = "Missing GITHUB_TOKEN environment variable"
                        print(f"[EXECUTOR] STOP: {err_msg}")
                        return {
                            "action_log": action_log + "- Push: failed (missing token)\n",
                            "execution_status": "FAILED",
                            "status": "FAILED",
                            "error": err_msg
                        }
                    
                    # Update remote with authenticated URL
                    auth_url = f"https://{token}@github.com/{repo_name}.git"
                    try:
                        print("[EXECUTOR] Updating remote URL to use PAT safely...")
                        run_cmd(["git", "remote", "set-url", "origin", auth_url], cwd=repo_path)
                        
                        print(f"[EXECUTOR] Retrying push for {branch_name}...")
                        run_cmd(["git", "push", "origin", branch_name], cwd=repo_path)
                        print("[EXECUTOR] Push succeeded on retry.")
                        action_log += "- Push: success (after auth retry)\n"
                    except RuntimeError as retry_e:
                        retry_err_str = mask_text(str(retry_e))
                        action_log += "- Push: failed after auth retry\n"
                        print(f"[EXECUTOR] STOP: Push FAILED on retry: {retry_err_str[:200]}")
                        return {
                            "action_log": action_log,
                            "execution_status": "FAILED",
                            "status": "FAILED",
                            "error": f"Push failed after auth retry: {retry_err_str[:200]}"
                        }
                else:
                    action_log += "- Push: failed\n"
                    print(f"[EXECUTOR] STOP: Push FAILED: {err_str[:200]}")
                    return {
                        "action_log": action_log,
                        "execution_status": "FAILED",
                        "status": "FAILED",
                        "error": f"Push failed: {err_str[:200]}"
                    }

            # ── Step 7: Verify branch exists remotely ──
            print(f"[EXECUTOR] Verifying branch {branch_name} exists on remote...")
            if verify_branch_exists_remote(repo_path, branch_name):
                action_log += "- Branch: verified\n"
                print(f"[EXECUTOR] Branch verified on remote.")
            else:
                action_log += "- Branch: missing\n"
                print(f"[EXECUTOR] STOP: Branch NOT found on remote after push.")
                return {
                    "action_log": action_log,
                    "execution_status": "FAILED",
                    "status": "FAILED",
                    "error": f"Push failed: branch not found on remote"
                }

            # ── Step 8: Determine base branch for PR ──
            base_branch = "main"
            try:
                result_check = run_cmd(["git", "ls-remote", "--heads", "origin", "main"], cwd=repo_path)
                if "main" not in result_check.stdout:
                    result_master = run_cmd(["git", "ls-remote", "--heads", "origin", "master"], cwd=repo_path)
                    if "master" in result_master.stdout:
                        base_branch = "master"
                    else:
                        action_log += "- PR: FAILURE (no valid base branch found: neither 'main' nor 'master')\n"
                        print("[EXECUTOR] STOP: No valid base branch found.")
                        return {
                            "action_log": action_log,
                            "execution_status": "FAILED",
                            "status": "FAILED",
                            "error": "No valid base branch found (neither 'main' nor 'master')."
                        }
            except Exception:
                pass  # Default to 'main'

            # ── Step 9: Create PR via API (with retry) ──
            print(f"[EXECUTOR] Creating PR via GitHub API (base: {base_branch})...")
            pr_title = state.get("pr_title", "").strip()
            if not pr_title or "QA Review: ..." in pr_title or "Incomplete Analysis" in pr_title:
                short_desc = reason.split('\n')[0][:40].strip()
                if pr_number:
                    pr_title = f"QA Review: #{pr_number} - {short_desc} -> {decision}"
                else:
                    pr_title = f"QA Review: {short_desc} -> {decision}"

            pr_res = create_pr(
                repo=repo_name,
                title=pr_title,
                body=report_content,
                head=branch_name,
                base=base_branch
            )

            if pr_res.get("status_code") == 201 and pr_res.get("pr_url"):
                pr_url = pr_res["pr_url"]
                action_log += f"- PR: SUCCESS ({pr_url})\n"
                print(f"[EXECUTOR] PR created: {pr_url}")
            else:
                api_error = pr_res.get("error", "Unknown API error")
                action_log += f"- PR: FAILURE (API status {pr_res.get('status_code')}: {mask_text(str(api_error))[:200]})\n"
                print(f"[EXECUTOR] PR creation FAILED: {mask_text(str(api_error))[:200]}")
                execution_status = "FAILED"

            # ── Step 10: Comment on original PR ──
            if pr_number:
                test_cases = "\n".join([f"- {tc}" for tc in state.get("test_cases", [])]) or "None"
                tests_passed = "\n".join([f"- {tp}" for tp in state.get("tests_passed", [])]) or "None"
                tests_failed = "\n".join([f"- {tf}" for tf in state.get("tests_failed", [])]) or "None"
                suggestions = "\n".join([f"- {s}" for s in state.get("suggestions", [])]) or "None"

                comment_body = (
                    f"## {'❌' if decision == 'REJECTED' else '⚠️'} Decision: {decision}\n\n"
                    f"## 🧠 Reason:\n{reason}\n\n"
                    f"## 🧪 Tests Generated:\n{test_cases}\n\n"
                    f"## ✅ Tests Passed:\n{tests_passed}\n\n"
                    f"## ❌ Tests Failed:\n{tests_failed}\n\n"
                    f"## 🔍 Execution Errors:\n{state.get('execution_errors', 'None')}\n\n"
                    f"## 🔍 Test Execution Reasoning:\n{state.get('test_execution_reasoning', 'None')}\n\n"
                    f"## 💡 Suggestions:\n{suggestions}\n\n"
                    f"## 📊 Confidence:\n{state.get('confidence', 'MEDIUM')}\n\n"
                    f"## 📝 Final Review:\n{state.get('final_comment', 'None')}\n\n"
                    f"---\n"
                    f"*Automated QA Pipeline — {now}*"
                )
                comment_res = comment_on_pr(repo=repo_name, pr_number=pr_number, body=comment_body)
                if comment_res.get("status_code") == 201:
                    comment_url = comment_res.get("comment_url", "N/A")
                    action_log += f"- PR Comment: SUCCESS ({comment_url})\n"
                    print(f"[EXECUTOR] Rejection comment posted on PR #{pr_number}.")
                else:
                    action_log += f"- PR Comment: FAILURE ({mask_text(str(comment_res.get('error', 'Unknown'))[:200])})\n"
                    print(f"[EXECUTOR] Failed to post rejection comment: {mask_text(str(comment_res.get('error'))[:200])}")
                    execution_status = "FAILED"

        elif decision == "APPROVED":
            print("[EXECUTOR] Decision: APPROVED. Posting approval comment on PR...")

            if pr_number:
                repo_name, repo_err = get_repo_from_origin(repo_path)
                if not repo_name:
                    return {
                        "action_log": action_log,
                        "execution_status": "FAILED",
                        "status": "FAILED",
                        "error": repo_err or "Unable to resolve repository from git origin."
                    }
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                test_cases = "\n".join([f"- {tc}" for tc in state.get("test_cases", [])]) or "None"
                tests_passed = "\n".join([f"- {tp}" for tp in state.get("tests_passed", [])]) or "None"
                tests_failed = "\n".join([f"- {tf}" for tf in state.get("tests_failed", [])]) or "None"
                suggestions = "\n".join([f"- {s}" for s in state.get("suggestions", [])]) or "None"

                comment_body = (
                    f"## ✅ Decision: APPROVED\n\n"
                    f"## 🧠 Reason:\n{reason}\n\n"
                    f"## 🧪 Tests Generated:\n{test_cases}\n\n"
                    f"## ✅ Tests Passed:\n{tests_passed}\n\n"
                    f"## ❌ Tests Failed:\n{tests_failed}\n\n"
                    f"## 🔍 Execution Errors:\n{state.get('execution_errors', 'None')}\n\n"
                    f"## 🔍 Test Execution Reasoning:\n{state.get('test_execution_reasoning', 'None')}\n\n"
                    f"## 💡 Suggestions:\n{suggestions}\n\n"
                    f"## 📊 Confidence:\n{state.get('confidence', 'MEDIUM')}\n\n"
                    f"## 📝 Final Review:\n{state.get('final_comment', 'None')}\n\n"
                    f"---\n"
                    f"*Automated QA Pipeline — {now}*"
                )
                comment_res = comment_on_pr(repo=repo_name, pr_number=pr_number, body=comment_body)
                if comment_res.get("status_code") == 201:
                    comment_url = comment_res.get("comment_url", "N/A")
                    action_log += f"- PR Comment: SUCCESS ({comment_url})\n"
                    print(f"[EXECUTOR] Approval comment posted on PR #{pr_number}.")
                else:
                    action_log += f"- PR Comment: FAILURE ({mask_text(str(comment_res.get('error', 'Unknown'))[:200])})\n"
                    print(f"[EXECUTOR] Failed to post approval comment: {mask_text(str(comment_res.get('error'))[:200])}")
                    execution_status = "FAILED"
            else:
                action_log += "- PR Comment: SKIPPED (no PR number available)\n"
                print("[EXECUTOR] No PR number — skipping approval comment.")

        return {
            "action_log": action_log,
            "execution_status": execution_status,
            "status": "COMPLETED"
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": f"Executor Error: {mask_text(str(e))}",
            "execution_status": "FAILED"
        }
