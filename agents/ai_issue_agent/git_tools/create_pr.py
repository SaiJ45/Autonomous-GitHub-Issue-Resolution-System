import os
import re
import subprocess
from dotenv import load_dotenv

load_dotenv()


def create_pull_request(title: str, body: str, head: str = None, base: str = "main") -> dict:
    """
    Create a GitHub Pull Request using the PyGitHub API.
    
    Returns:
        {
            "success": True,
            "data": {
                "pr_number": str,
                "url": str,
                "branch": str
            }
        }
    or:
        {"success": False, "error": "<reason>"}
    """
    token     = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPO")

    if not token:
        return {"success": False, "error": "GITHUB_TOKEN not set in environment"}
    if not repo_name:
        return {"success": False, "error": "GITHUB_REPO not set in environment"}

    # Determine the head branch if not provided
    if not head:
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd="repo_clone", capture_output=True, text=True
            )
            head = result.stdout.strip()
        except Exception as e:
            return {"success": False, "error": f"Could not determine current branch: {e}"}

    if not head:
        return {"success": False, "error": "Could not determine head branch"}
    if head in {"main", "master"}:
        return {"success": False, "error": f"Refusing to create PR from protected branch: {head}"}

    try:
        from github import Github, Auth, GithubException

        auth = Auth.Token(token)
        g    = Github(auth=auth)
        repo = g.get_repo(repo_name)

        # ── Check if a PR already exists for this branch (REMOVED: Rule 1 enforces unique PR creation) ──

        # ── Create new PR ──
        pr = repo.create_pull(title=title, body=body, head=head, base=base)

        print(f"[PR CREATED] #{pr.number}")
        print(f"[PR URL] {pr.html_url}")

        return {
            "success": True,
            "data": {
                "pr_number": str(pr.number),
                "url":        pr.html_url,
                "branch":     head
            }
        }

    except Exception as e:
        error_msg = str(e)
        print(f"[PR CREATION FAILED] {error_msg}")
        return {"success": False, "error": error_msg}


def generate_pr_content(issue, modified_files, patch_generator):
    issue_title = issue["title"]
    issue_body = issue.get("body") or ""

    clean_files = [
        f.replace("./repo_clone/", "").replace("./repo_clone\\", "")
        for f in modified_files
    ]

    files_list = "\n".join([f"- {f}" for f in clean_files]) or "- None"

    prompt = f"""
Write a clean, professional GitHub PR.

ISSUE:
{issue_title}

DETAILS:
{issue_body}

FILES:
{files_list}

RULES:
- Keep it SHORT (max 6–8 lines total)
- No fluff
- No unnecessary explanations

FORMAT:

<Short Title>

Problem:
<one line>

Fix:
<one line>

Files:
{files_list}

Result:
<one line>
"""

    response = patch_generator.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    text = response.choices[0].message.content.strip()

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    title = lines[0] if lines else "Fix issue"

    description = "\n".join(lines[1:]).strip()

    if not description:
        description = f"""Problem:
Issue identified.

Fix:
Applied correction.

Files:
{files_list}

Result:
Resolved successfully.
"""

    return title, description