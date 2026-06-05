import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "agents", "ai_issue_agent"))

from config import CLONE_PATH
from git_tools.clone_repo import clone_repo as issue_clone_repo
from git_tools.branch_manager import setup_developer_branch

def setup_repo(issue_id: str) -> dict:
    # Use existing tools
    issue_clone_repo()
    res = setup_developer_branch()
    if not res or not res.get("success"):
        return {"success": False, "error": res.get("error") if res else "Unknown branch error"}
    return {"success": True, "repo_path": CLONE_PATH, "branch": "developer"}
