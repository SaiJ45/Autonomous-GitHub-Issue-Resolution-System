import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "agents", "ai_issue_agent"))

from git_tools.commit_push import commit_and_push

def apply_patch_and_commit(repo_path: str, patched_files: dict, message: str) -> dict:
    # write to disk
    allowed_files = list(patched_files.keys())
    for file_path, content in patched_files.items():
        norm_path = file_path.replace("\\", "/")
        full_path = os.path.normpath(os.path.join(repo_path, norm_path))
        
        # ensure dir exists
        parent_dir = os.path.dirname(full_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    # commit
    res = commit_and_push(message, allowed_files)
    if not res.get("success"):
        return {"success": False, "error": res.get("error")}
    return {"success": True}
