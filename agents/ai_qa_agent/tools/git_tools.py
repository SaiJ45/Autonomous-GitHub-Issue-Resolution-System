import subprocess
from typing import List

def get_git_diff(repo_path: str, base_branch: str, feature_branch: str) -> str:
    """Returns the raw unified diff between the base branch and feature branch."""
    try:
        merge_base_cmd = ["git", "merge-base", base_branch, feature_branch]
        base_commit = subprocess.check_output(merge_base_cmd, cwd=repo_path, text=True).strip()
        
        diff_cmd = ["git", "diff", base_commit, feature_branch]
        diff_output = subprocess.check_output(diff_cmd, cwd=repo_path, text=True)
        return diff_output
    except subprocess.CalledProcessError as e:
        print(f"Git diff error: {e}")
        return ""

def get_modified_files(repo_path: str, base_branch: str, feature_branch: str) -> List[str]:
    """Returns a list of files modified in the PR."""
    try:
        merge_base_cmd = ["git", "merge-base", base_branch, feature_branch]
        base_commit = subprocess.check_output(merge_base_cmd, cwd=repo_path, text=True).strip()
        
        diff_cmd = ["git", "diff", "--name-only", base_commit, feature_branch]
        output = subprocess.check_output(diff_cmd, cwd=repo_path, text=True).strip()
        if not output:
            return []
        return [f for f in output.split('\n') if f]
    except subprocess.CalledProcessError as e:
        print(f"Git diff error: {e}")
        return []
