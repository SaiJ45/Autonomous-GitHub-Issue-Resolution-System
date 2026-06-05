import subprocess
import time
import re
try:
    from ..config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH


PROTECTED_BRANCHES = {"main", "developer", "master"}

def setup_developer_branch():
    branch_name = "developer"
    try:
        # 1. Checkout main & 2. Fetch/Reset
        main_branch = "main"
        try:
            subprocess.run(["git", "checkout", main_branch], cwd=CLONE_PATH, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            main_branch = "master"
            subprocess.run(["git", "checkout", main_branch], cwd=CLONE_PATH, check=True, capture_output=True)
            
        subprocess.run(["git", "fetch", "origin"], cwd=CLONE_PATH, check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{main_branch}"], cwd=CLONE_PATH, check=True)
        
        # 3. Clean stale branches
        branches_output = subprocess.check_output(
            ["git", "branch"], cwd=CLONE_PATH, text=True
        )
        local_branches = [b.strip().lstrip("* ") for b in branches_output.splitlines()]
        
        deleted = []
        for branch in local_branches:
            if branch not in PROTECTED_BRANCHES:
                subprocess.run(["git", "branch", "-D", branch], cwd=CLONE_PATH, capture_output=True)
                print(f"[CLEAN] Deleted stale branch: {branch}")
                deleted.append(branch)
        
        if deleted:
            print(f"[CLEAN] Deleted {len(deleted)} stale branch(es)")
        else:
            print("[CLEAN] No stale branches to delete")

        # 4. developer branch lifecycle
        if "developer" in local_branches:
            subprocess.run(["git", "checkout", "developer"], cwd=CLONE_PATH, check=True, capture_output=True)
            subprocess.run(["git", "rebase", f"origin/{main_branch}"], cwd=CLONE_PATH, check=True, capture_output=True)
            print("[OK] Rebased developer on origin/main")
        else:
            subprocess.run(["git", "checkout", "-b", "developer", f"origin/{main_branch}"], cwd=CLONE_PATH, check=True, capture_output=True)
            print("[SETUP] Created developer branch from origin/main")
            
        print(f"[OK] Switched to branch: {branch_name}")
        return {"success": True, "branch": branch_name}

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Branch setup failed: {e}")
        return {"success": False, "error": str(e)}
