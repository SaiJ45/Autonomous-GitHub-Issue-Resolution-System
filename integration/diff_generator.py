import subprocess

def generate_local_diff(repo_path: str) -> dict:
    try:
        diff_output = subprocess.check_output(
            ["git", "diff", "main...HEAD"],
            cwd=repo_path, text=True, stderr=subprocess.STDOUT
        )
        return {"success": True, "diff": diff_output}
    except subprocess.CalledProcessError:
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "master...HEAD"],
                cwd=repo_path, text=True, stderr=subprocess.STDOUT
            )
            return {"success": True, "diff": diff_output}
        except Exception as e:
            return {"success": False, "error": f"Failed to get diff: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to get diff: {e}"}
