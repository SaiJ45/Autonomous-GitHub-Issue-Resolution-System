import subprocess
import os
try:
    from ..config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH


def ensure_env_files():
    init_path = os.path.join(CLONE_PATH, "app", "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()



def reset_repo():
    try:
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=CLONE_PATH,
            check=True
        )

        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=CLONE_PATH,
            check=True
        )

        ensure_env_files()  # 🔥 CRITICAL FIX

        print("Repository reset to clean state")

    except subprocess.CalledProcessError:
        print("Failed to reset repository")
