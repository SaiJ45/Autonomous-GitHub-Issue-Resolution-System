import os
try:
    from ..config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH


def index_repo():

    files = []

    for root, dirs, filenames in os.walk(CLONE_PATH):

        for file in filenames:

            # 🔥 INCLUDE HTML FILES (CRITICAL FIX)
            if file.endswith((".py", ".js", ".ts", ".html", ".css", ".md", ".txt")):

                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except:
                    continue

                files.append({
                    "path": path,
                    "content": content
                })

    return files
