import re
import time


def generate_branch_name(issue):
    title = issue["title"].lower()

    # prefix logic
    if "fix" in title or "bug" in title or "error" in title:
        prefix = "fix"
    elif "add" in title or "feature" in title:
        prefix = "feature"
    else:
        prefix = "update"

    # clean title
    title = re.sub(r'[^a-z0-9\s-]', '', title)
    title = re.sub(r'\s+', '-', title).strip("-")
    title = title[:40]

    # 🔥 ADD UNIQUENESS
    timestamp = int(time.time())

    return f"{prefix}/{title}-{timestamp}"