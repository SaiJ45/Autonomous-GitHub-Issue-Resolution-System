import re
import os


# ------------------ LANGUAGE DETECTION ------------------

def detect_language(issue_text):
    issue = issue_text.lower()

    if any(word in issue for word in ["python", "function", "error", "bug", "exception"]):
        return "python"

    if "javascript" in issue or "js" in issue:
        return "js"

    if "html" in issue:
        return "html"

    if "css" in issue:
        return "css"

    return "unknown"


# ------------------ SYMBOL EXTRACTION ------------------

def extract_symbol(issue_text):
    """
    Extract function/class name from issue
    Example:
    - "Fix TypeError in format_user function"
    → format_user
    """

    patterns = [
        r'function\s+(\w+)',
        r'method\s+(\w+)',
        r'class\s+(\w+)',
        r'in\s+(\w+)',  # fallback (e.g., "error in calculate_total")
    ]

    issue = issue_text.lower()

    for pattern in patterns:
        match = re.search(pattern, issue)
        if match:
            return match.group(1)

    return None


# ------------------ FILE SEARCH ------------------

def find_file_by_symbol(symbol, files):
    """
    Find file containing the symbol (function/class name)
    """

    for file in files:
        if is_test_file(file["path"]):
            continue

        content = file["content"]

        if symbol in content:
            return file

    return None


# ------------------ MAIN FILE SELECTION ------------------

def select_relevant_file(issue_text, files):
    """
    🚨 STRICT: returns ONLY ONE file
    """

    lang = detect_language(issue_text)

    # 1. Filter by language
    filtered_files = [
        f for f in files
        if is_relevant_file(f["path"], issue_text) and not is_test_file(f["path"])
    ]

    if not filtered_files:
        filtered_files = files  # fallback

    # 2. Try symbol-based detection (STRONGEST SIGNAL)
    symbol = extract_symbol(issue_text)

    if symbol:
        match = find_file_by_symbol(symbol, filtered_files)
        if match:
            return match

    # 3. Fallback → simple keyword scoring (last resort)
    issue_words = re.findall(r"\w+", issue_text.lower())

    best_file = None
    best_score = -1

    for file in filtered_files:
        content = file["content"].lower()

        score = sum(1 for word in issue_words if word in content)

        if score > best_score:
            best_score = score
            best_file = file

    return best_file


# ------------------ VALIDATION ------------------

def is_relevant_file(file_path, issue_text):
    lang = detect_language(issue_text)

    if lang == "python":
        return file_path.endswith(".py")

    if lang == "js":
        return file_path.endswith(".js")

    if lang == "html":
        return file_path.endswith(".html")

    if lang == "css":
        return file_path.endswith(".css")

    return False


def is_test_file(file_path):
    return "test" in file_path.lower()


def validate_file_target(file_path, issue_text):
    """
    Extra safety check before editing
    """

    if is_test_file(file_path):
        return False  # NEVER edit test files

    return is_relevant_file(file_path, issue_text)