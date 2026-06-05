import re


def detect_intent(issue_text: str) -> str:
    issue = issue_text.lower()

    if "h2" in issue:
        return "H2_CONTENT"

    if "duplicate" in issue:
        return "DUPLICATE"

    if "syntax" in issue or "error" in issue:
        return "SYNTAX"

    return "GENERAL"


# 🔥 NEW: SCOPE VALIDATION
def validate_scope(original: str, updated: str) -> bool:
    if len(updated) > len(original) * 1.5:
        print("❌ Too much content added (scope violation)")
        return False
    return True


def validate_h2_content(original: str, content: str) -> bool:

    original_count = len(re.findall(r"<h2>", original, re.IGNORECASE))
    new_count = len(re.findall(r"<h2>", content, re.IGNORECASE))

    if new_count > original_count:
        print("❌ Extra <h2> tags added")
        return False

    matches = re.findall(r"<h2>(.*?)</h2>", content, re.IGNORECASE)

    if not matches:
        print("❌ No <h2> tag found")
        return False

    for m in matches:
        if not m.strip():
            print("❌ Empty <h2>")
            return False

    print("✅ H2 validation passed")
    return True


def validate_task(file_path: str, content: str, issue_text: str, original_code: str) -> bool:

    intent = detect_intent(issue_text)

    print(f"🔍 Intent: {intent}")

    # ✅ BASIC SANITY CHECKS
    if not content or len(content.strip()) < 5:
        print("❌ Content too small")
        return False

    if content.count("{") != content.count("}"):
        print("❌ Unbalanced braces")
        return False

    if not validate_scope(original_code, content):
        return False

    if intent == "H2_CONTENT":
        return validate_h2_content(original_code, content)

    return True