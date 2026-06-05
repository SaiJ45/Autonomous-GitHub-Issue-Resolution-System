import os
import re


_IDENTIFIER_RE = r"[A-Za-z_][A-Za-z0-9_]*"
_FILE_EXTENSIONS = (
    "py", "js", "jsx", "ts", "tsx", "java", "go", "rs", "rb", "cpp", "c", "cs",
)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto", "over",
    "under", "issue", "bug", "fix", "code", "file", "class", "function", "method",
    "module", "should", "would", "could", "there", "their", "about", "after",
    "before", "while", "where", "when", "then", "than", "because", "return",
    "returns", "handle", "handles", "handling", "using", "used", "use", "make",
    "makes", "change", "changes", "changing", "ensure", "ensures", "correct",
    "incorrect", "wrong", "right", "need", "needs", "must", "into", "value",
    "values", "input", "output", "error", "errors", "exception", "exceptions",
}


def _unique_sorted(items) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            ordered.append(cleaned)
    return ordered


def normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return ""
    return symbol.strip().strip("`").replace("::", ".")


def symbol_matches(symbol: str, available_symbols: set[str]) -> bool:
    normalized = normalize_symbol(symbol).lower()
    if not normalized:
        return False
    normalized_set = {normalize_symbol(item).lower() for item in available_symbols if isinstance(item, str)}
    if normalized in normalized_set:
        return True
    bare = normalized.split(".")[-1]
    if bare in normalized_set:
        return True
    return any(item.endswith(f".{bare}") for item in normalized_set)


def extract_issue_entities(issue_text: str) -> dict:
    text = issue_text if isinstance(issue_text, str) else ""
    lower = text.lower()

    functions = []
    classes = []
    modules = []
    file_paths = []

    for match in re.findall(rf"(?:function|method|def)\s+({_IDENTIFIER_RE})", text, re.IGNORECASE):
        functions.append(match)
    for match in re.findall(rf"\b({_IDENTIFIER_RE})\s*\(", text):
        functions.append(match)
    for match in re.findall(rf"(?:class)\s+({_IDENTIFIER_RE})", text, re.IGNORECASE):
        classes.append(match)
    for match in re.findall(r"\b([A-Z][A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?)\b", text):
        if "." in match:
            classes.append(match)
        elif re.search(r"[A-Z].*[A-Z]", match):
            classes.append(match)

    file_pattern = r"\b([A-Za-z0-9_./\\-]+\.(?:" + "|".join(_FILE_EXTENSIONS) + r"))\b"
    for match in re.findall(file_pattern, text, re.IGNORECASE):
        normalized = match.replace("\\", "/").strip("/")
        file_paths.append(normalized)
        modules.append(os.path.splitext(os.path.basename(normalized))[0])

    for quoted in re.findall(r"`([^`]+)`", text):
        cleaned = quoted.strip()
        if not cleaned:
            continue
        if re.fullmatch(file_pattern, cleaned, re.IGNORECASE):
            normalized = cleaned.replace("\\", "/").strip("/")
            file_paths.append(normalized)
            modules.append(os.path.splitext(os.path.basename(normalized))[0])
            continue
        if cleaned.endswith("()"):
            functions.append(cleaned[:-2])
            continue
        if "." in cleaned and "/" not in cleaned and "\\" not in cleaned:
            modules.append(cleaned)
            parts = cleaned.split(".")
            if parts[-1]:
                functions.append(parts[-1])
            continue
        if re.fullmatch(_IDENTIFIER_RE, cleaned):
            if cleaned[:1].isupper():
                classes.append(cleaned)
            else:
                functions.append(cleaned)

    tokens = re.findall(rf"\b{_IDENTIFIER_RE}\b", lower)
    domain_keywords = [
        token for token in tokens
        if len(token) >= 3 and token not in _STOPWORDS and not token.isdigit()
    ]

    return {
        "function_names": _unique_sorted(functions),
        "class_names": _unique_sorted(classes),
        "module_names": _unique_sorted(modules),
        "file_paths": _unique_sorted(file_paths),
        "domain_keywords": _unique_sorted(domain_keywords[:40]),
    }


def candidate_grounding(candidate: dict, issue_entities: dict) -> dict:
    if not isinstance(candidate, dict):
        return {
            "score": 0.0,
            "has_direct_symbol_match": False,
            "matched_functions": [],
            "matched_classes": [],
            "matched_modules": [],
            "matched_paths": [],
            "matched_keywords": [],
        }

    path = str(candidate.get("path", "") or "").replace("\\", "/")
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    snippet = str(candidate.get("snippet", "") or "")
    snippet_lower = snippet.lower()
    structure = candidate.get("structure", {}) if isinstance(candidate.get("structure", {}), dict) else {}

    available_functions = {normalize_symbol(name) for name in structure.get("functions", []) if isinstance(name, str)}
    available_classes = {normalize_symbol(name) for name in structure.get("classes", []) if isinstance(name, str)}
    available_modules = {normalize_symbol(name) for name in structure.get("imports", []) if isinstance(name, str)}
    available_modules.add(stem)
    available_symbols = available_functions | available_classes | available_modules

    matched_functions = [
        fn for fn in issue_entities.get("function_names", [])
        if symbol_matches(fn, available_symbols) or normalize_symbol(fn).lower() in snippet_lower
    ]
    matched_classes = [
        cls for cls in issue_entities.get("class_names", [])
        if symbol_matches(cls, available_symbols) or normalize_symbol(cls).lower() in snippet_lower
    ]
    matched_modules = []
    for mod in issue_entities.get("module_names", []):
        normalized = normalize_symbol(mod).lower()
        if (
            symbol_matches(mod, available_symbols)
            or normalized == stem.lower()
            or normalized in path.lower()
        ):
            matched_modules.append(mod)

    matched_paths = []
    for issue_path in issue_entities.get("file_paths", []):
        normalized = issue_path.replace("\\", "/").strip("/").lower()
        if normalized and (normalized == path.lower() or normalized.endswith(path.lower()) or path.lower().endswith(normalized)):
            matched_paths.append(issue_path)

    keyword_hits = []
    for keyword in issue_entities.get("domain_keywords", []):
        lowered = keyword.lower()
        if lowered in path.lower() or lowered in snippet_lower:
            keyword_hits.append(keyword)

    score = (
        len(matched_paths) * 10.0
        + len(matched_functions) * 8.0
        + len(matched_classes) * 8.0
        + len(matched_modules) * 5.0
        + min(len(keyword_hits), 8) * 1.5
    )

    return {
        "score": score,
        "has_direct_symbol_match": bool(matched_paths or matched_functions or matched_classes or matched_modules),
        "matched_functions": _unique_sorted(matched_functions),
        "matched_classes": _unique_sorted(matched_classes),
        "matched_modules": _unique_sorted(matched_modules),
        "matched_paths": _unique_sorted(matched_paths),
        "matched_keywords": _unique_sorted(keyword_hits[:12]),
    }


def build_candidate_grounding_map(candidate_files: list[dict], issue_text: str) -> tuple[dict, dict]:
    issue_entities = extract_issue_entities(issue_text)
    grounding_map = {}
    for candidate in candidate_files if isinstance(candidate_files, list) else []:
        if not isinstance(candidate, dict):
            continue
        path = str(candidate.get("path", "") or "").replace("\\", "/").strip("/")
        if not path:
            continue
        grounding_map[path] = candidate_grounding(candidate, issue_entities)
    return issue_entities, grounding_map
