from agents.ai_issue_agent.agents.issue_grounding import build_candidate_grounding_map, extract_issue_entities
from agents.ai_issue_agent.agents.output_validators import LLMOutputValidator


def test_issue_grounding_prefers_real_symbols():
    issue = "Fix parse_user() in parser.py so empty input returns None instead of crashing."
    candidates = [
        {
            "path": "src/parser.py",
            "snippet": "def parse_user(text):\n    return text.strip()\n",
            "structure": {"functions": ["parse_user"], "classes": [], "imports": []},
        },
        {
            "path": "src/other.py",
            "snippet": "def unrelated(value):\n    return value\n",
            "structure": {"functions": ["unrelated"], "classes": [], "imports": []},
        },
    ]

    entities, grounding = build_candidate_grounding_map(candidates, issue)

    assert "parse_user" in [name.lower() for name in entities["function_names"]]
    assert grounding["src/parser.py"]["has_direct_symbol_match"] is True
    assert grounding["src/other.py"]["has_direct_symbol_match"] is False


def test_behavioral_change_rejects_format_only_python_patch():
    original = """def parse_user(text):\n    return text.strip()\n"""
    patched = """def parse_user(text):\n    return text.strip()\n\n"""

    ok, reason = LLMOutputValidator.validate_behavioral_change(
        original,
        patched,
        filename="parser.py",
        target_symbols=["parse_user"],
    )

    assert ok is False
    assert "identical" in reason.lower() or "ast is unchanged" in reason.lower()


def test_behavioral_change_requires_grounded_target_to_change():
    original = """def parse_user(text):\n    return text.strip()\n\ndef format_user(name):\n    return name.lower()\n"""
    patched = """def parse_user(text):\n    return text.strip()\n\ndef format_user(name):\n    return name.upper()\n"""

    ok, reason = LLMOutputValidator.validate_behavioral_change(
        original,
        patched,
        filename="parser.py",
        target_symbols=["parse_user"],
    )

    assert ok is False
    assert "grounded target" in reason.lower()


def test_analyze_python_changes_detects_removed_symbols_and_empty_bodies():
    original = """def parse_user(text):\n    return text.strip()\n\ndef format_user(name):\n    return name.lower()\n"""
    patched = """def parse_user(text):\n    pass\n"""

    analysis = LLMOutputValidator.analyze_python_changes(original, patched)

    assert "format_user" in analysis["removed_definitions"]
    assert "parse_user" in analysis["empty_definitions"]
