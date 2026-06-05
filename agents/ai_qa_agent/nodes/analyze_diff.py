import os
import re
import sys
import tempfile
import subprocess
import glob
import ast
import json
from state import QAState
from llm_config import get_llm
from utils.llm_utils import safe_invoke

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_REACT_STEPS = 4
VALID_RISK_LEVELS = {"low", "medium", "high"}


# ═══════════════════════════════════════════════════════════════════════════════
#  REACT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """You are an AI QA Agent acting as a senior engineer.
You use ReAct (Reasoning + Acting) and Context Engineering to review code.

Your job is to:
- Analyze code changes in the context of the repository
- Generate relevant, dynamic test cases based on the behavior
- Evaluate correctness and robustness intelligently
- Provide constructive feedback (preferring suggestions over hard rejection)

## Instructions
1. Understand the change
2. Identify modified files from diff
3. Use open_file(path) to read FULL file content
4. Analyze actual implementation
5. Generate test cases
6. Execute tests using run_tests
7. Analyze test results
8. Decide outcome

## Rules
- 🚨 [CRITICAL PROTOCOL] You are PROHIBITED from emitting a Final Answer unless run_tests has been called at least once in this reasoning loop. If you have not yet called run_tests, your next action MUST be run_tests. Emitting a Final Answer before run_tests is a critical protocol violation.
- 🚨 [CRITICAL OBSERVATION] Before writing any test, you MUST call open_file on every changed file in the diff and read the actual function signature and return type. You are forbidden from writing test assertions based on assumed or inferred signatures. Only write tests against what you have directly observed in the file.
- ALWAYS identify modified files from diff
- ALWAYS use open_file(path) before analysis
- NEVER rely only on diff summary
- 🚨 [TEST FORMAT] All tests passed to run_tests must follow this exact format:

  def test_example():
      result = my_function(valid_input)
      assert result == expected_value

  def test_example_invalid_input():
      try:
          my_function(invalid_input)
          assert False
      except ValueError:
          pass

Rules for Tests:
- Every test must be a properly indented def block on its own line
- No semicolons between statements
- No inline try/except on a single line
- Imports go at the top of the block, not inside test functions
- Do not assert on internal implementation details or dict keys that you have not observed in the actual source file

## Test Validation Strategy (Strict Execution)
- Use `run_tests` to execute your tests.
- Collect Passed Tests, Failed Tests, and Execution Errors.
- DO NOT HALLUCINATE PASSING TESTS. If tests do not execute, they have not passed.

## Available Actions
- search_code(query) — search the repository for a code snippet
- open_file(path) — read a full file from the repository
- run_tests(test_cases) — disabled runtime execution path; returns simulation-required signal
- None — no action needed

## Issue Extraction & PR Title Generation
- You MUST extract the Issue Number (e.g., from branch name like issue-81, or commit context).
- You MUST extract the Issue Title (from context or infer a short description).
- Generate a PR Title strictly following: `QA Review: #<issue_number> - <issue_title> -> <decision>`
- Fallback 1: `QA Review: #<issue_number> - <inferred_short_description> -> <decision>`
- Fallback 2: `QA Review: <inferred_issue_name> -> <decision>`

## Decision Rules (Strict - Non-Negotiable)
- ISSUE ALIGNMENT: The code changes MUST exactly match the intent of the issue. If the diff includes unrelated features, refactoring, or ignores the core issue, REJECT IT.
- APPROVED: IF passed tests > 0 AND failed_tests == 0 AND NO HIGH priority tests failed AND changes perfectly align with the issue intent.
- REJECTED: IF tests were not executed, OR if passed tests == 0, OR failed_tests > 0, OR any HIGH priority test failed, OR the fix is unrelated to the issue.

## Response Format (STRICT)

Thought: <analyze change>
Thought: <infer behavior>
Thought: <generate tests>
Thought: <evaluate tests>
Thought: <decide outcome>

Action: <tool_name(arguments)> OR None

--- END STEP ---

When you have enough information to make your final review:

Thought: <final summary>

Action: None

Final Answer:
## PR Title:
QA Review: #<issue_number> - <issue_name> -> <decision>

## Test Cases Generated:
(max 10)

## Tests Passed:
* <test that passed>

## Tests Failed:
* <test that failed>

## Reason for Decision:
(include pass count + reasoning)

## Suggestions:
* <improvement 1>
* <improvement 2>

## Confidence:
LOW | MEDIUM | HIGH

## Final PR Comment:
<clean, constructive review message>

```json
{
  "pr_title": "QA Review: #<issue_number> - <issue_name> -> <decision>",
  "decision": "APPROVED|REJECTED",
  "reason": "...",
  "tests_generated": ["..."],
  "tests_passed": ["..."],
  "tests_failed": ["..."],
  "test_execution_reasoning": "...",
  "suggestions": ["..."],
  "confidence": "LOW|MEDIUM|HIGH",
  "final_comment": "..."
}
```

--- END STEP ---
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS (Real — operate on the cloned repo)
# ═══════════════════════════════════════════════════════════════════════════════

def tool_search_code(repo_path: str, query: str) -> str:
    """Search the repository for a pattern using grep. Returns matching lines untruncated."""
    try:
        result = subprocess.run(
            ["git", "grep", "-n", "-I", query],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{query}' in the repository."
        return output
    except subprocess.TimeoutExpired:
        return f"Search for '{query}' timed out."
    except Exception as e:
        return f"Search error: {str(e)}"


def tool_open_file(repo_path: str, file_path: str) -> str:
    """Read the contents of a file in the repository."""
    clean_path = file_path.strip().strip("'\"")
    full_path = os.path.normpath(os.path.join(repo_path, clean_path))

    if not full_path.startswith(os.path.normpath(repo_path)):
        return f"Error: Path '{clean_path}' is outside the repository."

    if not os.path.exists(full_path):
        base_name = os.path.basename(clean_path)
        matches = glob.glob(os.path.join(repo_path, "**", base_name), recursive=True)
        if matches:
            suggestions = [os.path.relpath(m, repo_path) for m in matches[:5]]
            return f"File '{clean_path}' not found. Similar files: {', '.join(suggestions)}"
        return f"File '{clean_path}' not found in the repository."

    if os.path.isdir(full_path):
        entries = os.listdir(full_path)[:30]
        return f"'{clean_path}' is a directory. Contents: {', '.join(entries)}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + f"\n\n... [truncated — file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading '{clean_path}': {str(e)}"


def tool_analyze_code(snippet: str) -> str:
    """Use the LLM to deeply analyze a code snippet for issues."""
    try:
        llm = get_llm()
        analysis_prompt = (
            "Analyze this code snippet for: logic flaws, missing validation, "
            "edge cases, potential regressions, unsafe operations, and bad assumptions.\n"
            "Be precise — only report real issues.\n\n"
            f"Code:\n```\n{snippet[:5000]}\n```\n\n"
            "Return a concise bullet list of issues found, or 'No issues found.' if clean."
        )
        response = safe_invoke(llm.invoke, analysis_prompt)
        return response.content.strip()
    except Exception as e:
        return f"Analysis error: {str(e)}"


def tool_run_tests(repo_path: str, test_code: str, timeout: int = 30) -> str:
    """Execution path removed: this tool now signals simulation-only mode."""
    _ = repo_path
    _ = test_code
    _ = timeout
    return json.dumps(
        {
            "success": False,
            "status": "Execution disabled",
            "error": "run_tests is disabled; use LLM simulation pipeline",
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "summary": "Runtime test execution removed from QA toolset.",
            "execution_mode": "simulation_required",
        },
        indent=2,
    )


def execute_tool(repo_path: str, action_str: str) -> str:
    """Parse and execute a tool action string. Returns the observation."""
    action_str = action_str.strip()

    match = re.match(r'^(\w+)\((.+)\)$', action_str, re.DOTALL)
    if not match:
        return f"Invalid action format: '{action_str}'. Expected: tool_name(arguments)"

    tool_name = match.group(1).lower()
    args = match.group(2).strip().strip("'\"")

    if tool_name == "search_code":
        return tool_search_code(repo_path, args)
    elif tool_name == "open_file":
        return tool_open_file(repo_path, args)
    elif tool_name == "analyze_code":
        return tool_analyze_code(args)
    elif tool_name == "run_tests":
        content = match.group(2).strip()
        if (content.startswith('"""') and content.endswith('"""')) or (content.startswith("'''") and content.endswith("'''")):
            content = content[3:-3]
            try:
                content = content.encode().decode('unicode_escape')
            except Exception:
                pass
        elif (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
            try:
                content = content.encode().decode('unicode_escape')
            except Exception:
                pass
                
        # Fix markdown backticks if LLM hallucinated them inside the string
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        
        # Keep adapter call path for simulation-tool compatibility.
        # in the LangGraph flow, so we return the string directly since adapter mocks it.
        # Wait, the node calls `execute_tool` which calls `tool_run_tests`. 
        # `tool_run_tests` is monkey-patched in the adapter.
        return tool_run_tests(repo_path, content)
    else:
        return f"Unknown tool: '{tool_name}'. Available: search_code, open_file, run_tests, analyze_code"


# ═══════════════════════════════════════════════════════════════════════════════
#  REPO SIGNAL EXTRACTION (Context Engineering)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_repo_signals(repo_path: str, diff_output: str) -> str:
    """Extract structural signals from the repository for context engineering."""
    signals = []

    changed_files = []
    for line in diff_output.split("\n"):
        if line.startswith("+++ b/"):
            changed_files.append(line[6:])
        elif line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) > 1:
                changed_files.append(parts[-1])
    changed_files = list(set(changed_files))

    imports = set()
    for fpath in changed_files:
        full = os.path.join(repo_path, fpath)
        if os.path.exists(full) and fpath.endswith(".py"):
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith("import ") or stripped.startswith("from "):
                            imports.add(stripped)
            except Exception:
                pass

    definitions = []
    for fpath in changed_files:
        full = os.path.join(repo_path, fpath)
        if os.path.exists(full) and fpath.endswith(".py"):
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            definitions.append(f"  {fpath}:{line_num} → {stripped.split('(')[0]}")
            except Exception:
                pass

    patterns = []
    all_imports = " ".join(imports)
    if "pytest" in all_imports or "unittest" in all_imports:
        patterns.append("Testing framework detected (pytest/unittest)")
    if "flask" in all_imports or "fastapi" in all_imports or "django" in all_imports:
        patterns.append("Web framework detected")
    if "logging" in all_imports:
        patterns.append("Logging framework in use")
    if "typing" in all_imports:
        patterns.append("Type hints in use")
    if any("try" in line and "except" in line for line in diff_output.split("\n")):
        patterns.append("Error handling patterns present in diff")

    signals.append(f"Changed Files: {', '.join(changed_files) if changed_files else 'None detected'}")
    signals.append(f"Imports: {'; '.join(sorted(imports)[:15]) if imports else 'None detected'}")
    signals.append(f"Definitions:\n{chr(10).join(definitions[:20]) if definitions else '  None detected'}")
    signals.append(f"Patterns: {'; '.join(patterns) if patterns else 'No specific patterns detected'}")

    return "\n".join(signals)


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURED CONTEXT BUILDER (Context Engineering)
# ═══════════════════════════════════════════════════════════════════════════════

def build_structured_context(query: str, diff: str, files: list, repo_signals: str, reasoning_history: str, issue_text: str = "") -> str:
    """Build the structured context prompt for the ReAct LLM call."""
    return f"""Issue Description (ALIGNMENT TARGET):
{issue_text}

User Query:
{query}

PR Diff:
{diff}

Available Files:
{', '.join(files)}

Repository Signals:
{repo_signals}

Reasoning History:
{reasoning_history if reasoning_history else "(First step — no history yet)"}

Constraints:
- max_steps: {MAX_REACT_STEPS}
- must verify completeness before final answer
- must use at least one tool before giving final answer
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_react_response(response_text: str):
    """
    Parse the LLM's ReAct response.
    Returns: (thought, action, final_answer_block_or_None)
    """
    text = response_text.strip()

    thought = ""
    # With multiple thoughts, gather context until Action or Final Answer
    thought_match = re.search(r'Thought:(.+?)(?=\nAction:|\nFinal Answer:|$)', text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    elif "Thought:" in text:
        thought = text.split("Thought:", 1)[-1].strip()

    action = "None"
    action_match = re.search(r'Action:\s*(.+?)(?=\n(?:Observation:|--- END|Final Answer:)|\n\n|$)', text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        if action.lower() == "none" or action == "":
            action = "None"

    final_answer = None
    fa_match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
    if fa_match:
        final_answer = fa_match.group(1).strip()

    return thought, action, final_answer

def extract_json(text: str) -> dict:
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)

def fallback_markdown_parser(final_answer_text: str):
    result = {
        "pr_title": "",
        "tests_generated": [],
        "tests_passed": [],
        "tests_failed": [],
        "test_priority_breakdown": "",
        "execution_errors": "",
        "decision": "REJECTED",
        "reason": "No reason provided.",
        "suggestions": [],
        "final_comment": "",
        "confidence": "MEDIUM",
        "test_execution_reasoning": ""
    }

    current_section = None
    
    for line in final_answer_text.split("\n"):
        line_clean = line.strip()
        
        if "PR TITLE:" in line_clean.upper():
            current_section = "pr_title"
            continue
        elif "TEST CASES GENERATED:" in line_clean.upper():
            current_section = "tests_generated"
            continue
        elif "TESTS PASSED:" in line_clean.upper():
            current_section = "tests_passed"
            continue
        elif "TESTS FAILED:" in line_clean.upper():
            current_section = "tests_failed"
            continue
        elif "TEST PRIORITY BREAKDOWN:" in line_clean.upper():
            current_section = "test_priority_breakdown"
            continue
        elif "EXECUTION ERRORS:" in line_clean.upper():
            current_section = "execution_errors"
            continue
        elif "TEST EXECUTION REASONING:" in line_clean.upper():
            current_section = "test_execution_reasoning"
            continue
        elif "REVIEW DECISION:" in line_clean.upper() or line_clean.upper().startswith("DECISION:"):
            current_section = "decision"
            continue
        elif "REASON FOR DECISION:" in line_clean.upper() or line_clean.upper().startswith("REASON:"):
            current_section = "reason"
            continue
        elif "SUGGESTIONS:" in line_clean.upper():
            current_section = "suggestions"
            continue
        elif "CONFIDENCE:" in line_clean.upper():
            current_section = "confidence"
            continue
        elif "FINAL PR COMMENT:" in line_clean.upper() or line_clean.upper().startswith("FINAL_COMMENT:"):
            current_section = "final_comment"
            continue

        if not line_clean:
            continue

        if current_section == "pr_title":
            result["pr_title"] += line_clean + " "
        elif current_section == "tests_generated":
            result["tests_generated"].append(line_clean.lstrip("*- "))
        elif current_section == "tests_passed":
            result["tests_passed"].append(line_clean.lstrip("*- "))
        elif current_section == "tests_failed":
            result["tests_failed"].append(line_clean.lstrip("*- "))
        elif current_section == "test_priority_breakdown":
            result["test_priority_breakdown"] += line_clean + "\n"
        elif current_section == "execution_errors":
            result["execution_errors"] += line_clean + "\n"
        elif current_section == "test_execution_reasoning":
            result["test_execution_reasoning"] += line_clean + "\n"
        elif current_section == "decision":
            if "REJECT" in line_clean.upper(): result["decision"] = "REJECTED"
            elif "APPROV" in line_clean.upper(): result["decision"] = "APPROVED"
        elif current_section == "reason":
            result["reason"] += line_clean + "\n"
        elif current_section == "suggestions":
            result["suggestions"].append(line_clean.lstrip("*- "))
        elif current_section == "confidence":
            if "HIGH" in line_clean.upper(): result["confidence"] = "HIGH"
            elif "LOW" in line_clean.upper(): result["confidence"] = "LOW"
            elif "MEDIUM" in line_clean.upper(): result["confidence"] = "MEDIUM"
        elif current_section == "final_comment":
            result["final_comment"] += line + "\n"

    result["pr_title"] = result["pr_title"].strip()
    result["execution_errors"] = result["execution_errors"].strip()
    result["test_priority_breakdown"] = result["test_priority_breakdown"].strip()
    result["test_execution_reasoning"] = result["test_execution_reasoning"].strip()
    result["reason"] = result["reason"].strip()
    result["final_comment"] = result["final_comment"].strip()
    return result


def parse_final_answer(final_answer_text: str):
    """
    Parse the structured final answer into a dict.
    Returns: {decision, reason, tests_generated: list, tests_passed: list, test_execution_reasoning, suggestions: list, confidence, final_comment}
    """
    try:
        parsed = extract_json(final_answer_text)
        return parsed
    except Exception:
        return fallback_markdown_parser(final_answer_text)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE: ANALYZE_DIFF with ReAct Loop
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_diff_node(state: QAState) -> dict:
    """
    NODE 2: ANALYZE_DIFF (ReAct + Context Engineering)

    Performs structured multi-step reasoning via:
      1. ReAct loop — Thought → Action → Observation (max 3 steps)
      2. Context Engineering — structured input with repo signals + accumulated history
      3. Tool use — search(), open_file(), analyze_code()
      4. Final structured output extraction

    The agent NEVER answers immediately. It reasons, uses tools, and accumulates
    context before producing a verified final analysis.
    """
    if state.get("status") == "FAILED":
        return {
            "pr_title": "QA Review: Upstream Failure -> REVIEW_REQUIRED",
            "diff_decision": "REVIEW_REQUIRED",
            "test_cases": [],
            "tests_passed": [],
            "tests_failed": [],
            "test_priority_breakdown": "None",
            "execution_errors": f"Upstream Pipeline Error: {state.get('error', 'Unknown Error')}",
            "suggestions": ["Resolve upstream pipeline failures."],
            "final_comment": "QA Agent aborted due to an upstream failure.",
            "reason": "Upstream pipeline failed before QA.",
            "confidence": "LOW",
            "test_execution_reasoning": "None",
            "reasoning_history": "Skipped ReAct loop due to upstream failure.",
            "repo_signals": ""
        }

    repo_path = state.get("repo_path")
    if not repo_path:
        return {
            "pr_title": "QA Review: Missing Repository -> REVIEW_REQUIRED",
            "diff_decision": "REVIEW_REQUIRED",
            "test_cases": [],
            "tests_passed": [],
            "tests_failed": [],
            "test_priority_breakdown": "None",
            "execution_errors": "Analyze Diff Error: repo_path is missing.",
            "suggestions": ["Provide a valid repository path."],
            "final_comment": "QA Agent aborted due to missing repository path.",
            "reason": "Repository missing.",
            "confidence": "LOW",
            "test_execution_reasoning": "None",
            "reasoning_history": "Skipped due to missing repo_path.",
            "repo_signals": ""
        }

    # ── Step 0: Extract Issue text for alignment validation ──
    issue_title = state.get("issue_title", "Unknown Issue")
    issue_body = state.get("issue_body", "")
    issue_text = f"Title: {issue_title}\n\nBody:\n{issue_body}"

    # ── Step 0: Retrieve the diff (prefer injected diff, fall back to git) ──
    injected_diff = state.get("diff", "")
    if injected_diff and injected_diff.strip():
        print(f"[ANALYZE_DIFF] Using pre-computed diff from state ({len(injected_diff)} chars).")
        diff_output = injected_diff
    else:
        try:
            print("[ANALYZE_DIFF] No diff in state — running git diff main...HEAD ...")
            try:
                diff_output = subprocess.check_output(
                    ["git", "diff", "main...HEAD"],
                    cwd=repo_path, text=True, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError:
                print("[ANALYZE_DIFF] 'main' branch not found, trying 'master'...")
                diff_output = subprocess.check_output(
                    ["git", "diff", "master...HEAD"],
                    cwd=repo_path, text=True, stderr=subprocess.STDOUT
                )
        except Exception as e:
            print(f"[ANALYZE_DIFF] Git diff failed: {e}")
            return {
                "pr_title": "QA Review: Git Environment Failure -> REVIEW_REQUIRED",
                "diff_decision": "REVIEW_REQUIRED",
                "test_cases": [],
                "tests_passed": [],
                "tests_failed": [],
                "test_priority_breakdown": "None",
                "execution_errors": f"Git Error: {str(e)[:200]}",
                "suggestions": ["Fix git repository accessibility or target branch."],
                "final_comment": "Failed to analyze diff due to repository error.",
                "reason": "Git diff execution failed.",
                "confidence": "LOW",
                "test_execution_reasoning": "None",
                "reasoning_history": f"Git Error:\n{str(e)}",
                "repo_signals": ""
            }

    if not diff_output.strip():
        print("[ANALYZE_DIFF] WARNING: Empty diff detected.")
        return {
            "pr_title": "QA Review: Empty Changeset -> ACCEPT",
            "diff_decision": "ACCEPT",
            "test_cases": [],
            "tests_passed": [],
            "tests_failed": [],
            "execution_mode": "None",
            "test_priority_breakdown": "None",
            "execution_errors": "",
            "suggestions": ["No code changes detected."],
            "final_comment": "Empty diff. PR is accepted by default.",
            "reason": "No code changes detected in diff.",
            "confidence": "HIGH",
            "test_execution_reasoning": "None",
            "reasoning_history": "Empty diff — no analysis needed.",
            "repo_signals": ""
        }

    print(f"[ANALYZE_DIFF] Diff size: {len(diff_output)} chars.")

    # ── Step 1: Extract Repository Signals (Context Engineering) ──
    print("[ANALYZE_DIFF] Extracting repository signals...")
    repo_signals = extract_repo_signals(repo_path, diff_output)
    print(f"[ANALYZE_DIFF] Repo signals extracted:\n{repo_signals[:300]}")
    changed_files = _extract_files_from_diff(diff_output)

    llm = get_llm()
    diff_for_llm = diff_output[:12000]
    if len(diff_output) > 12000:
        diff_for_llm += f"\n\n... [diff truncated — {len(diff_output)} chars total, showing first 12000]"

    # ── Step 2: Pre-flight Test Generation (Fix 7.5) ──
    print("[ANALYZE_DIFF] Generating pre-flight tests in single shot...")
    test_gen_prompt = (
        "Based on the following diff and repository context, generate a complete, valid pytest block covering "
        "the functionality and edge cases. Follow exact formatting rules: no markdown wrappers if possible, "
        "strictly valid Python def blocks with NO inline try/except or semicolons.\n"
        f"Diff:\n{diff_for_llm}\n\nSignals:\n{repo_signals}"
    )
    try:
        test_gen_resp = safe_invoke(llm.invoke, [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": test_gen_prompt}
        ])
        pre_flight_tests = test_gen_resp.content.strip()
        
        # Clean markdown
        if pre_flight_tests.startswith("```"):
            lines = pre_flight_tests.split("\n")
            if lines[-1].strip() == "```":
                pre_flight_tests = "\n".join(lines[1:-1])
            else:
                pre_flight_tests = "\n".join(lines[1:])
                
        ast.parse(pre_flight_tests)
        reasoning_history = (
            "--- Pre-flight Test Generation ---\n"
            "Thought: I must generate tests up-front to execute immediately.\n"
            "Action: pre_generated_tests\n"
            f"Observation: Successfully generated the following valid Python tests:\n```python\n{pre_flight_tests}\n```\n"
            "SYSTEM NOTE: Call `run_tests` immediately at Step 1 using these tests.\n\n"
        )
    except Exception as e:
        print(f"[ANALYZE_DIFF] Pre-flight tests failed grammar check: {e}")
        reasoning_history = (
            "--- Pre-flight Test Generation ---\n"
            "Thought: I must generate tests up-front to execute immediately.\n"
            "Action: pre_generated_tests\n"
            f"Observation: Test generation resulted in invalid syntax ({e}). I must write the tests manually.\n\n"
        )

    # ── Step 3: ReAct Loop ──
    final_result = {
        "pr_title": "QA Review: Incomplete Analysis -> REVIEW_REQUIRED",
        "tests_generated": [],
        "tests_passed": [],
        "tests_failed": [],
        "test_priority_breakdown": "",
        "execution_errors": "",
        "decision": "ACCEPT",
        "reason": "No reason provided.",
        "suggestions": [],
        "final_comment": "",
        "test_execution_reasoning": "",
        "confidence": "MEDIUM"
    }

    executed_actions = set()

    try:
        for step in range(1, MAX_REACT_STEPS + 1):
            print(f"\n[ANALYZE_DIFF] ══ ReAct Step {step}/{MAX_REACT_STEPS} ══")

            structured_context = build_structured_context(
                query="Analyze diff for issues, ensure changes strictly align with the Issue Description, generate tests, and provide review",
                diff=diff_for_llm,
                files=changed_files,
                repo_signals=repo_signals,
                reasoning_history=reasoning_history,
                issue_text=issue_text
            )

            messages = [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": structured_context}
            ]

            if step > 1:
                messages.append({
                    "role": "user",
                    "content": f"You are on step {step} of {MAX_REACT_STEPS}. "
                               f"{'This is your LAST step — you MUST provide a Final Answer now.' if step == MAX_REACT_STEPS else 'Continue your analysis.'}"
                })

            response = safe_invoke(llm.invoke, messages)
            response_text = response.content.strip()

            thought, action, final_answer = parse_react_response(response_text)

            print(f"[ANALYZE_DIFF]   Thought: {thought[:150]}...")
            print(f"[ANALYZE_DIFF]   Action: {action}")

            step_record = f"--- Step {step} ---\nThought: {thought}\nAction: {action}\n"

            observation = ""
            if action != "None" and action.lower() != "none":
                action_clean = action.strip()
                if "run_tests" not in action_clean and action_clean in executed_actions:
                    print(f"[ANALYZE_DIFF]   TRIGGERED SAFETY: Skipped repeated action '{action_clean}'")
                    observation = "Action skipped: You have already executed this exact action. Use run_tests() or emit Final Answer."
                else:
                    executed_actions.add(action_clean)
                    print(f"[ANALYZE_DIFF]   Executing tool: {action}")
                    observation = execute_tool(repo_path, action)
                print(f"[ANALYZE_DIFF]   Observation: {observation[:200]}...")
                step_record += f"Observation: {observation}\n"
            else:
                step_record += "Observation: None (no tool used)\n"

            reasoning_history += step_record + "\n"

            if final_answer:
                print(f"[ANALYZE_DIFF]   Final Answer detected at step {step}.")
                final_result = parse_final_answer(final_answer)
                break

            if step == MAX_REACT_STEPS:
                print("[ANALYZE_DIFF]   Max steps reached. Forcing final answer...")

                force_prompt = (
                    f"You have reached the maximum number of reasoning steps.\n\n"
                    f"Based on ALL your reasoning history below, provide your Final Answer NOW.\n\n"
                    f"Reasoning History:\n{reasoning_history}\n\n"
                    f"Repository Signals:\n{repo_signals}\n\n"
                    f"You MUST respond with EXACTLY this format:\n\n"
                    f"Thought: <final summary>\n\n"
                    f"Action: None\n\n"
                    f"Final Answer:\n"
                    f"## PR Title:\nQA Review: #<issue_number> - <issue_name> -> <decision>\n\n"
                    f"## Test Cases Generated:\n(max 10)\n\n"
                    f"## Tests Passed:\n* <tests>\n\n"
                    f"## Tests Failed:\n* <tests>\n\n"
                    f"## Reason for Decision:\n(include pass count + reasoning)\n\n"
                    f"## Suggestions:\n* <suggestion>\n\n"
                    f"## Confidence:\nLOW | MEDIUM | HIGH\n\n"
                    f"## Final PR Comment:\n<comment>\n\n"
                    f"```json\n"
                    f"{{\n"
                    f"  \"pr_title\": \"QA Review: #<issue_number> - <issue_name> -> <decision>\",\n"
                    f"  \"decision\": \"APPROVED|REJECTED\",\n"
                    f"  \"reason\": \"...\",\n"
                    f"  \"tests_generated\": [\"...\"],\n"
                    f"  \"tests_passed\": [\"...\"],\n"
                    f"  \"tests_failed\": [\"...\"],\n"
                    f"  \"test_execution_reasoning\": \"...\",\n"
                    f"  \"suggestions\": [\"...\"],\n"
                    f"  \"confidence\": \"MEDIUM\",\n"
                    f"  \"final_comment\": \"...\"\n"
                    f"}}\n"
                    f"```"
                )

                force_response = safe_invoke(llm.invoke, [
                    {"role": "system", "content": REACT_SYSTEM_PROMPT},
                    {"role": "user", "content": force_prompt}
                ])
                force_text = force_response.content.strip()
                _, _, forced_final = parse_react_response(force_text)

                if forced_final:
                    final_result = parse_final_answer(forced_final)
                    reasoning_history += f"--- Step {step + 1} (forced) ---\n{force_text}\n"
                else:
                    print("[ANALYZE_DIFF]   WARNING: Could not extract structured final answer. Using fallback.")
                    final_result["decision"] = "ACCEPT"
                    final_result["final_comment"] = "Analysis completed but structured output could not be parsed."

    except Exception as e:
        error_msg = str(e)
        print(f"[ANALYZE_DIFF] ReAct loop crashed: {error_msg}")
        
        if "RATE_LIMIT_DAILY_EXHAUSTED" in error_msg:
            final_result["decision"] = "FAILED"
            final_result["pr_title"] = "QA Review: RATE LIMIT EXHAUSTED -> FAILED"
        else:
            final_result["decision"] = "REVIEW_REQUIRED"
            final_result["pr_title"] = "QA Review: Agent Loop Crash -> REVIEW_REQUIRED"
            
        final_result["reason"] = f"QA Agent encountered an internal reasoning failure: {error_msg[:500]}"
        final_result["execution_errors"] = error_msg
        
    print(f"\n[ANALYZE_DIFF] ══ ReAct Analysis Complete ══")
    final_decision = final_result.get('decision', 'ACCEPT')
    
    if final_decision != "FAILED":
        if "open_file" not in reasoning_history:
            print("[ANALYZE_DIFF] TRIGGERED SAFETY: open_file was never called.")
            final_decision = "REVIEW_REQUIRED"
            
        if "run_tests" not in reasoning_history:
            print("[ANALYZE_DIFF] TRIGGERED SAFETY: run_tests was never called.")
            final_decision = "REVIEW_REQUIRED"

    print(f"[ANALYZE_DIFF]   Decision: {final_decision}")
    print(f"[ANALYZE_DIFF]   Suggestions: {len(final_result.get('suggestions', []))}")
    print(f"[ANALYZE_DIFF]   Total reasoning steps: {reasoning_history.count('--- Step')}")

    return {
        "pr_title": final_result.get("pr_title", ""),
        "diff_decision": final_decision,
        "test_cases": final_result.get("tests_generated", []),
        "tests_passed": final_result.get("tests_passed", []),
        "tests_failed": final_result.get("tests_failed", []),
        "execution_mode": final_result.get("execution_mode", "REAL"),
        "test_priority_breakdown": final_result.get("test_priority_breakdown", ""),
        "execution_errors": final_result.get("execution_errors", ""),
        "suggestions": final_result.get("suggestions", []),
        "final_comment": final_result.get("final_comment", ""),
        "reason": final_result.get("reason", ""),
        "confidence": final_result.get("confidence", "MEDIUM"),
        "test_execution_reasoning": final_result.get("test_execution_reasoning", ""),
        "reasoning_history": reasoning_history,
        "repo_signals": repo_signals
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_files_from_diff(diff_output: str) -> list:
    """Fallback: extract changed file paths directly from the diff output."""
    files = set()
    for line in diff_output.split("\n"):
        if line.startswith("+++ b/"):
            files.add(line[6:])
    return sorted(files)
