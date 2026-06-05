"""Quick verification of decision node logic — strict 4 rules."""
from nodes.decision import decision_node

# Rule 1: Tests didn't run -> REJECTED
r1 = decision_node({"tests_ran": False, "test_passed": False, "issues_found": []})
assert r1["decision"] == "REJECTED", f"Rule 1 failed: {r1}"
print(f"Rule 1 OK: tests not run -> {r1['decision']}")

# Rule 2: Tests failed -> REJECTED
r2 = decision_node({"tests_ran": True, "test_passed": False, "exit_code": 1, "issues_found": [], "test_failures": ["FAILED test_foo"]})
assert r2["decision"] == "REJECTED", f"Rule 2 failed: {r2}"
print(f"Rule 2 OK: tests failed -> {r2['decision']}")

# Rule 3: Any issues found -> REJECTED
r3 = decision_node({"tests_ran": True, "test_passed": True, "issues_found": ["minor style inconsistency"]})
assert r3["decision"] == "REJECTED", f"Rule 3 failed: {r3}"
print(f"Rule 3 OK: issues found -> {r3['decision']}")

# Rule 4: All good (tests passed, 0 issues) -> APPROVED
r4 = decision_node({"tests_ran": True, "test_passed": True, "issues_found": []})
assert r4["decision"] == "APPROVED", f"Rule 4 failed: {r4}"
print(f"Rule 4 OK: all clear -> {r4['decision']}")

print("\nOK: All decision rules verified.")
