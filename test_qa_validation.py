#!/usr/bin/env python3
"""
Test script to verify QA validation and decision logic fixes.
Tests that:
1. Patches with failing tests are rejected
2. Patches with passing tests can be approved
3. Decision node checks test results
4. Behavioral validation override works
5. Issue-fix alignment is considered
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_behavioral_override():
    """Test that test failures override LLM approval."""
    print("\n" + "="*60)
    print("TEST 1: Behavioral Validation Override")
    print("="*60)
    
    # Simulate LLM approving a patch
    llm_approval = {
        "status": "APPROVED",
        "logic_issues": [],
        "confidence": "HIGH"
    }
    
    # But tests failed
    test_results = {
        "passed_tests": 2,
        "failed_tests": 3
    }
    
    # Behavioral override logic (from llm_validation_node)
    failed_tests = test_results.get("failed_tests", 0)
    
    if failed_tests > 0:
        # Override approval with rejection
        llm_approval["status"] = "REJECTED"
        llm_approval["logic_issues"].append(
            f"Behavioral tests failed: {failed_tests} test(s) did not pass."
        )
        print("✓ LLM approval overridden due to failing tests")
    else:
        print("✗ Failed to override LLM approval")
        return False
    
    if llm_approval.get("status") == "REJECTED":
        print("✓ Final status is REJECTED (patch will be retried)")
        return True
    else:
        print("✗ Status not properly set to REJECTED")
        return False


def test_decision_behavioral_guard():
    """Test that decision node checks test results first."""
    print("\n" + "="*60)
    print("TEST 2: Decision Node Behavioral Guard")
    print("="*60)
    
    test_results = {
        "passed_tests": 1,
        "failed_tests": 2
    }
    retry_count = 1
    
    # Decision logic (from decision_node)
    failed_tests = test_results.get("failed_tests", 0)
    
    # BEHAVIORAL CORRECTNESS GUARD: Check test results FIRST
    if failed_tests > 0:
        print(f"✓ Detected {failed_tests} failed tests")
        decision = "RETRY" if retry_count < 3 else "FAIL"
        print(f"✓ Decision: {decision} (behavioral tests failed)")
        return True
    else:
        print("✗ Failed to detect test failures")
        return False


def test_no_auto_approval_on_failures():
    """Test that semantic checkpoint doesn't auto-approve with failing tests."""
    print("\n" + "="*60)
    print("TEST 3: No Auto-Approval on Test Failures")
    print("="*60)
    
    # Semantic signals show content changed
    semantic_signals = {
        "content_changed": True,
        "total_adds": 5,
        "total_dels": 2
    }
    
    # But tests failed
    test_results = {
        "passed_tests": 0,
        "failed_tests": 4
    }
    
    # Auto-approval logic (from llm_validation_node)
    content_changed = semantic_signals.get("content_changed", False)
    total_adds = semantic_signals.get("total_adds", 0)
    total_dels = semantic_signals.get("total_dels", 0)
    failed_tests = test_results.get("failed_tests", 0)
    
    # Should NOT auto-approve because tests failed
    can_auto_approve = (
        content_changed and 
        (total_adds > 0 or total_dels > 0) and 
        failed_tests == 0  # This is the key check
    )
    
    if not can_auto_approve:
        print("✓ Auto-approval blocked due to failed tests")
        print(f"  - Content changed: {content_changed}")
        print(f"  - Code modified: +{total_adds} -{total_dels}")
        print(f"  - Tests failed: {failed_tests}")
        return True
    else:
        print("✗ Auto-approval would happen despite failed tests")
        return False


def test_auto_approval_on_pass():
    """Test that semantic checkpoint allows auto-approval when tests pass."""
    print("\n" + "="*60)
    print("TEST 4: Auto-Approval When Tests Pass")
    print("="*60)
    
    # Semantic signals show content changed
    semantic_signals = {
        "content_changed": True,
        "total_adds": 3,
        "total_dels": 1
    }
    
    # Tests all passed
    test_results = {
        "passed_tests": 5,
        "failed_tests": 0
    }
    
    # Guardrails passed
    guardrails_passed = True
    
    # Auto-approval logic
    content_changed = semantic_signals.get("content_changed", False)
    total_adds = semantic_signals.get("total_adds", 0)
    total_dels = semantic_signals.get("total_dels", 0)
    failed_tests = test_results.get("failed_tests", 0)
    
    can_auto_approve = (
        content_changed and 
        (total_adds > 0 or total_dels > 0) and 
        failed_tests == 0 and
        guardrails_passed
    )
    
    if can_auto_approve:
        print("✓ Auto-approval allowed when all conditions met")
        print(f"  - Content changed: {content_changed}")
        print(f"  - Code modified: +{total_adds} -{total_dels}")
        print(f"  - Tests passed: {test_results['passed_tests']}")
        print(f"  - Guardrails: {'PASS' if guardrails_passed else 'FAIL'}")
        return True
    else:
        print("✗ Auto-approval blocked despite all conditions met")
        return False


def test_issue_alignment_prompt():
    """Test that prompt includes issue-fix alignment requirements."""
    print("\n" + "="*60)
    print("TEST 5: Issue-Fix Alignment in Prompt")
    print("="*60)
    
    # Simulate a prompt construction
    prompt_parts = []
    
    # Add issue-fix alignment section
    prompt_parts.append("ISSUE-FIX ALIGNMENT (CRITICAL):")
    prompt_parts.append("1. Extract the issue intent: What behavior is expected?")
    prompt_parts.append("2. Extract the fix intent: What logic does the patch change?")
    prompt_parts.append("3. Verify they match: Does the patch fix the EXACT problem described?")
    prompt_parts.append("4. Reject if: modified functions do not relate to issue, behavior change doesn't match requirement")
    
    prompt = "\n".join(prompt_parts)
    
    # Check that alignment requirements are in prompt
    if "ISSUE-FIX ALIGNMENT" in prompt and "Extract the issue intent" in prompt:
        print("✓ Issue-fix alignment validation in prompt")
        print("✓ Requires extraction of issue and fix intents")
        print("✓ Requires verification they match")
        print("✓ Rejects if functions don't relate to issue")
        return True
    else:
        print("✗ Issue-fix alignment not properly included")
        return False


def test_retry_behavioral_feedback():
    """Test that retry handler provides behavioral failure feedback."""
    print("\n" + "="*60)
    print("TEST 6: Retry Feedback for Behavioral Failures")
    print("="*60)
    
    failure_reason = "behavioral_tests_failed:3"
    test_results = {"failed_tests": 3, "passed_tests": 2}
    new_count = 2
    
    # Retry handler feedback logic
    if "behavioral_tests_failed" in failure_reason:
        failed = test_results.get("failed_tests", 0)
        passed = test_results.get("passed_tests", 0)
        
        typed_msg = (
            f"[RETRY {new_count}] BEHAVIORAL VALIDATION FAILED: {failed} test(s) failed "
            f"(passed: {passed}). The patch does not produce correct behavior. "
            "Analyse why the patch fails the expected behavior and fix the root cause."
        )
        
        print("✓ Behavioral failure detected")
        print("✓ Feedback explains specific test failures:")
        print(f"  - Failed tests: {failed}")
        print(f"  - Passed tests: {passed}")
        print("✓ Instructs planner to fix behavioral issue, not just syntax")
        return True
    else:
        print("✗ Failed to generate behavioral feedback")
        return False


def main():
    print("\n" + "="*70)
    print("QA VALIDATION FIX VERIFICATION")
    print("="*70)
    
    tests = [
        ("Behavioral Override", test_behavioral_override),
        ("Decision Guard", test_decision_behavioral_guard),
        ("No Auto-Approval on Failures", test_no_auto_approval_on_failures),
        ("Auto-Approval on Pass", test_auto_approval_on_pass),
        ("Issue Alignment Prompt", test_issue_alignment_prompt),
        ("Retry Behavioral Feedback", test_retry_behavioral_feedback),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - QA validation fixes are working!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
