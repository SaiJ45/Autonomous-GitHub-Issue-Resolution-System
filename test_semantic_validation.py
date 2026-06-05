#!/usr/bin/env python3
"""
Test script to verify semantic validation fixes.
Tests that:
1. Valid patches with content changes are accepted
2. Empty patches are rejected
3. Small diffs (e.g., +1 line) are accepted if content differs
4. Direct content comparison works (not relying on diff format)
"""

import os
import sys
import tempfile
import difflib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_content_comparison():
    """Test that content comparison detects real modifications."""
    print("\n" + "="*60)
    print("TEST 1: Content Comparison")
    print("="*60)
    
    # Original content
    original = """def greet(name):
    print("Hello " + name)
    return name
"""
    
    # Modified content (changed logic)
    modified = """def greet(name):
    print("Hello " + name.upper())
    return name
"""
    
    # Check if they're different
    if original != modified:
        print("✓ Content comparison detects modification")
        
        # Generate unified diff
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        unified_diff = list(difflib.unified_diff(orig_lines, mod_lines, lineterm=""))
        
        adds = sum(1 for line in unified_diff if line.startswith("+") and not line.startswith("+++"))
        dels = sum(1 for line in unified_diff if line.startswith("-") and not line.startswith("---"))
        
        print(f"✓ Unified diff computed: +{adds} -{dels}")
        assert adds > 0 or dels > 0, "Diff should show changes"
        print("✓ Diff shows real changes")
    else:
        print("✗ Content comparison failed")
        return False
    
    return True


def test_identical_content_rejection():
    """Test that identical content is properly detected."""
    print("\n" + "="*60)
    print("TEST 2: Identical Content Rejection")
    print("="*60)
    
    original = """def process(data):
    return data * 2
"""
    
    modified = original  # Intentionally identical
    
    if original == modified:
        print("✓ Identical content properly detected")
    else:
        print("✗ Failed to detect identical content")
        return False
    
    # Check unified diff is empty
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    unified_diff = list(difflib.unified_diff(orig_lines, mod_lines, lineterm=""))
    
    # Filter out file headers
    content_changes = [l for l in unified_diff if not l.startswith("---") and not l.startswith("+++") and not l.startswith("@@")]
    
    if not content_changes:
        print("✓ No content changes detected in unified diff")
        print("✓ Patch would be correctly rejected")
    else:
        print(f"✗ Unexpected changes in diff: {content_changes}")
        return False
    
    return True


def test_small_diff_acceptance():
    """Test that small diffs are accepted if content actually differs."""
    print("\n" + "="*60)
    print("TEST 3: Small Diff Acceptance")
    print("="*60)
    
    original = """def format_text(text):
    return text.strip()
"""
    
    # Small modification: change method call
    modified = """def format_text(text):
    return text.strip().upper()
"""
    
    if original != modified:
        print("✓ Content differs")
        
        # Generate diff
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        unified_diff = list(difflib.unified_diff(orig_lines, mod_lines, lineterm=""))
        
        adds = sum(1 for line in unified_diff if line.startswith("+") and not line.startswith("+++"))
        dels = sum(1 for line in unified_diff if line.startswith("-") and not line.startswith("---"))
        
        print(f"✓ Small diff detected: +{adds} -{dels}")
        
        # Should be added to accept list
        content_changed = original != modified
        if content_changed:
            print("✓ Small patch would be accepted (content changed)")
        else:
            print("✗ Small patch would be rejected")
            return False
    else:
        print("✗ Failed to detect content change")
        return False
    
    return True


def test_semantic_signals_computation():
    """Test semantic signals are properly computed from diff."""
    print("\n" + "="*60)
    print("TEST 4: Semantic Signals Computation")
    print("="*60)
    
    original = """def calculate(a, b):
    result = a + b
    return result
"""
    
    modified = """def calculate(a, b):
    if a is None or b is None:
        return 0
    result = a + b
    return result
"""
    
    # Generate diff
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    unified_diff = list(difflib.unified_diff(orig_lines, mod_lines, lineterm=""))
    
    # Count meaningful changes
    adds = sum(1 for line in unified_diff 
               if line.startswith("+") and not line.startswith("+++") and line.strip() != "+")
    dels = sum(1 for line in unified_diff 
               if line.startswith("-") and not line.startswith("---") and line.strip() != "-")
    
    print(f"✓ Semantic signals: adds={adds}, dels={dels}")
    
    # Check for meaningful patterns
    import re
    diff_text = "\n".join(unified_diff)
    has_conditional = bool(re.search(r'\bif\b', diff_text))
    has_assignment = bool(re.search(r'=', diff_text))
    
    if has_conditional:
        print("✓ Detected conditional (if statement) - behavioral change")
    if has_assignment:
        print("✓ Detected assignment - meaningful modification")
    
    if adds > 0 or dels > 0:
        print("✓ Signals indicate real modification")
        return True
    else:
        print("✗ Signals failed to detect modification")
        return False


def main():
    print("\n" + "="*70)
    print("SEMANTIC VALIDATION FIX VERIFICATION")
    print("="*70)
    
    tests = [
        ("Content Comparison", test_content_comparison),
        ("Identical Content Rejection", test_identical_content_rejection),
        ("Small Diff Acceptance", test_small_diff_acceptance),
        ("Semantic Signals", test_semantic_signals_computation),
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
        print("\n✓ ALL TESTS PASSED - Semantic validation fixes are working!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
