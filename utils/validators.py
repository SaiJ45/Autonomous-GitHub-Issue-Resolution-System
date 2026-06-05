import os

from typing import Any

def validate_patch(patch: Any) -> bool:
    """Validate that the patch is a dictionary mapping file paths to content strings."""
    if not isinstance(patch, dict) or not patch:
        return False

    for k, v in patch.items():
        if not isinstance(k, str) or not k.strip():
            return False
        normalized = k.replace("\\", "/").strip()
        if (
            os.path.isabs(k)
            or ":" in normalized.split("/")[0]
            or normalized.startswith("/")
            or normalized.startswith("../")
            or "/../" in normalized
            or normalized in {".", ".."}
        ):
            return False
        if not isinstance(v, str) or not v.strip():
            return False
    return True

def validate_qa_output(output: Any) -> bool:
    """Validate that the QA output matches the expected strict dict interface."""
    if not isinstance(output, dict):
        return False
    
    required_keys = {"status", "passed_tests", "failed_tests", "issues", "suggestions"}
    if not required_keys.issubset(set(output.keys())):
        return False
        
    if output["status"] not in ["APPROVED", "REJECTED"]:
        return False
        
    if not isinstance(output["passed_tests"], int) or not isinstance(output["failed_tests"], int):
        return False
        
    if not isinstance(output["issues"], list):
        return False
        
    if not isinstance(output["suggestions"], str):
        return False
        
    return True
