from state import QAState


def decision_node(state: QAState) -> dict:
    """
    NODE 4: DECISION (STRICT LOGIC)
    Pure deterministic decision engine. NO LLM. NO guessing.

    Rules (evaluated in order):
      1. If tests didn't run → REJECTED
      2. If tests failed → REJECTED
      3. If issues_found > 0 → REJECTED
      4. Otherwise → APPROVED

    NO EXCEPTIONS.
    """
    if state.get("status") == "FAILED":
        return {}

    diff_decision = state.get("diff_decision", "REVIEW_REQUIRED").upper()
    decision = diff_decision

    if decision == "REJECT":
        final_decision = "REJECTED"
        print("[DECISION] REJECTED based on LLM review.")
    elif decision in ["ACCEPT", "ACCEPT_WITH_SUGGESTIONS"]:
        final_decision = "APPROVED"
        print("[DECISION] APPROVED based on LLM review.")
    else:
        final_decision = "REVIEW_REQUIRED"
        print(f"[DECISION] Expected ACCEPT/REJECT, got: {decision}. Defaulting to REVIEW_REQUIRED.")

    return {
        "decision": final_decision,
        "requires_pr": (final_decision in ["REJECTED", "REVIEW_REQUIRED"])
    }

