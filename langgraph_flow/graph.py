from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph_flow.state import AgentState, DecisionType, FailureType, PipelineStatus
from langgraph_flow.nodes import (
    fetch_issues_node,
    select_issue_node,
    setup_repo_node,
    issue_agent_node,
    issue_pr_node,
    integration_node,
    diff_node,
    test_generation_agent_node,
    test_simulation_agent_node,
    llm_validation_node,
    qa_pr_node,
    decision_node,
    qa_normalization_node,
    retry_handler_node,
    merge_node,
    final_merge_pr_node,
    fail_node,
    complete_node
)


def _sget(state: AgentState | dict, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _route_or_fail(state: AgentState | dict) -> str:
    status = _sget(state, "status", "")
    if isinstance(status, str) and (status.startswith("FAILED") or status == PipelineStatus.REJECTED.value):
        return "fail"
    return "continue"


def _route_recoverable_stage(state: AgentState | dict) -> str:
    status = _sget(state, "status", "")
    decision = _sget(state, "decision")
    retry_count = _sget(state, "retry_count", 0) or 0
    failure_type = _sget(state, "failure_type")

    if isinstance(status, PipelineStatus):
        status = status.value
    if isinstance(decision, DecisionType):
        decision = decision.value

    if failure_type == FailureType.RATE_LIMIT.value:
        return "fail"
    if retry_count >= 4:
        return "fail"
    if decision == DecisionType.RETRY.value or status == PipelineStatus.REJECTED.value:
        return "retry"
    if isinstance(status, str) and status.startswith("FAILED"):
        return "fail"
    return "continue"


def _route_decision(state: AgentState | dict) -> str:
    decision = _sget(state, "decision")
    if decision is None:
        return DecisionType.RETRY.value
    if isinstance(decision, DecisionType):
        return decision.value
    if isinstance(decision, str):
        return decision
    return DecisionType.RETRY.value


def build_graph():
    builder = StateGraph(AgentState)

    # ── Register all nodes ──
    builder.add_node("fetch_issues", fetch_issues_node)
    builder.add_node("select_issue", select_issue_node)
    builder.add_node("setup_repo", setup_repo_node)
    builder.add_node("issue_agent", issue_agent_node)
    builder.add_node("issue_pr", issue_pr_node)
    builder.add_node("integration", integration_node)
    builder.add_node("diff", diff_node)
    builder.add_node("test_generation_agent", test_generation_agent_node)
    builder.add_node("test_simulation_agent", test_simulation_agent_node)
    builder.add_node("llm_validation", llm_validation_node)
    builder.add_node("qa_pr", qa_pr_node)
    builder.add_node("decision", decision_node)
    builder.add_node("qa_normalization", qa_normalization_node)
    builder.add_node("retry", retry_handler_node)
    builder.add_node("merge", merge_node)
    builder.add_node("final_merge_pr", final_merge_pr_node)
    builder.add_node("fail", fail_node)
    builder.add_node("complete", complete_node)

    # ── Phase 1: Linear pre-processing ──
    builder.set_entry_point("fetch_issues")
    builder.add_conditional_edges(
        "fetch_issues",
        _route_or_fail,
        {"continue": "select_issue", "fail": "fail"},
    )
    builder.add_conditional_edges(
        "select_issue",
        _route_or_fail,
        {"continue": "setup_repo", "fail": "fail"},
    )
    builder.add_conditional_edges(
        "setup_repo",
        _route_or_fail,
        {"continue": "issue_agent", "fail": "fail"},
    )

    # ── Phase 2: Issue Agent → Integration → Issue PR → Diff ──
    # issue_agent failure MUST stop pipeline — do NOT continue to integration
    builder.add_conditional_edges(
        "issue_agent",
        _route_recoverable_stage,
        {"continue": "integration", "retry": "retry", "fail": "fail"},
    )
    builder.add_conditional_edges(
        "integration",
        _route_recoverable_stage,
        {"continue": "issue_pr", "retry": "retry", "fail": "fail"},
    )
    builder.add_edge("issue_pr", "diff")
    builder.add_conditional_edges(
        "diff",
        _route_recoverable_stage,
        {"continue": "test_generation_agent", "retry": "retry", "fail": "fail"},
    )
    builder.add_edge("test_generation_agent", "test_simulation_agent")
    builder.add_edge("test_simulation_agent", "llm_validation")

    # ── Phase 3: QA Agent → QA PR → Decision ──
    builder.add_edge("llm_validation", "qa_pr")
    builder.add_edge("qa_pr", "decision")

    # ── Phase 4: Decision routing ──
    builder.add_conditional_edges(
        "decision",
        _route_decision,
        {
            DecisionType.APPROVED.value: "merge",
            DecisionType.RETRY.value: "retry",
            DecisionType.FAIL.value: "fail",
            DecisionType.FAILED.value: "fail",
        }
    )

    # ── Retry loop: retry → check status → qa_normalization → issue_agent ──
    # retry_handler_node may return FAILED_TERMINATED when max exceeded;
    # route that to fail instead of blindly continuing.
    def _route_after_retry(state) -> str:
        status = _sget(state, "status", "")
        if isinstance(status, str) and ("FAILED" in status.upper()):
            return "fail"
        return "continue"

    builder.add_conditional_edges(
        "retry",
        _route_after_retry,
        {"continue": "qa_normalization", "fail": "fail"},
    )
    builder.add_edge("qa_normalization", "issue_agent")

    # ── Terminal edges ──
    builder.add_edge("merge", "final_merge_pr")
    builder.add_edge("final_merge_pr", END)
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_graph()
    print("Graph built successfully.")
