from langgraph.graph import StateGraph, END
from app.orchestrator.nodes import (
    supervisor_node,
    researcher_node,
    synthesizer_node,
    critic_node,
    evaluator_node,
)
from app.orchestrator.state import MASISState

# ✅ TypedDict state — supports both dict-style access (state["key"]) in nodes
#    and Pydantic validation at the API boundary via MASISInput.to_state()
builder = StateGraph(MASISState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("critic", critic_node)
builder.add_node("evaluator", evaluator_node)

builder.set_entry_point("supervisor")

# Core processing flow
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", "critic")
builder.add_edge("critic", "evaluator")
builder.add_edge("evaluator", "supervisor")


# =========================================
# Supervisor Controlled Routing
# =========================================
def route_from_supervisor(state: MASISState):

    # 🔴 HITL stops graph
    if state.get("requires_human_review"):
        return "end"

    last_trace = state.get("trace", [])[-1] if state.get("trace") else {}

    # 🔁 Explicit retry only if supervisor decided retry
    if last_trace.get("decision") == "retry":
        return "retry"

    # 🆕 First run: no answer yet, start the pipeline
    if state.get("draft_answer") is None:
        return "first_run"

    # ✅ Finalize
    return "end"


builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "retry": "researcher",
        "first_run": "researcher",  # Semantically distinct from retry
        "end": END,
    },
)

graph = builder.compile()
