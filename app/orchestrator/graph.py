from langgraph.graph import StateGraph, END
from app.orchestrator.nodes import (
    supervisor_node,
    researcher_node,
    synthesizer_node,
    critic_node,
)

builder = StateGraph(dict)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("supervisor")

# Core processing flow
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", "critic")
builder.add_edge("critic", "supervisor")


# =========================================
# Supervisor Controlled Routing
# =========================================
def route_from_supervisor(state):

    # 🔥 FIRST RUN (no draft yet)
    if state.get("draft_answer") is None:
        return "retry"

    # 🔴 HITL
    if state.get("requires_human_review"):
        return "end"

    last_trace = state.get("trace", [])[-1] if state.get("trace") else {}

    # 🔁 Retry
    if last_trace.get("decision") == "retry":
        return "retry"

    # ✅ Finalize
    return "end"


builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "retry": "researcher",
        "end": END,
    },
)

graph = builder.compile()