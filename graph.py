from langgraph.graph import StateGraph, END
from nodes import retrieve, generate, evaluate, router
from state import State
graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    router,
    {
        "end": END,
        "retry": "retrieve"
    }
)

app = graph.compile()