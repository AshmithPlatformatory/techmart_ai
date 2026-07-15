"""
Compiles the LangGraph workflow and configures the execution graph.
Integrates Langfuse observability using the updated v3.x Langchain CallbackHandler.
Maps the execution nodes and conditional routing logic.
"""
from langgraph.graph import StateGraph, END, START
from langfuse.langchain import CallbackHandler
from src.graph.state import AgentState
from src.graph.nodes import (
    router_node, support_worker, order_worker, catalog_worker,
    history_worker, synthesizer_node
)

def route_workers(state: AgentState):
    intents = state.get("active_intents", [])
    if not intents:
        return ["synthesizer"]
    routes = []
    if "support" in intents: routes.append("support_worker")
    if "orders" in intents: routes.append("order_worker")
    if "catalog" in intents: routes.append("catalog_worker")
    if "history" in intents: routes.append("history_worker")
    if not routes:
        return ["synthesizer"]
    return routes

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("support_worker", support_worker)
builder.add_node("order_worker", order_worker)
builder.add_node("catalog_worker", catalog_worker)
builder.add_node("history_worker", history_worker)
builder.add_node("synthesizer", synthesizer_node)
# sink_node is removed from the graph — it runs as asyncio.create_task()
# in adapter.py after streaming completes, so it never blocks the response.

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_workers, {
    "support_worker": "support_worker",
    "order_worker": "order_worker",
    "catalog_worker": "catalog_worker",
    "history_worker": "history_worker",
    "synthesizer": "synthesizer"
})

builder.add_edge("support_worker", "synthesizer")
builder.add_edge("order_worker", "synthesizer")
builder.add_edge("catalog_worker", "synthesizer")
builder.add_edge("history_worker", "synthesizer")

# Graph ends at synthesizer — no blocking sink step.
builder.add_edge("synthesizer", END)

voice_agent_graph = builder.compile()

def invoke_graph_with_tracing(initial_state: dict):
    langfuse_handler = CallbackHandler()
    config = {"callbacks": [langfuse_handler]}
    return voice_agent_graph.invoke(initial_state, config=config)

async def stream_graph_with_tracing(initial_state: dict):
    langfuse_handler = CallbackHandler()
    config = {"callbacks": [langfuse_handler]}
    async for chunk in voice_agent_graph.astream(initial_state, config=config, stream_mode=["messages", "values"]):
        yield chunk