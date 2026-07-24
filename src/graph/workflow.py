# Minimal professional comment: LangGraph configuration for the Single Tool-Calling Agent.
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langfuse.langchain import CallbackHandler
from src.graph.state import AgentState
from src.graph.nodes import agent_node, summarize_conversation_node
from src.graph.tools import get_support_faq, search_catalog, get_order_status, get_customer_history, escalate_to_human, search_legal_tos, check_complaint_eligibility, raise_complaint_ticket
from langchain_core.messages import AIMessage

def should_continue(state: AgentState):
    """Determine whether to invoke tools, summarize, or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM made a tool call, route to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
        
    # Otherwise, check if we need to summarize
    if len(messages) > 6:
        return "summarize"
        
    return END

# Bind tools to ToolNode
tools = [get_support_faq, search_legal_tos, search_catalog, get_order_status, get_customer_history, escalate_to_human, check_complaint_eligibility, raise_complaint_ticket]
tool_node = ToolNode(tools)

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("summarize", summarize_conversation_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END
    }
)

builder.add_edge("tools", "agent")
builder.add_edge("summarize", END)

voice_agent_graph = builder.compile(checkpointer=MemorySaver())

def invoke_graph_with_tracing(initial_state: dict, thread_id: str):
    langfuse_handler = CallbackHandler()
    config = {"callbacks": [langfuse_handler], "configurable": {"thread_id": thread_id}}
    return voice_agent_graph.invoke(initial_state, config=config)

async def stream_graph_with_tracing(initial_state: dict, thread_id: str):
    langfuse_handler = CallbackHandler()
    config = {"callbacks": [langfuse_handler], "configurable": {"thread_id": thread_id}}
    # Using astream_events(version="v2") for better tool vs text separation
    async for event in voice_agent_graph.astream_events(initial_state, config=config, version="v2"):
        yield event