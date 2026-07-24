# Minimal professional comment: State definitions for LangGraph single agent architecture.
import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    customer_profile: Dict[str, Any]
    session_id: str
    ticket_id: str
    handoff_status: str
    user_emotion: str
    summary: str