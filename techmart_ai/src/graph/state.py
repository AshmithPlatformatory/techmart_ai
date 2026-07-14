import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    customer_profile: Dict[str, Any]
    active_intents: List[str]
    detected_language: str
    retrieved_context: Annotated[List[str], operator.add]
    session_id: str
    handoff_status: str
    user_emotion: str