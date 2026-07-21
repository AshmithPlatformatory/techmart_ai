import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    rag_contexts: Annotated[List[str], operator.add]
    customer_profile: Dict[str, Any]
    active_intents: List[str]
    english_query: str
    session_id: str
    ticket_id: str
    handoff_status: str
    user_emotion: str
    structured_memory: Dict[str, Any]