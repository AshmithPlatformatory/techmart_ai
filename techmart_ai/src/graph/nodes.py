import uuid
import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from src.graph.state import AgentState
from src.graph.tools import get_support_context, get_catalog_context, get_order_context, get_history_context
from src.db.clickhouse_client import get_client
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
router_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
synth_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

class RouterOutput(BaseModel):
    intents: list[str] = Field(description="List of intents: support, orders, catalog, history")
    handoff_action: str = Field(description="One of: 'none', 'offer_handoff', 'accept_handoff', 'reject_handoff'", default="none")

def router_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    handoff_status = state.get("handoff_status", "None")

    prompt = f"""Analyze the query and determine required datasets from: support, orders, catalog, history. 
Current handoff status: {handoff_status}.
If the user wants to modify the database (e.g. cancel order, make order), output handoff_action='offer_handoff'.
If handoff was 'Offered' and user agrees, output 'accept_handoff'. If they decline, output 'reject_handoff'.
Query: {last_msg}"""
    structured_llm = router_llm.with_structured_output(RouterOutput)
    result = structured_llm.invoke(prompt)
    valid = [i for i in result.intents if i in ["support", "orders", "catalog", "history"]]

    new_handoff_status = handoff_status
    if result.handoff_action == "offer_handoff":
        new_handoff_status = "Offered"
    elif result.handoff_action == "accept_handoff":
        new_handoff_status = "Accepted"
    elif result.handoff_action == "reject_handoff":
        new_handoff_status = "Rejected"

    return {"active_intents": valid, "handoff_status": new_handoff_status}

def support_worker(state: AgentState) -> dict:
    q = state["messages"][-1].content
    data = get_support_context(q)
    return {"retrieved_context": [f"[SUPPORT]\n{data}"]}

def order_worker(state: AgentState) -> dict:
    cid = state.get("customer_profile", {}).get("customer_id", "")
    if not cid:
        return {"retrieved_context": ["[ORDERS]\nNo customer ID."]}
    data = get_order_context(cid)
    return {"retrieved_context": [f"[ORDERS]\n{data}"]}

def catalog_worker(state: AgentState) -> dict:
    q = state["messages"][-1].content
    data = get_catalog_context(q)
    return {"retrieved_context": [f"[CATALOG]\n{data}"]}

def history_worker(state: AgentState) -> dict:
    cid = state.get("customer_profile", {}).get("customer_id", "")
    if not cid:
        return {"retrieved_context": ["[HISTORY]\nNo customer ID."]}
    data = get_history_context(cid)
    return {"retrieved_context": [f"[HISTORY]\n{data}"]}

def synthesizer_node(state: AgentState) -> dict:
    handoff_status = state.get("handoff_status", "None")
    if handoff_status == "Offered":
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="I can't conduct this operation, would you like to connect to a human expert?")]}
    elif handoff_status == "Accepted":
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="Transferring you to a human expert.")]}

    context_str = "\n".join(state.get("retrieved_context", []))
    sys_prompt = "You are a TechMart voice agent. Use the context to answer. Speak conversationally. No markdown."

    user_emotion = state.get("user_emotion", "")
    if user_emotion:
        sys_prompt += f"\nThe user currently sounds: {user_emotion}. Adjust your tone and empathy accordingly to match or soothe this emotion."

    if context_str:
        sys_prompt += f"\nContext:\n{context_str}"

    msgs = [SystemMessage(content=sys_prompt)] + state["messages"]
    resp = synth_llm.invoke(msgs)
    return {"messages": [resp]}

def sink_node(state: AgentState) -> dict:
    if len(state["messages"]) >= 2:
        turn_text = f"User: {state['messages'][-2].content}\nAgent: {state['messages'][-1].content}"
    else:
        turn_text = f"Agent: {state['messages'][-1].content}"

    prompt = f"Summarize this interaction briefly:\n{turn_text}"
    summary = router_llm.invoke(prompt).content
    embedding = model.encode(summary).tolist()

    client = get_client()
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    session_id = state.get("session_id", "UNKNOWN")
    cid = state.get("customer_profile", {}).get("customer_id", "")
    phone = state.get("customer_profile", {}).get("phone", "")
    now = datetime.datetime.now()

    client.insert("call_tickets", [[
        ticket_id, session_id, cid, phone, now, now, "Completed", turn_text, summary, embedding
    ]], column_names=["ticket_id", "session_id", "customer_id", "caller_phone", "call_start_time", "call_end_time", "call_status", "full_transcript", "summary", "summary_embedding"])

    return {}