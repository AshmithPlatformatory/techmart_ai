import uuid
import datetime
import asyncio
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from src.graph.state import AgentState
from src.graph.tools import get_support_context, get_catalog_context, get_order_context, get_history_context
from src.db.clickhouse_client import get_client
from src.model_loader import get_sentence_transformer


_router_llm = None
_synth_llm = None

def get_router_llm():
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return _router_llm

def get_synth_llm():
    global _synth_llm
    if _synth_llm is None:
        _synth_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return _synth_llm

class RouterOutput(BaseModel):
    intents: list[str] = Field(description="List of intents: support, orders, catalog, history")
    handoff_action: str = Field(description="One of: 'none', 'offer_handoff', 'accept_handoff', 'reject_handoff'", default="none")

async def router_node(state: AgentState) -> dict:
    recent_msgs = state["messages"][-4:]
    conversation_history = "\n".join([f"{'User' if m.type == 'human' else 'Agent'}: {m.content}" for m in recent_msgs])
    english_query = state["messages"][-1].content
    handoff_status = state.get("handoff_status", "None")

    prompt = f"""Analyze the conversation history and determine required datasets.
Current handoff status: {handoff_status}.

Intents Definitions:
- "support": ONLY if the user asks about general FAQs or policies.
- "orders": ONLY if the user explicitly asks about their past purchases, order status, or tracking.
- "catalog": ONLY if the user asks for product recommendations, prices, inventory, or laptops/backups.
- "history": ONLY if the user asks about previous calls or support tickets.
If unsure, return an empty list of intents.

If the user wants to perform ANY WRITE OPERATION (e.g. place a new order, modify an account, cancel an order), this agent CANNOT do it. You MUST output handoff_action='offer_handoff'.
NOTE: Checking, fetching, viewing, or asking for order history, catalog, or account details are READ operations. DO NOT trigger a handoff for these.
If handoff was 'Offered' and user agrees, output 'accept_handoff'. If they decline, output 'reject_handoff'.

Conversation History:
{conversation_history}"""
    structured_llm = get_router_llm().with_structured_output(RouterOutput)
    try:
        result = await structured_llm.ainvoke(prompt)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Router LLM Error: {e}")
        # Graceful fallback: return empty active intents and maintain handoff status
        result = RouterOutput(intents=[], handoff_action="none")

    valid = [i for i in result.intents if i in ["support", "orders", "catalog", "history"]]

    new_handoff_status = "None"
    if result.handoff_action == "offer_handoff":
        new_handoff_status = "Offered"
    elif result.handoff_action == "accept_handoff":
        new_handoff_status = "Accepted"
    elif result.handoff_action == "reject_handoff":
        new_handoff_status = "None"

    return {"active_intents": valid, "handoff_status": new_handoff_status, "english_query": english_query}

def support_worker(state: AgentState) -> dict:
    q = state.get("english_query", state["messages"][-1].content)
    data = get_support_context(q)
    return {"rag_contexts": [f"[SUPPORT]\n{data}"]}

def order_worker(state: AgentState) -> dict:
    cid = state.get("customer_profile", {}).get("customer_id", "")
    if not cid:
        return {"rag_contexts": ["[ORDERS]\nNo customer ID."]}
    data = get_order_context(cid)
    return {"rag_contexts": [f"[ORDERS]\n{data}"]}

def catalog_worker(state: AgentState) -> dict:
    q = state.get("english_query", state["messages"][-1].content)
    data = get_catalog_context(q)
    return {"rag_contexts": [f"[CATALOG]\n{data}"]}

def history_worker(state: AgentState) -> dict:
    cid = state.get("customer_profile", {}).get("customer_id", "")
    if not cid:
        return {"rag_contexts": ["[HISTORY]\nNo customer ID."]}
    data = get_history_context(cid)
    return {"rag_contexts": [f"[HISTORY]\n{data}"]}

async def synthesizer_node(state: AgentState) -> dict:
    sys_prompt = "You are a TechMart voice agent. Use the context to answer. Speak conversationally. No markdown."
    
    handoff_status = state.get("handoff_status", "None")
    if handoff_status == "Offered":
        sys_prompt += "\nINSTRUCTION: The user wants an action you cannot perform. Offer to transfer them to a human expert."
    elif handoff_status == "Accepted":
        sys_prompt += "\nINSTRUCTION: The user accepted the handoff. Tell them you are transferring them to a human expert now."
    
    customer_info = state.get("customer_profile", {})
    if customer_info:
        sys_prompt += f"\nYou are speaking with {customer_info.get('name', 'a customer')} (Phone: {customer_info.get('phone', '')}). They are a {customer_info.get('loyalty_tier', '')} tier customer. You ALREADY have their information. NEVER ask them to verify or provide their phone number or email."

    user_emotion = state.get("user_emotion", "")
    if user_emotion:
        sys_prompt += f"\nThe user currently sounds: {user_emotion}. Adjust your tone and empathy accordingly to match or soothe this emotion."

    sys_prompt += (
        f"\nCRITICAL RULE: You are a READ-ONLY agent. You CANNOT place orders, cancel orders, or modify any account data. If the user asks you to perform a write operation, you MUST refuse, clarify you can only provide information, and offer to transfer them to a human. NEVER hallucinate order IDs or confirmations."
        f"\nDo NOT use ANY markdown formatting (no asterisks `**`, no bolding, no lists). Output plain text only."
    )

    rag_contexts = state.get("rag_contexts", [])
    if rag_contexts:
        sys_prompt += "\n\n=== KNOWLEDGE CONTEXT ===\n" + "\n".join(rag_contexts)
        sys_prompt += "\n\nCRITICAL RULE: Only use the KNOWLEDGE CONTEXT if it directly answers the user's latest query. DO NOT mention past orders or unrelated facts from the context unless explicitly requested."

    msgs = [SystemMessage(content=sys_prompt)] + state["messages"]
    resp = await get_synth_llm().ainvoke(msgs)
    return {"messages": [resp]}

async def write_call_ticket(state: AgentState) -> None:
    """Write a call ticket to ClickHouse.

    This is NOT a LangGraph node — it is called as a fire-and-forget
    asyncio.create_task() from adapter.py after the graph finishes
    streaming. This keeps it off the user-response critical path.
    """
    try:
        if len(state["messages"]) >= 2:
            turn_text = f"User: {state['messages'][-2].content}\nAgent: {state['messages'][-1].content}"
        else:
            turn_text = f"Agent: {state['messages'][-1].content}"

        if len(turn_text) > 4000:
            turn_text = "..." + turn_text[-4000:]

        prompt = f"Summarize this interaction briefly:\n{turn_text}"
        summary_msg = await get_synth_llm().ainvoke(prompt)
        summary = summary_msg.content
        def _sync_blocking_work(summary_text, ticket, sid, customer_id, ph, start, end, status, transcript):
            emb = get_sentence_transformer().encode(summary_text).tolist()
            client = get_client()
            client.insert("call_tickets", [[
                ticket, sid, customer_id, ph, start, end, status, transcript, summary_text, emb
            ]], column_names=["ticket_id", "session_id", "customer_id", "caller_phone", "call_start_time", "call_end_time", "call_status", "full_transcript", "summary", "summary_embedding"])

        ticket_id = state.get("ticket_id", f"TKT-{uuid.uuid4().hex[:8].upper()}")
        session_id = state.get("session_id", "UNKNOWN")
        cid = state.get("customer_profile", {}).get("customer_id", "")
        phone = state.get("customer_profile", {}).get("phone", "")
        now = datetime.datetime.now()

        await asyncio.to_thread(
            _sync_blocking_work, 
            summary, ticket_id, session_id, cid, phone, now, now, "Completed", turn_text
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"write_call_ticket failed (non-fatal): {e}")