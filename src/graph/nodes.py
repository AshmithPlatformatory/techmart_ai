# Minimal professional comment: Single LLM agent node and background summarization node for LangGraph.
import datetime
import asyncio
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_groq import ChatGroq
from src.graph.state import AgentState
from src.graph.tools import get_support_faq, search_catalog, get_order_status, get_customer_history, escalate_to_human, search_legal_tos, check_complaint_eligibility, raise_complaint_ticket
from src.db.clickhouse_client import get_client
from src.model_loader import get_sentence_transformer
from pydantic import BaseModel, Field

_agent_llm = None
_summary_llm = None

def get_agent_llm():
    global _agent_llm
    if _agent_llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, streaming=True)
        tools = [get_support_faq, search_legal_tos, search_catalog, get_order_status, get_customer_history, escalate_to_human, check_complaint_eligibility, raise_complaint_ticket]
        _agent_llm = llm.bind_tools(tools)
    return _agent_llm

def get_summary_llm():
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return _summary_llm

async def agent_node(state: AgentState) -> dict:
    # If the last message was the escalate tool, trigger handoff immediately
    if state["messages"] and getattr(state["messages"][-1], "name", "") == "escalate_to_human":
        return {"messages": [AIMessage(content="Transferring you now...")], "handoff_status": "Accepted"}

    target_code = state.get("customer_profile", {}).get("detected_language", "en-IN")
    lang_map = {
        "hi-IN": "Hindi", "kn-IN": "Kannada", "en-IN": "English", "ml-IN": "Malayalam",
        "ta-IN": "Tamil", "te-IN": "Telugu", "bn-IN": "Bengali", "gu-IN": "Gujarati",
        "mr-IN": "Marathi", "pa-IN": "Punjabi", "od-IN": "Odia"
    }
    target_lang_name = lang_map.get(target_code, "English")

    sys_prompt = (
        "Role: TechMart AI voice assistant. Empathetic, concise, professional.\n"
        "RULES:\n"
        "- Voice Audio: 1-3 short sentences. No lists. No formatting (*, bold, md).\n"
        "- Truth: KNOWLEDGE CONTEXT is absolute. State facts confidently. Never apologize for 'lack of access'.\n"
        "- Scope: Answer about orders, products, policies. CANNOT modify accounts, cancel, or process transactions.\n"
        "- OOD: Politely refuse non-TechMart topics in 1 sentence.\n"
        "- Chitchat: Warm 1-sentence reply, then redirect to TechMart.\n"
        "- Multi-Intent: CRITICAL: If the user asks for multiple distinct items or topics (e.g., a phone AND a camera, or a refund policy AND a shipping policy), you MUST execute multiple parallel tool calls for each discrete request.\n"
        f"- Lang: CRITICAL: Output ONLY in {target_lang_name}.\n"
        "- Numbers: CRITICAL: Format prices with commas (159,990).\n"
    )
    
    handoff_status = state.get("handoff_status", "None")
    if handoff_status == "Accepted":
        return {"messages": [AIMessage(content="Transferring you now...")]}
    elif handoff_status == "Offered":
        sys_prompt += "\nSTATE: HANDOFF (Offer transfer to human for unsupported actions)."
    elif handoff_status == "Rejected":
        sys_prompt += "\nSTATE: REJECTED (User declined transfer. Ask how else to help)."

    customer_info = state.get("customer_profile", {})
    if customer_info:
        sys_prompt += f"\nUSER: {customer_info.get('name', '')} | ID: {customer_info.get('customer_id', '')} | Phone: {customer_info.get('phone', '')} | Tier: {customer_info.get('loyalty_tier', '')}. (Do not ask to verify identity)."

    sys_prompt += "\nIf they want to file a complaint, ALWAYS use check_complaint_eligibility first. DO NOT write the ticket until they verbally confirm it."

    summary = state.get("summary", "")
    if summary:
        sys_prompt += f"\n\nPAST CONVERSATION SUMMARY:\n{summary}"
        
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    
    resp = await get_agent_llm().ainvoke(messages)
    
    if isinstance(resp.content, str) and resp.content:
        resp.content = resp.content.replace("*", "").replace("#", "")
        
    return {"messages": [resp]}

async def summarize_conversation_node(state: AgentState) -> dict:
    """Background node to summarize old messages when sliding window exceeds 6 messages."""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    if len(messages) <= 6:
        return {}
        
    messages_to_summarize = messages[:-6]
    
    summary_prompt = (
        f"This is the existing summary of the conversation:\n{summary}\n\n"
        "Extend the summary by incorporating the following new messages:\n"
    )
    for m in messages_to_summarize:
        if isinstance(m, HumanMessage):
            summary_prompt += f"User: {m.content}\n"
        elif isinstance(m, AIMessage) and m.content:
            summary_prompt += f"AI: {m.content}\n"
            
    summary_prompt += "\nReturn ONLY the updated bulleted summary."
    
    new_summary_msg = await get_summary_llm().ainvoke(summary_prompt)
    delete_actions = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {"summary": new_summary_msg.content, "messages": delete_actions}

class TicketOutput(BaseModel):
    summary: str = Field(description="A single, concise sentence summarizing the customer support phone call.")

async def write_call_ticket(ticket_id: str, session_id: str, customer_profile: dict, full_transcript: str) -> None:
    """Write a call ticket to ClickHouse at the end of the call."""
    try:
        if len(full_transcript) > 20000:
            full_transcript = "..." + full_transcript[-20000:]
        prompt = f"Summarize the following customer support phone call for a CRM ticket log.\nFull Transcript:\n{full_transcript}"
        structured_llm = get_summary_llm().with_structured_output(TicketOutput)
        summary_msg = await structured_llm.ainvoke(prompt)
        summary_text = summary_msg.summary
        
        def _sync_blocking_work(summary_t, ticket, sid, customer_id, ph, start, end, status, transcript):
            emb = get_sentence_transformer().encode(summary_t).tolist()
            client = get_client()
            client.insert("call_tickets", [[
                ticket, sid, customer_id, ph, start, end, status, transcript, summary_t, emb
            ]], column_names=["ticket_id", "session_id", "customer_id", "caller_phone", "call_start_time", "call_end_time", "call_status", "full_transcript", "summary", "summary_embedding"])

        cid = customer_profile.get("customer_id", "")
        phone = customer_profile.get("phone", "")
        now = datetime.datetime.now()
        await asyncio.to_thread(_sync_blocking_work, summary_text, ticket_id, session_id, cid, phone, now, now, "Completed", full_transcript)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"write_call_ticket failed (non-fatal): {e}")
