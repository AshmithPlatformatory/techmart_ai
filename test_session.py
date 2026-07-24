import asyncio
import uuid
from langchain_core.messages import HumanMessage
from src.graph.workflow import stream_graph_with_tracing, voice_agent_graph
from src.db.clickhouse_client import get_client

async def main():
    session_id = uuid.uuid4().hex
    phone_number = "9632492407"
    
    # Lookup customer ID
    client = get_client()
    res = client.query("SELECT customer_id, name, loyalty_tier FROM customers WHERE phone = %(ph)s", parameters={"ph": phone_number})
    if res.result_rows:
        customer_id = res.result_rows[0][0]
        name = res.result_rows[0][1]
        loyalty_tier = res.result_rows[0][2]
    else:
        customer_id = "CUST12345"
        name = "Test User"
        loyalty_tier = "Standard"

    customer_profile = {
        "customer_id": customer_id,
        "name": name,
        "phone": phone_number,
        "loyalty_tier": loyalty_tier,
        "detected_language": "en-IN"
    }

    # Initial full state push
    state_delta = {
        "messages": [],
        "customer_profile": customer_profile,
        "session_id": session_id,
        "ticket_id": uuid.uuid4().hex,
        "handoff_status": "None",
        "user_emotion": "neutral",
        "summary": ""
    }

    questions = [
        "Hi can you recommend some good phones in the range of 50,000 to 60000 ?",
        "Hi how are you",
        "Can you tell me what are the ingredients needed to make vanilla cake ?",
        "Make a list of things i have ordered before",
        "What are the indivual ratings or things i have ordered before",
        "What were their proces ?",
        "Can you tell me some top rated, mid range laptops you have ?",
        "what is the return policy on the first one ?",
        "Hey can you tell the prices of phones you listed that time ?"
    ]

    print(f"==================================================")
    print(f"--- Starting TechMart AI Voice Agent Test ---")
    print(f"Session ID: {session_id}")
    print(f"Customer: {name} ({phone_number})")
    print(f"==================================================\n")

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/9] USER: {q}")
        state_delta["messages"] = [HumanMessage(content=q)]
        
        print("AGENT: ", end="", flush=True)
        
        # Stream the graph execution
        async for event in stream_graph_with_tracing(state_delta, session_id):
            if event["event"] == "on_chat_model_stream":
                if event.get("metadata", {}).get("langgraph_node") == "agent":
                    chunk = event["data"]["chunk"]
                    if chunk.content and isinstance(chunk.content, str):
                        print(chunk.content.replace("*", "").replace("#", ""), end="", flush=True)
                        
            elif event["event"] == "on_tool_start":
                # In the new architecture, Tool Calling represents our "Intent"
                tool_name = event["name"]
                tool_args = event["data"].get("input", {})
                print(f"\n   [INTENT DETECTED: Triggering '{tool_name}' with args {tool_args}]\n   ", end="", flush=True)
        
        print() # Newline after agent finishes speaking
        
        # After the first turn, we only need to pass the new HumanMessage in the delta, 
        # LangGraph's MemorySaver handles the rest!
        state_delta = {}

        # Let's peek into LangGraph's real persistent memory state
        current_graph_state = voice_agent_graph.get_state({"configurable": {"thread_id": session_id}}).values
        if current_graph_state.get("summary"):
            print(f"   [MEMORY TRACKER - Sliding Window Summary]: {current_graph_state.get('summary')}")

if __name__ == "__main__":
    asyncio.run(main())
