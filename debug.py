import asyncio
import uuid
from langchain_core.messages import HumanMessage

async def main():
    try:
        from src.graph.workflow import stream_graph_with_tracing
        
        session_id = uuid.uuid4().hex
        
        customer_profile = {
            "customer_id": "CUST-0221",
            "name": "Arun Kumar",
            "phone": "1234567890",
            "loyalty_tier": "Gold",
            "detected_language": "en-IN"
        }

        state_delta = {
            "messages": [],
            "customer_profile": customer_profile,
            "session_id": session_id,
            "ticket_id": uuid.uuid4().hex,
            "handoff_status": "None",
            "user_emotion": "neutral",
            "summary": ""
        }

        q = "Can you check my previous order status?"
        state_delta["messages"] = [HumanMessage(content=q)]
        
        print("AGENT: ", end="", flush=True)
        async for event in stream_graph_with_tracing(state_delta, session_id):
            if event["event"] == "on_chat_model_stream":
                if event.get("metadata", {}).get("langgraph_node") == "agent":
                    chunk = event["data"]["chunk"]
                    if chunk.content and isinstance(chunk.content, str):
                        print(chunk.content.replace("*", "").replace("#", ""), end="", flush=True)
            elif event["event"] == "on_tool_start":
                tool_name = event["name"]
                tool_args = event["data"].get("input", {})
                print(f"\n   [INTENT DETECTED: Triggering '{tool_name}' with args {tool_args}]\n   ", end="", flush=True)
            
        print("\n")
        
    except Exception as e:
        import traceback
        print("\nCRITICAL ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
