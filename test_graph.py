import asyncio
from langchain_core.messages import HumanMessage
from src.graph.workflow import voice_agent_graph

async def main():
    print("Initializing LangGraph Rigorous Test...")
    
    # We test 3 extreme real-world scenarios:
    # 1. Complex hierarchical catalog search (Brand + Tech Spec + Budget)
    # 2. Support lookup (Checking if vector search fetches the exact policy)
    # 3. Multi-Intent Routing (Checking if it fetches order history AND catalog simultaneously)
    test_cases = [
        "I'm looking for an HP laptop with GDDR6 RAM that costs under 200000.",
        "What is your return policy if the item I received is damaged?",
        "Can you check the status of my recent orders? Also, what gaming consoles do you have in stock?"
    ]

    # Mock customer profile injected by the adapter in production
    mock_customer = {
        "customer_id": "CUST-001",
        "name": "Ashmith",
        "phone": "9632492407",
        "loyalty_tier": "Gold"
    }

    for idx, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {idx}: {query}")
        print(f"{'='*60}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "customer_profile": mock_customer,
            "session_id": f"test_session_{idx}"
        }
        
        # Checkpointing requires a unique thread_id per conversation thread
        config = {"configurable": {"thread_id": f"test_thread_{idx}"}}
        
        try:
            # Execute the LangGraph
            result = await voice_agent_graph.ainvoke(initial_state, config=config)
            
            # Extract outputs
            intents = result.get('active_intents', [])
            handoff = result.get('handoff_status', 'None')
            final_message = result['messages'][-1].content
            
            print(f"-> INTENTS ROUTED: {intents}")
            print(f"-> HANDOFF STATUS: {handoff}\n")
            print(f"-> SYNTHESIZED RESPONSE:\n{final_message}\n")
            
        except Exception as e:
            print(f"TEST FAILED with Error: {e}")

if __name__ == "__main__":
    # Windows asyncio fix for ProactorEventLoop
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
