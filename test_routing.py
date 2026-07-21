import asyncio
from langchain_core.messages import HumanMessage
from src.graph.workflow import voice_agent_graph

async def main():
    print("Initializing LangGraph Rigorous Routing Test...\n")
    
    # We test every possible routing branch and edge case defined in the router_node
    test_cases = [
        # 1. Pure Support Intent
        "What are your store policies regarding returning a broken item?",
        
        # 2. Pure Orders Intent
        "Can you track my most recent purchase?",
        
        # 3. Pure Catalog Intent
        "Do you have any Apple laptops with at least 16GB RAM in stock?",
        
        # 4. Pure History Intent
        "Can you look up my past support tickets?",
        
        # 5. Multi-Intent (Simultaneous routing)
        "What is the status of my order, and also do you sell 4K monitors?",
        
        # 6. Out-of-Domain / Chitchat (Should route to none/synthesizer)
        "Hey! How is the weather today? What is 2 plus 2?",
        
        # 7. Handoff: Write Operation (Should trigger offer_handoff)
        "I accidentally ordered the wrong item, I need to cancel my order right now.",
        
        # 8. Handoff: Explicit Human Demand (Should trigger accept_handoff)
        "This is frustrating, let me speak to a real human manager immediately.",
        
        # 9. Handoff: Explicit Rejection (Should trigger reject_handoff)
        "No, please do not transfer me to a human, I just want to know if you can do it."
    ]

    mock_customer = {
        "customer_id": "CUST-001",
        "name": "Ashmith",
        "phone": "9632492407",
        "loyalty_tier": "Gold"
    }

    for idx, query in enumerate(test_cases, 1):
        print(f"{'='*70}")
        print(f"TEST {idx}: {query}")
        print(f"{'='*70}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "customer_profile": mock_customer,
            "session_id": f"routing_test_session_{idx}"
        }
        
        config = {"configurable": {"thread_id": f"routing_test_thread_{idx}"}}
        
        try:
            result = await voice_agent_graph.ainvoke(initial_state, config=config)
            
            intents = result.get('active_intents', [])
            handoff = result.get('handoff_status', 'None')
            final_message = result['messages'][-1].content
            
            print(f"-> INTENTS ROUTED : {intents}")
            print(f"-> HANDOFF STATUS : {handoff}")
            print(f"-> RESPONSE       :\n{final_message}\n")
            
        except Exception as e:
            print(f"TEST FAILED with Error: {e}")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
