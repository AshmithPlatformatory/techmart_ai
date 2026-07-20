import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.graph.state import AgentState
from src.graph.workflow import route_workers, voice_agent_graph
from src.graph.nodes import router_node, RouterOutput

class TestLangGraphRouting:

    def test_route_workers_empty_intents(self):
        """Test that empty intents route directly to the synthesizer."""
        state = {"active_intents": []}
        routes = route_workers(state)
        assert routes == ["synthesizer"]

    @pytest.mark.parametrize("intents,expected_routes", [
        (["support"], ["support_worker"]),
        (["orders"], ["order_worker"]),
        (["catalog"], ["catalog_worker"]),
        (["history"], ["history_worker"]),
        (["support", "catalog"], ["support_worker", "catalog_worker"]),
        (["orders", "history"], ["order_worker", "history_worker"]),
        (["unknown_intent"], ["synthesizer"]),
        (["support", "invalid_intent"], ["support_worker"]),
    ])
    def test_route_workers_valid_intents(self, intents, expected_routes):
        """Test that valid intents correctly map to their worker nodes."""
        state = {"active_intents": intents}
        routes = route_workers(state)
        assert sorted(routes) == sorted(expected_routes)

    @pytest.mark.asyncio
    @patch("src.graph.nodes.get_router_llm")
    async def test_router_node_parsing(self, mock_get_router_llm):
        """Test that the router node correctly invokes LLM and filters valid intents."""
        mock_router_llm = MagicMock()
        mock_get_router_llm.return_value = mock_router_llm
        mock_structured_llm = MagicMock()
        mock_router_llm.with_structured_output.return_value = mock_structured_llm
        
        # Scenario 1: Valid intents returned
        mock_structured_llm.ainvoke = AsyncMock(return_value=RouterOutput(standalone_query="mock query", intents=["support", "catalog", "invalid"], handoff_action="none"))
        state = {"messages": [HumanMessage(content="I need help with a product")], "handoff_status": "None"}
        
        result = await router_node(state)
        # Should filter out "invalid"
        assert sorted(result["active_intents"]) == ["catalog", "support"]

        # Scenario 2: Empty intents returned
        mock_structured_llm.ainvoke = AsyncMock(return_value=RouterOutput(standalone_query="mock query", intents=["fake"], handoff_action="none"))
        result = await router_node(state)
        assert result["active_intents"] == []

    @patch("src.graph.workflow.CallbackHandler")
    @patch("src.graph.nodes.get_synth_llm")
    @patch("src.graph.nodes.get_router_llm")
    @patch("src.graph.nodes.get_sentence_transformer")
    @patch("src.graph.nodes.get_client")
    @patch("src.graph.nodes.get_support_context")
    @patch("src.graph.nodes.get_order_context")
    async def test_full_graph_execution_support_and_orders(
        self, mock_get_order, mock_get_support, mock_get_client, mock_model, mock_router_getter, mock_synth_getter, mock_langfuse
    ):
        """Integration test for the entire graph routing execution with mock LLMs and tools."""
        
        # 1. Mock Router LLM
        mock_router_llm = MagicMock()
        mock_router_getter.return_value = mock_router_llm
        mock_structured = MagicMock()
        mock_router_llm.with_structured_output.return_value = mock_structured
        mock_structured.ainvoke = AsyncMock(return_value=RouterOutput(standalone_query="mock query", intents=["support", "orders"], handoff_action="none"))
        mock_router_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary"))
        
        # 2. Mock RAG Tools
        mock_get_support.return_value = "Support FAQ data"
        mock_get_order.return_value = "Order #123 data"
        
        # 3. Mock Synth LLM
        mock_synth_llm = MagicMock()
        mock_synth_getter.return_value = mock_synth_llm
        mock_synth_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here is your support and order info."))
        
        # 4. Mock Clickhouse and Embeddings
        mock_model.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        mock_clickhouse = MagicMock()
        mock_get_client.return_value = mock_clickhouse

        initial_state = {
            "messages": [HumanMessage(content="Where is my order and what is your return policy?")],
            "customer_profile": {"customer_id": "C100"},
            "session_id": "test_sess_1",
            "active_intents": [],
            "rag_contexts": [],
            "ticket_id": "TKT-123",
            "handoff_status": "None",
            "user_emotion": "neutral",
            "english_query": "Where is my order and what is your return policy?"
        }

        # Execute Graph (async — all nodes are async def)
        import uuid
        config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        final_state = await voice_agent_graph.ainvoke(initial_state, config=config)

        # --- Assertions ---
        # The graph state must always have messages
        assert "messages" in final_state
        assert len(final_state["messages"]) >= 2  # original HumanMessage + synthesizer AIMessage

        # Workers store their output in state["rag_contexts"] (via operator.add reducer).
        # The synthesizer reads rag_contexts to build its system prompt internally —
        # it does NOT add SystemMessage objects to state["messages"].
        rag = final_state.get("rag_contexts", [])
        assert any("[SUPPORT]" in ctx for ctx in rag), f"Expected [SUPPORT] in rag_contexts, got: {rag}"
        assert any("Support FAQ data" in ctx for ctx in rag), f"Support data missing from rag_contexts: {rag}"
        assert any("[ORDERS]" in ctx for ctx in rag), f"Expected [ORDERS] in rag_contexts, got: {rag}"
        assert any("Order #123 data" in ctx for ctx in rag), f"Order data missing from rag_contexts: {rag}"
        # Verify get_order_context was called with both customer_id AND english_query (new 2-arg signature)
        mock_get_order.assert_called_once()
        call_args = mock_get_order.call_args[0]
        assert call_args[0] == "C100", f"Expected customer_id 'C100', got: {call_args[0]}"
        assert len(call_args) == 2, f"Expected 2 args (customer_id, query), got: {len(call_args)}"

        # The synthesizer's AIMessage is the last message in state["messages"]
        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage), f"Last message should be AIMessage, got: {type(last_msg)}"
        assert last_msg.content == "Here is your support and order info."
