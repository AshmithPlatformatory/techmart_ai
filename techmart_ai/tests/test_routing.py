import pytest
from unittest.mock import patch, MagicMock
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

    @patch("src.graph.nodes.router_llm")
    def test_router_node_parsing(self, mock_router_llm):
        """Test that the router node correctly invokes LLM and filters valid intents."""
        # Mock structured output
        mock_structured_llm = MagicMock()
        mock_router_llm.with_structured_output.return_value = mock_structured_llm
        
        # Scenario 1: Valid intents returned
        mock_structured_llm.invoke.return_value = RouterOutput(intents=["support", "catalog", "invalid"])
        state = {"messages": [HumanMessage(content="I need help with a product")]}
        
        result = router_node(state)
        # Should filter out "invalid"
        assert sorted(result["active_intents"]) == ["catalog", "support"]

        # Scenario 2: Empty intents returned
        mock_structured_llm.invoke.return_value = RouterOutput(intents=["fake"])
        result = router_node(state)
        assert result["active_intents"] == []

    @patch("src.graph.workflow.CallbackHandler")
    @patch("src.graph.nodes.synth_llm")
    @patch("src.graph.nodes.router_llm")
    @patch("src.graph.nodes.model")
    @patch("src.graph.nodes.get_client")
    @patch("src.graph.nodes.get_support_context")
    @patch("src.graph.nodes.get_order_context")
    def test_full_graph_execution_support_and_orders(
        self, mock_get_order, mock_get_support, mock_get_client, mock_model, mock_router, mock_synth, mock_langfuse
    ):
        """Integration test for the entire graph routing execution with mock LLMs and tools."""
        
        # 1. Mock Router LLM
        mock_structured = MagicMock()
        mock_router.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RouterOutput(intents=["support", "orders"])
        mock_router.invoke.return_value = AIMessage(content="Summary")
        
        # 2. Mock RAG Tools
        mock_get_support.return_value = "Support FAQ data"
        mock_get_order.return_value = "Order #123 data"
        
        # 3. Mock Synth LLM
        mock_synth.invoke.return_value = AIMessage(content="Here is your support and order info.")
        
        # 4. Mock Clickhouse and Embeddings
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        mock_clickhouse = MagicMock()
        mock_get_client.return_value = mock_clickhouse

        initial_state = {
            "messages": [HumanMessage(content="Where is my order and what is your return policy?")],
            "customer_profile": {"customer_id": "C100"},
            "session_id": "test_sess_1",
            "active_intents": [],
            "retrieved_context": []
        }

        # Execute Graph
        final_state = voice_agent_graph.invoke(initial_state)

        # Assertions
        assert "retrieved_context" in final_state
        assert len(final_state["retrieved_context"]) == 2
        
        contexts = "".join(final_state["retrieved_context"])
        assert "[SUPPORT]" in contexts
        assert "Support FAQ data" in contexts
        assert "[ORDERS]" in contexts
        assert "Order #123 data" in contexts
        
        assert final_state["messages"][-1].content == "Here is your support and order info."
        
        # Verify Clickhouse Sink was called
        mock_clickhouse.insert.assert_called_once()
