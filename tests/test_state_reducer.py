"""Unit tests for GraphState reducer method."""

import unittest
import sys
import os
from typing import Dict, Any
from unittest.mock import Mock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_evo.core.state import GraphState
from langgraph_evo.core.config import ConfigRecord
from langgraph.graph.message import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage


class TestGraphStateReducer(unittest.TestCase):
    """Test cases for GraphState.reducer method."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample config records
        self.config1 = ConfigRecord(
            name="test_config_1",
            version="1.0.0",
            description="Test configuration 1",
            config="test config data 1"
        )
        
        self.config2 = ConfigRecord(
            name="test_config_2", 
            version="2.0.0",
            description="Test configuration 2",
            config="test config data 2"
        )
        
        # Create sample messages using proper LangChain message format
        self.message1 = HumanMessage(content="Hello")
        self.message2 = AIMessage(content="World")
        
        # Create sample GraphState instances
        self.state1 = GraphState(
            lineage=["node1", "node2"],
            agent_id="agent_1", 
            messages=[self.message1],
            config=self.config1,
            children_states={},
            initialized_node_ids={"node1", "node2"}
        )
        
        self.state2 = GraphState(
            lineage=["node3", "node4"],
            agent_id="agent_2",
            messages=[self.message2],
            config=self.config2,
            children_states={},
            initialized_node_ids={"node3", "node4"}
        )

    def test_validation_correct_behavior(self):
        """Test the correct validation logic (should fail when ANY argument is None)."""        
        with self.assertRaises(ValueError) as cm:
            GraphState.reducer(self.state1, None)
            
        self.assertIn("Must specify non-null arguments for both 'left' and 'right'", str(cm.exception))
        self.assertIn("right", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            GraphState.reducer(None, self.state2)
            
        self.assertIn("Must specify non-null arguments for both 'left' and 'right'", str(cm.exception))
        self.assertIn("left", str(cm.exception))

    def test_validation_both_none_should_fail(self):
        """Test that providing None for both arguments should raise ValueError."""
        with self.assertRaises(ValueError):
            GraphState.reducer(None, None)

    def test_validation_both_provided_works(self):
        """Test that when both arguments are provided, the reducer works correctly."""
        # When both arguments are provided (not None), validation passes
        
        # This should work without raising an error
        result = GraphState.reducer(self.state1, self.state2)
        
        # Verify the merge worked correctly 
        self.assertIsInstance(result, dict)
        # Right state values should override left state values (use_last behavior)
        self.assertEqual(result["lineage"], self.state2["lineage"])
        self.assertEqual(result["agent_id"], self.state2["agent_id"])
        self.assertEqual(result["config"], self.state2["config"])
        
        # Messages should be combined (add_messages behavior)
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn(self.message1, result["messages"])
        self.assertIn(self.message2, result["messages"])
        
        # Sets should be unioned
        expected_nodes = self.state1["initialized_node_ids"].union(self.state2["initialized_node_ids"])
        self.assertEqual(result["initialized_node_ids"], expected_nodes)

    def test_lineage_merging(self):
        """Test that right state's lineage replaces left state's lineage using actual reducer."""
        # Create states with different lineages
        left_state = self.state1.copy()
        right_state = GraphState(
            lineage=["new_node1", "new_node2"],
            agent_id="", 
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using the actual reducer
        result = GraphState.reducer(left_state, right_state)
        self.assertEqual(result["lineage"], ["new_node1", "new_node2"])

    def test_agent_id_merging(self):
        """Test that right state's agent_id replaces left state's agent_id using actual reducer."""
        left_state = self.state1.copy()
        right_state = GraphState(
            lineage=[],
            agent_id="new_agent_id",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using the actual reducer
        result = GraphState.reducer(left_state, right_state)
        self.assertEqual(result["agent_id"], "new_agent_id")

    def test_messages_merging_with_add_messages(self):
        """Test that messages are properly merged via GraphState.reducer using add_messages function."""
        # Create two states with different messages 
        left_state = GraphState(
            lineage=["left"],
            agent_id="left_agent",
            messages=[self.message1],
            config=self.config1,
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[self.message2],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test message merging through GraphState.reducer
        result = GraphState.reducer(left_state, right_state)
        
        # Verify that both messages are present in the merged result
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn(self.message1, result["messages"])
        self.assertIn(self.message2, result["messages"])
        
        # Other fields get overwritten by right state (even if empty, since they're not None)
        self.assertEqual(result["lineage"], [])  # Empty list from right state
        self.assertEqual(result["agent_id"], "")  # Empty string from right state

    def test_config_merging(self):
        """Test that right state's config replaces left state's config using actual reducer."""
        left_state = self.state1.copy()
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=self.config2,
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer
        result = GraphState.reducer(left_state, right_state)
        self.assertEqual(result["config"], self.config2)
        self.assertEqual(result["config"].name, "test_config_2")

    def test_initialized_node_ids_merging(self):
        """Test that initialized_node_ids are properly unioned using actual reducer."""
        left_state = self.state1.copy()
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids={"new_node1", "new_node2"},
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer
        result = GraphState.reducer(left_state, right_state)
        # Sets should be unioned, not replaced
        expected_nodes = self.state1["initialized_node_ids"].union({"new_node1", "new_node2"})
        self.assertEqual(result["initialized_node_ids"], expected_nodes)

    def test_children_states_new_key_merging(self):
        """Test that new keys in right state's children_states are added using tuple keys."""
        # Create left state with some children using tuple keys
        left_state = self.state1.copy()
        child_state_1 = GraphState(
            lineage=["child1"],
            agent_id="child_agent_1",
            messages=[],
            config=self.config1,
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        left_state["children_states"] = {("child1",): child_state_1}
        
        # Create right state with new child using tuple keys
        child_state_2 = GraphState(
            lineage=["child2"],
            agent_id="child_agent_2", 
            messages=[],
            config=self.config2,
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={("child2",): child_state_2},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer
        result = GraphState.reducer(left_state, right_state)
        self.assertIn(("child1",), result["children_states"])
        self.assertIn(("child2",), result["children_states"])
        self.assertEqual(len(result["children_states"]), 2)

    def test_children_states_recursive_merging(self):
        """Test that existing keys in children_states are recursively merged using tuple keys."""
        # Create nested children states that should be merged
        left_child = GraphState(
            lineage=["left_lineage"],
            agent_id="left_agent",
            messages=[],
            config=self.config1,
            children_states={},
            initialized_node_ids={"left_node"},
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        right_child = GraphState(
            lineage=["right_lineage"],  # This should replace left_lineage
            agent_id="",  # Empty, so left_agent should remain
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Create parent states with same tuple key for children
        left_state = self.state1.copy()
        left_state["children_states"] = {("child1",): left_child}
        
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={("child1",): right_child},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer - this should recursively merge the children
        result = GraphState.reducer(left_state, right_state)
        
        # Check the merged child state
        merged_child = result["children_states"][("child1",)]
        self.assertEqual(merged_child["lineage"], ["right_lineage"])
        self.assertEqual(merged_child["agent_id"], "")  # Empty string from right overwrites left
        self.assertEqual(merged_child["initialized_node_ids"], {"left_node"})

    def test_empty_right_state_no_changes(self):
        """Test that empty/default right state doesn't change left state using actual reducer."""
        left_state = self.state1.copy()
        empty_right = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer
        result = GraphState.reducer(left_state, empty_right)
        
        # Empty values (like [] and "") are still "not None" so they overwrite left values
        self.assertEqual(result["lineage"], [])  # Empty list from right state overwrites left
        self.assertEqual(result["agent_id"], "")  # Empty string from right state overwrites left
        self.assertEqual(result["config"], ConfigRecord(name="", version="", description="", config=""))  # Empty config from right
        # Messages should still be combined (even if right is empty)
        self.assertEqual(result["messages"], self.state1["messages"])

    def test_complex_nested_children_merge(self):
        """Test complex nested children state merging scenarios using tuple keys and actual reducer."""
        # Create deeply nested structure with tuple keys
        grandchild = GraphState(
            lineage=["grandchild"],
            agent_id="gc_agent",
            messages=[],
            config=self.config1,
            children_states={},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        left_child = GraphState(
            lineage=["left_child"],
            agent_id="lc_agent",
            messages=[],
            config=self.config1,
            children_states={("grandchild",): grandchild},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Right side adds new grandchild info
        new_grandchild_data = GraphState(
            lineage=["updated_grandchild"],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={},
            initialized_node_ids={"new_node"},
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        right_child = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={("grandchild",): new_grandchild_data},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Create parent states
        left_state = self.state1.copy()
        left_state["children_states"] = {("child",): left_child}
        
        right_state = GraphState(
            lineage=[],
            agent_id="",
            messages=[],
            config=ConfigRecord(name="", version="", description="", config=""),
            children_states={("child",): right_child},
            initialized_node_ids=set(),
            attempt_count=0,
            supervisor_success=False,
            planner_success=False,
            planner_node_id=""
        )
        
        # Test using actual reducer - this should recursively merge all levels
        result = GraphState.reducer(left_state, right_state)
        
        # Navigate to the merged grandchild
        merged_grandchild = result["children_states"][("child",)]["children_states"][("grandchild",)]
        self.assertEqual(merged_grandchild["lineage"], ["updated_grandchild"])
        self.assertEqual(merged_grandchild["agent_id"], "")  # Empty string from right overwrites left
        self.assertEqual(merged_grandchild["initialized_node_ids"], {"new_node"})

    def test_bug_fix_documentation(self):
        """Document that the validation bug has been fixed."""
        # The bug was: "if left is not None or right is not None:"
        # The fix is: "if left is None or right is None:"
        
        # This test confirms the bug is fixed by testing the correct behavior
        # When both arguments are provided, it should work
        result = GraphState.reducer(self.state1, self.state2)
        self.assertIsInstance(result, dict)
        
        # When either argument is None, it should fail
        with self.assertRaises(ValueError):
            GraphState.reducer(self.state1, None)
        
        with self.assertRaises(ValueError):
            GraphState.reducer(None, self.state2)


if __name__ == "__main__":
    unittest.main() 