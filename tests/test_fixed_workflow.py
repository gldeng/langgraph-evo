"""Unit tests for the fixed workflow implementation."""

import unittest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from src.langgraph_evo.core import create_workflow
from src.langgraph_evo.core.state import TaskState
from src.langgraph_evo.core.workflow import WorkflowError


class TestFixedWorkflow(unittest.TestCase):
    """Test cases for the FixedWorkflow implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.workflow = create_workflow()
        self.test_input = {
            "query": "Test workflow execution",
            "options": {
                "verbose": True
            }
        }
        self.task_id = "test_task_123"
    
    def test_workflow_execution(self):
        """Test that the workflow executes all nodes in the correct sequence."""
        # Run the workflow
        result = self.workflow.run(self.test_input, self.task_id)
        
        # Verify basic structure of the result
        self.assertEqual(result["task_id"], self.task_id)
        self.assertIn("result", result)
        self.assertIsInstance(result["steps"], list)
        self.assertTrue(result["success"])
        
        # Get unique agents by their name, preserving order
        actual_agents = []
        seen = set()
        for step in result["steps"]:
            agent = step.get("agent")
            if agent and agent not in seen:
                seen.add(agent)
                actual_agents.append(agent)
        
        # Check that all expected agents appear in the steps
        expected_agents = ["InputProcessor", "TaskAnalyzer", "TaskExecutor", "OutputFormatter"]
        for expected in expected_agents:
            self.assertIn(expected, actual_agents, f"Missing expected agent: {expected}")
        
        # Verify the output contains the expected data
        self.assertIn("result", result["result"])
        self.assertIn("metadata", result["result"])
        self.assertEqual(result["result"]["metadata"]["task_id"], self.task_id)
    
    def test_custom_configuration(self):
        """Test that the workflow accepts custom configuration."""
        # Create a workflow with custom configuration
        custom_config = {
            "input_processor_config": {"custom_option": "test_value"}
        }
        custom_workflow = create_workflow(custom_config)
        
        # Verify the configuration was stored correctly
        self.assertEqual(
            custom_workflow.input_processor.config.get("custom_option"),
            "test_value"
        )
        
        # Ensure the workflow still works with custom config
        result = custom_workflow.run(self.test_input, self.task_id)
        self.assertEqual(result["task_id"], self.task_id)
        self.assertIn("result", result)
        self.assertTrue(result["success"])
    
    def test_error_handling(self):
        """Test that the workflow properly handles errors during execution."""
        # Create a workflow with a mock executor that raises an exception
        workflow = create_workflow()
        
        # Create a test agent that raises an exception
        class ErrorAgent(MagicMock):
            def process(self, state):
                raise ValueError("Test error")
        
        # Replace the task executor with our error agent
        workflow.task_executor = ErrorAgent()
        
        # Run the workflow
        result = workflow.run(self.test_input, self.task_id)
        
        # Verify error structure is correct
        self.assertEqual(result["task_id"], self.task_id)
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test error", result["error"])
        self.assertTrue(result["metadata"].get("error", False))


if __name__ == "__main__":
    unittest.main() 