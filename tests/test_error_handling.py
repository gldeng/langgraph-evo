#!/usr/bin/env python
"""Test script to verify error handling in LangGraph workflows.

This script tests the error handling and routing capabilities of the workflow,
ensuring that errors are properly propagated and handled.
"""

import sys
import os
import unittest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.langgraph_evo.core import create_workflow
from src.langgraph_evo.core.state import TaskState
from src.langgraph_evo.core.workflow import WorkflowError


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in the workflow."""
    
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
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through the workflow."""
        # Create a mock agent that raises an exception
        mock_agent = MagicMock()
        mock_agent.process.side_effect = ValueError("Test error")
        
        # Replace the task analyzer with our mock
        self.workflow.task_analyzer = mock_agent
        
        # Rebuild the graph
        self.workflow.graph = self.workflow._build_graph()
        self.workflow.runnable = self.workflow.graph.compile()
        
        # Run the workflow
        result = self.workflow.run(self.test_input, self.task_id)
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test error", result["error"])
        
        # Verify steps - should include all nodes in the workflow
        agent_steps = [step["agent"] for step in result["steps"]]
        self.assertIn("InputProcessor", agent_steps)
        self.assertIn("TaskAnalyzer", agent_steps)
        
        # Check for the presence of downstream nodes
        self.assertIn("TaskExecutor", agent_steps)
        self.assertIn("OutputFormatter", agent_steps)
        
        # Verify the status of the nodes after the error
        executor_steps = [step for step in result["steps"] if step["agent"] == "TaskExecutor"]
        if executor_steps and "action" in executor_steps[0]:
            self.assertEqual(executor_steps[0]["action"], "skip_due_to_error",
                         "TaskExecutor did not correctly detect error state")
        
        formatter_steps = [step for step in result["steps"] if step["agent"] == "OutputFormatter"]
        if formatter_steps and "action" in formatter_steps[0]:
            self.assertEqual(formatter_steps[0]["action"], "skip_due_to_error",
                         "OutputFormatter should be called with skip_due_to_error action")
    
    def test_error_in_executor(self):
        """Test error handling when the error occurs in the task executor."""
        # Create a mock agent that raises an exception
        mock_agent = MagicMock()
        mock_agent.process.side_effect = ValueError("Executor test error")
        
        # Replace the task executor with our mock
        self.workflow.task_executor = mock_agent
        
        # Rebuild the graph
        self.workflow.graph = self.workflow._build_graph()
        self.workflow.runnable = self.workflow.graph.compile()
        
        # Run the workflow
        result = self.workflow.run(self.test_input, self.task_id)
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Executor test error", result["error"])
        
        # Verify steps - should include all nodes in the workflow
        agent_steps = [step["agent"] for step in result["steps"]]
        self.assertIn("InputProcessor", agent_steps)
        self.assertIn("TaskAnalyzer", agent_steps)
        self.assertIn("TaskExecutor", agent_steps)
        self.assertIn("OutputFormatter", agent_steps)


if __name__ == "__main__":
    unittest.main() 