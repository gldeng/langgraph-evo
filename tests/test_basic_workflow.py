"""Basic tests for the workflow functionality."""

import unittest
import sys
import pprint
from typing import Dict, Any
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_evo import create_fixed_workflow
from langgraph_evo.core.agents import TaskExecutorAgent


class SimpleErrorAgent(TaskExecutorAgent):
    """A simple agent that raises an error when processed."""
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def process(self, state):
        """Raise an error for testing."""
        raise ValueError("Test error")


class TestBasicWorkflow(unittest.TestCase):
    """Basic tests for the workflow."""
    
    def test_simple_workflow(self):
        """Test creating and running a simple workflow."""
        # Create workflow
        workflow = create_fixed_workflow()
        
        # Input data
        data = {"test": "data"}
        
        # Run workflow
        result = workflow.run(data)
        
        # Assertions
        self.assertIn("task_id", result)
        self.assertIn("result", result)
        self.assertIn("steps", result)
        self.assertTrue(result.get("success", False))
        self.assertNotIn("error", result)
    
    def test_basic_error_handling(self):
        """Test basic error handling in the workflow."""
        # Create workflow
        workflow = create_fixed_workflow()
        
        # Replace an agent with one that will error
        workflow.task_executor = SimpleErrorAgent()
        
        # Rebuild the graph with new agent
        workflow.graph = workflow._build_graph()
        workflow.runnable = workflow.graph.compile()
        
        # Input data
        data = {"test": "data"}
        
        # Run workflow (should handle the error)
        result = workflow.run(data)
        
        print("\n=== TEST ERROR HANDLING RESULT ===")
        pprint.pprint(result)
        print("==================================")
        
        # Check that we get an error response
        self.assertIn("task_id", result)
        self.assertIn("error", result)
        self.assertIn("Test error", result["error"])
        self.assertFalse(result["success"])
        self.assertTrue(result["metadata"].get("error"))


if __name__ == "__main__":
    unittest.main() 