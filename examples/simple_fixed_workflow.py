#!/usr/bin/env python
"""Simple example demonstrating the use of a fixed LangGraph workflow.

This example shows how to create and run a basic workflow with fixed configuration
to process a simple task, including error handling demonstration.
"""

import sys
import os
import json
from typing import Dict, Any
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_evo.core.workflow import create_workflow, WorkflowError
from langgraph_evo.core.agents import TaskExecutorAgent, InputProcessorAgent


class CustomInputProcessor(InputProcessorAgent):
    """Custom input processor that preserves the original options."""
    
    def process(self, state):
        """Process the input data while preserving original options."""
        # Get the original options before processing
        original_options = state.get("task_input", {}).get("options", {})
        
        # Process normally using parent method
        updated_state = super().process(state)
        
        # Put the original options back
        if "parameters" in updated_state.get("task_input", {}):
            updated_state["task_input"]["parameters"] = original_options
        
        return updated_state


class ErrorTestAgent(TaskExecutorAgent):
    """A test agent that raises an error for demonstration purposes."""
    
    def process(self, state):
        """Raise an error instead of processing normally."""
        # Print debug info to help diagnose why error isn't triggered
        print(f"ErrorTestAgent processing state: {state.get('task_input', {}).get('parameters', {})}")
        
        # Force an error for the demonstration
        if state.get("task_input", {}).get("parameters", {}).get("should_error"):
            print("Triggering test error...")
            raise ValueError("This is a test error to demonstrate error handling")
        return super().process(state)


def main():
    """Run the example workflow."""
    parser = argparse.ArgumentParser(description="Run a simple LangGraph workflow example")
    parser.add_argument("--error", action="store_true", help="Trigger an error in the workflow")
    args = parser.parse_args()
    
    print("=== Running Fixed Workflow Example ===\n")
    
    # Create a workflow
    workflow = create_workflow()
    
    # Replace the input processor to preserve options
    workflow.input_processor = CustomInputProcessor()
    
    # If we're testing errors, replace the executor with our error test agent
    if args.error:
        print("Using error test agent for demonstration\n")
        workflow.task_executor = ErrorTestAgent()
    
    # Always rebuild the graph when agents change
    workflow.graph = workflow._build_graph()
    workflow.runnable = workflow.graph.compile()
    
    # Define a simple input task
    task_input = {
        "query": "Process this simple text example",
        "options": {
            "verbose": True,
            "format": "json",
            # Only set should_error if --error flag is passed
            "should_error": args.error
        }
    }
    
    # Generate a task ID
    task_id = "example_task_001"
    
    try:
        # Run the workflow
        print(f"Running workflow with task_id: {task_id}")
        result = workflow.run(task_input, task_id)
        
        # Print the result
        print("\n=== Workflow Result ===")
        print(f"Success: {result.get('success', False)}")
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print("Output:")
            print(json.dumps(result["result"], indent=2))
        
        print("\nExecution Steps:")
        for step in result.get("steps", []):
            print(f"- {step.get('agent')}: {step.get('action')}")
        
    except Exception as e:
        print(f"Uncaught exception: {type(e).__name__}: {str(e)}")
    
    print("\n=== Example Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 