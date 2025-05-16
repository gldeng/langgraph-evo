#!/usr/bin/env python
"""Debug script to test workflow error handling."""

import sys
import os
import pprint
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_evo import create_fixed_workflow
from langgraph_evo.core.agents import TaskExecutorAgent, InputProcessorAgent, TaskAnalyzerAgent, OutputFormatterAgent


class DebugInputProcessorAgent(InputProcessorAgent):
    def process(self, state):
        print("DEBUG: InputProcessorAgent process method called")
        return super().process(state)


class DebugTaskAnalyzerAgent(TaskAnalyzerAgent):
    def process(self, state):
        print("DEBUG: TaskAnalyzerAgent process method called")
        return super().process(state)


class DebugOutputFormatterAgent(OutputFormatterAgent):
    def process(self, state):
        print("DEBUG: OutputFormatterAgent process method called")
        return super().process(state)


class CustomErrorAgent(TaskExecutorAgent):
    """A custom agent that raises an error when processed."""
    
    def __init__(self, config=None):
        super().__init__(config)
        print("DEBUG: CustomErrorAgent initialized")
    
    def process(self, state):
        """Raise an error for testing."""
        print("DEBUG: CustomErrorAgent process method called")
        print(f"DEBUG: State received: {state}")
        raise ValueError("Test error from CustomErrorAgent")


def main():
    """Run the workflow with error handling test."""
    print("\n=== Starting Workflow Error Handling Test ===\n")
    
    # Create workflow
    workflow = create_fixed_workflow()
    
    # Replace all agents with debug versions
    workflow.input_processor = DebugInputProcessorAgent()
    workflow.task_analyzer = DebugTaskAnalyzerAgent()
    workflow.task_executor = CustomErrorAgent()
    workflow.output_formatter = DebugOutputFormatterAgent()
    
    # Rebuild the graph with new agents
    workflow.graph = workflow._build_graph()
    workflow.runnable = workflow.graph.compile()
    
    print("Workflow agents after replacement:")
    print(f"Input processor: {workflow.input_processor}")
    print(f"Task analyzer: {workflow.task_analyzer}")
    print(f"Task executor: {workflow.task_executor}")
    print(f"Output formatter: {workflow.output_formatter}")
    
    # Input data
    data = {"test": "data"}
    
    try:
        print("\n--- Running workflow with error agent ---")
        # Run workflow (should handle the error)
        result = workflow.run(data)
        
        print("\n=== TEST ERROR HANDLING RESULT ===")
        pprint.pprint(result)
        print("==================================")
        
        # Check for error
        if "error" in result:
            print(f"\nERROR FOUND: {result['error']}")
            print(f"SUCCESS FLAG: {result.get('success', 'Not set')}")
        else:
            print("\nNO ERROR FOUND IN RESULT")
            print(f"SUCCESS FLAG: {result.get('success', 'Not set')}")
            print("Steps recorded:")
            for step in result.get("steps", []):
                print(f"  - {step.get('agent')}: {step.get('action')}")
    
    except Exception as e:
        print(f"\nEXCEPTION CAUGHT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 