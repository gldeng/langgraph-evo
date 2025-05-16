#!/usr/bin/env python
"""Dynamic workflow example with multiple agents.

This example demonstrates how to create a dynamic workflow with multiple agents
that can be configured at runtime for different use cases.
"""

import sys
import os
import json
import uuid
from typing import Dict, Any, List, Optional, Callable
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_evo.core.workflow import create_workflow
from langgraph_evo.core.agents import InputProcessorAgent, TaskExecutorAgent, OutputFormatterAgent
from langgraph_evo.core.state import TaskState, merge_state


class ResearchAgent:
    """Agent that conducts research on a topic."""
    
    def __init__(self, research_depth: int = 1):
        """Initialize with configurable research depth."""
        self.research_depth = research_depth
    
    def process(self, state: TaskState) -> Dict[str, Any]:
        """Process the state by conducting research."""
        # Get the input query
        task_input = state.get("task_input", {})
        query = task_input.get("query", "")
        
        # Simulate conducting research
        research_results = []
        for i in range(self.research_depth):
            research_results.append({
                "source": f"source-{i+1}",
                "content": f"Research finding {i+1} for query: {query}"
            })
        
        # Update context with research results
        context = state.get("context", {})
        context["research_results"] = research_results
        
        # Add a message about the research
        messages = [{
            "role": "system", 
            "content": f"Conducted research with depth {self.research_depth}. Found {len(research_results)} results."
        }]
        
        # Add a step
        steps = [{
            "agent": "ResearchAgent",
            "action": "research",
            "details": f"Depth: {self.research_depth}, Results: {len(research_results)}"
        }]
        
        # Return the updated state
        return {
            "context": context,
            "messages": messages,
            "steps": steps
        }


class ContentCreationAgent:
    """Agent that creates content based on research and input."""
    
    def __init__(self, format_type: str = "summary"):
        """Initialize with configurable format type."""
        self.format_type = format_type
    
    def process(self, state: TaskState) -> Dict[str, Any]:
        """Process the state by creating content."""
        # Get the research results from context
        context = state.get("context", {})
        research_results = context.get("research_results", [])
        
        # Get the input query
        task_input = state.get("task_input", {})
        query = task_input.get("query", "")
        
        # Create content based on format type
        content = ""
        if self.format_type == "summary":
            content = f"Summary of {len(research_results)} research findings on: {query}"
            for i, result in enumerate(research_results):
                content += f"\n- Finding {i+1}: {result['content']}"
        
        elif self.format_type == "analysis":
            content = f"Detailed analysis of research on: {query}\n\n"
            for i, result in enumerate(research_results):
                content += f"Source {i+1} ({result['source']}): {result['content']}\n\n"
                
        elif self.format_type == "brief":
            content = f"Brief overview of research on: {query}\n"
            if research_results:
                content += f"Key finding: {research_results[0]['content']}"
        
        # Update context with generated content
        context["generated_content"] = content
        
        # Add a message about the content creation
        messages = [{
            "role": "system", 
            "content": f"Created {self.format_type} content based on research."
        }]
        
        # Add a step
        steps = [{
            "agent": "ContentCreationAgent",
            "action": "create_content",
            "details": f"Format: {self.format_type}, Length: {len(content)} chars"
        }]
        
        # Return the updated state
        return {
            "context": context,
            "messages": messages,
            "steps": steps
        }


class DynamicWorkflow:
    """Dynamic workflow that can be configured with different agents."""
    
    def __init__(self):
        """Initialize the dynamic workflow."""
        # Create a base workflow
        self.workflow = create_workflow()
        
        # Initialize agent slots
        self.research_agent = None
        self.content_creation_agent = None
        
        # Track whether the graph needs to be rebuilt
        self.needs_rebuild = False
    
    def set_research_agent(self, agent: Optional[ResearchAgent] = None):
        """Set or remove the research agent."""
        self.research_agent = agent
        self.needs_rebuild = True
    
    def set_content_creation_agent(self, agent: Optional[ContentCreationAgent] = None):
        """Set or remove the content creation agent."""
        self.content_creation_agent = agent
        self.needs_rebuild = True
    
    def build_graph(self):
        """Build the workflow graph based on configured agents."""
        if not self.needs_rebuild:
            return
        
        # Define the custom graph edges
        edges = {}
        current_node = "input_processor"
        
        # Add research agent if configured
        if self.research_agent:
            edges[current_node] = lambda x: "research_agent"
            edges["research_agent"] = lambda x: "task_executor" if not self.content_creation_agent else "content_creation_agent"
            current_node = "research_agent"
        
        # Add content creation agent if configured
        if self.content_creation_agent:
            if current_node == "input_processor":
                edges[current_node] = lambda x: "content_creation_agent"
            edges["content_creation_agent"] = lambda x: "task_executor"
            current_node = "content_creation_agent"
        
        # Connect to task executor if not already connected
        if current_node == "input_processor":
            edges[current_node] = lambda x: "task_executor"
        
        # Complete the graph
        edges["task_executor"] = lambda x: "output_formatter"
        
        # Create the graph with custom edges
        self.workflow.graph = self.workflow._build_graph(edges)
        
        # Add custom agents to the graph nodes
        if self.research_agent:
            self.workflow.graph.add_node("research_agent", self.research_agent.process)
        
        if self.content_creation_agent:
            self.workflow.graph.add_node("content_creation_agent", self.content_creation_agent.process)
        
        # Compile the runnable graph
        self.workflow.runnable = self.workflow.graph.compile()
        
        # Reset the rebuild flag
        self.needs_rebuild = False
    
    def run(self, task_input: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the dynamic workflow with the given input."""
        # Ensure the graph is built
        self.build_graph()
        
        # Generate a task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Run the workflow
        return self.workflow.run(task_input, task_id)


def main():
    """Run the dynamic workflow example."""
    parser = argparse.ArgumentParser(description="Run a dynamic workflow example with configurable agents")
    parser.add_argument("--research", action="store_true", help="Include research agent")
    parser.add_argument("--research-depth", type=int, default=2, help="Research depth (1-5)")
    parser.add_argument("--content", action="store_true", help="Include content creation agent")
    parser.add_argument("--format", choices=["summary", "analysis", "brief"], default="summary", 
                      help="Content format (summary, analysis, brief)")
    parser.add_argument("--query", type=str, default="Explain the concept of workflow orchestration",
                      help="Query to process")
    args = parser.parse_args()
    
    # Print welcome message
    print("=== Running Dynamic Workflow Example ===\n")
    
    # Create the dynamic workflow
    workflow = DynamicWorkflow()
    
    # Configure the workflow based on arguments
    if args.research:
        depth = max(1, min(5, args.research_depth))  # Clamp between 1-5
        print(f"Adding Research Agent (depth: {depth})")
        workflow.set_research_agent(ResearchAgent(research_depth=depth))
    
    if args.content:
        print(f"Adding Content Creation Agent (format: {args.format})")
        workflow.set_content_creation_agent(ContentCreationAgent(format_type=args.format))
    
    # Define task input
    task_input = {
        "query": args.query,
        "options": {
            "verbose": True,
            "format": "json"
        }
    }
    
    # Generate a task ID
    task_id = f"dynamic_example_{uuid.uuid4().hex[:8]}"
    
    try:
        # Run the workflow
        print(f"\nRunning workflow with task_id: {task_id}")
        print(f"Query: {args.query}")
        
        result = workflow.run(task_input, task_id)
        
        # Print the result
        print("\n=== Workflow Result ===")
        print(f"Success: {result.get('success', False)}")
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            # Print the content if it was generated
            if "context" in result.get("result", {}) and "generated_content" in result["result"]["context"]:
                print("\nGenerated Content:")
                print(result["result"]["context"]["generated_content"])
            else:
                print("\nOutput:")
                print(json.dumps(result["result"], indent=2))
        
        print("\nExecution Steps:")
        for step in result.get("steps", []):
            print(f"- {step.get('agent')}: {step.get('action')} | {step.get('details', '')}")
        
    except Exception as e:
        print(f"Uncaught exception: {type(e).__name__}: {str(e)}")
    
    print("\n=== Example Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 