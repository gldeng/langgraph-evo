"""PSI POC example using the langgraph_evo package with tool registry."""
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
# Load environment variables from .env file
load_dotenv()

# Set up langchain tools
from langchain_tavily import TavilySearch

# Import from the langgraph_evo package
from langgraph_evo.core.store import initialize_configs
from langgraph_evo.core.tool_registry import register_tool, register_standard_tools
from langgraph_evo import create_psi_graph

# Register the tools in the registry
web_search = TavilySearch(max_results=3)
register_tool("web_search", web_search)

# Register standard tools (add, multiply, divide)
register_standard_tools()

# Sample graph configuration - fixed YAML format
graph_config = """
tools:
  - name: web_search
    script: web_search
    description: Search the web for information.
  - name: add
    script: add
    description: Add two numbers.
  - name: multiply
    script: multiply
    description: Multiply two numbers.
  - name: divide
    script: divide
    description: Divide two numbers.
nodes:
  - name: supervisor
    type: react
    is_entry_point: true
    config:
      model: openai:gpt-4.1
      prompt: |
          You are a supervisor managing two agents:
          - a research agent. Assign research-related tasks to this agent
          - a math agent. Assign math-related tasks to this agent
          Assign work to one agent at a time, do not call agents in parallel.
          Do not do any work yourself.
          
          IMPORTANT: The human may ask follow-up questions. Use your memory of previous 
          interactions to provide context-aware responses. Refer to previous research 
          and calculations when answering follow-ups.
  - name: research_agent
    type: react
    config:
      model: openai:gpt-3.5-turbo
      prompt: |
          You are a research agent.
          INSTRUCTIONS:
          - Assist ONLY with research-related tasks, DO NOT do any math
          - After you're done with your tasks, respond to the supervisor directly
          - Respond ONLY with the results of your work, do NOT include ANY other text.
          - When processing follow-up questions, refer to previously researched information when appropriate.
      tools:
        - web_search
  - name: math_agent
    type: react
    config:
      model: openai:gpt-4.1
      prompt: |
          You are a math agent.
          INSTRUCTIONS:
          - Assist ONLY with math-related tasks
          - After you're done with your tasks, respond to the supervisor directly
          - Respond ONLY with the results of your work, do NOT include ANY other text.
          - When processing follow-up questions, refer to previously calculated values when appropriate.
      tools:
        - add
        - multiply
        - divide
edges:
  - from: supervisor
    to: research_agent
  - from: supervisor
    to: math_agent
"""

def print_messages(messages):
    """Helper function to print message history in a readable format."""
    print("\nMessage History:")
    for i, message in enumerate(messages):
        if isinstance(message, dict):
            print(f"[{i}] Role: {message.get('role', 'unknown')}")
            content = message.get('content', '')
            print(f"Content: {content}")
        else:
            print(f"[{i}] {str(message)[:100]}...")
        print("-" * 40)

def run_psi_system():
    """Run the PSI system with support for follow-up questions."""
    # Create the store instance to be shared
    store = InMemoryStore()

    # Create the PSI graph with main task handler
    psi = create_psi_graph(store)

    # Initialize configurations in the store
    initialize_configs(store, graph_config)

    # Log store status
    print("\nStore initialization status:")
    try:
        configs = store.search(("agent", "configs"))
        print(f"Config found in store: {'Yes' if configs else 'No'}")
        print(f"Config count: {len(configs)}")
    except Exception as e:
        print(f"Error accessing store: {e}")

    # Initial state with the first user question
    current_state = {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ],
        "initialized_node_ids": {}
    }

    # Run the PSI system in a loop to support follow-up questions
    try:
        
        while True:
            print("\nProcessing question...")
            # Invoke the PSI system with the current state and additional context
            current_state = psi.invoke(
                current_state,
                config={
                    "metadata": {
                        "agent_id": ("root",)
                    }
                }
            )
            
            # Print the conversation history
            if "messages" in current_state:
                print_messages(current_state["messages"])
            else:
                print("No messages found in the state")
            
            # Ask for a follow-up question
            follow_up = input("\nFollow-up question (or type 'exit' to quit): ").strip()
            
            # Check if the user wants to exit
            if follow_up.lower() in ["exit", "quit", "bye"]:
                print("Exiting conversation.")
                break
            
            # Add the follow-up question to the conversation
            current_state["messages"].append({
                "role": "user",
                "content": follow_up
            })
            
        
    except Exception as e:
        print(f"Error running the system: {e}")

if __name__ == "__main__":
    run_psi_system() 