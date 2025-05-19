import re
from typing import Annotated, Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages, AnyMessage



from dotenv import load_dotenv
import os
import uuid

# Load environment variables from .env file
load_dotenv()

from langchain_tavily import TavilySearch

from langgraph.prebuilt import create_react_agent


from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


web_search = TavilySearch(max_results=3)


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


# Handoffs
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="Assign task to a math agent.",
)

supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)

from langgraph.graph import END

research_agent = create_react_agent(
    model="openai:gpt-3.5-turbo",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)
# for chunk in research_agent.stream(
#     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
# ):
#     pretty_print_messages(chunk)


graph_config = '''
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
        tools:
          - add
          - multiply
          - divide
edges:
    - from: supervisor
      to: research_agent
    - from: supervisor
      to: math_agent
'''

# Data models to represent the configuration
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

# Tool configuration models
class BaseTool(BaseModel):
    """Base model for all tools."""
    name: str
    description: str

class ScriptTool(BaseTool):
    """Tool that references a Python function by name."""
    script: str

class Tool(BaseModel):
    """Tool definition with script reference and description."""
    name: str
    script: str
    description: str

# Node configuration models
class NodeConfigBase(BaseModel):
    """Base configuration for all node types."""
    pass

class ReactNodeConfig(NodeConfigBase):
    """Configuration specific to 'react' type nodes."""
    model: str
    tools: Optional[List[str]] = None  # Tool names referenced from top-level tools
    prompt: Optional[str] = None  # Can be multi-line string in YAML using | or > syntax

class NodeConfig(BaseModel):
    """Configuration for a single node."""
    type: Literal["react", "custom", "task_handler"]
    config: Union[ReactNodeConfig, Dict[str, Any]]

class Node(BaseModel):
    """Complete node definition."""
    name: str
    type: str
    is_entry_point: Optional[bool] = None
    config: Dict[str, Any]

# Edge configuration model
class Edge(BaseModel):
    """Connection between nodes."""
    from_: str = Field(..., alias="from")
    to: str

# Complete graph configuration model
class GraphConfig(BaseModel):
    """Complete graph configuration."""
    tools: Optional[List[Tool]] = None
    nodes: List[Node]
    edges: List[Edge]
    
    class Config:
        """Pydantic configuration."""
        populate_by_name = True  # Allow populating from_/from aliases

# Helper function to parse the configuration
def parse_graph_config(config_str: str) -> GraphConfig:
    """Parse graph configuration from YAML string."""
    import yaml
    config_dict = yaml.safe_load(config_str)
    
    # Process the config to match our model expectations
    processed_config = config_dict.copy()
    
    # Process nodes to ensure tool references are properly handled
    if "nodes" in processed_config:
        for node in processed_config["nodes"]:
            if "config" in node and "tools" in node["config"]:
                # Ensure tools is a list of strings (tool names)
                if isinstance(node["config"]["tools"], list):
                    # Convert any non-string entries to strings if needed
                    node["config"]["tools"] = [
                        tool if isinstance(tool, str) else tool["name"] 
                        for tool in node["config"]["tools"]
                    ]
            
            # Ensure prompt is properly handled, especially for multi-line strings
            if "config" in node and "prompt" in node["config"]:
                # YAML loader should already handle |, >, etc. notation properly,
                # but we can add additional processing if needed
                pass
    
    return GraphConfig(**processed_config)

# Example usage:
try:
    parsed_config = parse_graph_config(graph_config)
    print(f"Successfully parsed configuration with {len(parsed_config.nodes)} nodes and {len(parsed_config.edges)} edges")
except Exception as e:
    print(f"Error parsing configuration: {e}")


def create_graph(config: GraphConfig):
    """Create a graph based on the configuration."""
    # Create a mapping of tool names to actual tool objects
    tool_map = {}
    if config.tools:
        for tool_config in config.tools:
            if hasattr(tool_config, 'script') and tool_config.script:
                # Get the tool object by evaluating the script name
                try:
                    tool_obj = eval(tool_config.script)
                    tool_map[tool_config.name] = tool_obj
                except (NameError, AttributeError) as e:
                    print(f"Warning: Could not evaluate tool script '{tool_config.script}': {e}")
    
    # Create handoff tools for node transitions
    handoff_tools = {}
    for edge in config.edges:
        tool_name = f"transfer_to_{edge.to}"
        if tool_name not in handoff_tools:
            handoff_tools[tool_name] = create_handoff_tool(
                agent_name=edge.to, 
                description=f"Transfer to {edge.to}"
            )
    
    # Create nodes
    nodes = {}
    for node in config.nodes:
        if node.type == "react":
            # Collect tools for this node
            node_tools = []
            
            # Add any tools specified in the node config
            if "tools" in node.config and node.config["tools"]:
                for tool_name in node.config["tools"]:
                    if tool_name in tool_map:
                        node_tools.append(tool_map[tool_name])
                    else:
                        print(f"Warning: Tool '{tool_name}' referenced in node '{node.name}' not found in tool map")
            
            # Add handoff tools for edges originating from this node
            for edge in config.edges:
                if edge.from_ == node.name and f"transfer_to_{edge.to}" in handoff_tools:
                    node_tools.append(handoff_tools[f"transfer_to_{edge.to}"])
            
            # Get prompt from config or use default
            prompt = node.config.get("prompt", f"You are the {node.name} agent.")
            
            # Create the react agent
            nodes[node.name] = create_react_agent(
                model=node.config["model"],
                tools=node_tools,
                prompt=prompt,
                name=node.name
            )
    
    # Create the graph structure
    graph = StateGraph(GraphState)
    
    # Add all nodes to the graph
    for node_name, node_handler in nodes.items():
        graph.add_node(node_name, node_handler)
    
    # Connect the entry point node to START and END
    try:
        entry_node = next(node for node in config.nodes if node.is_entry_point)
        graph.add_edge(START, entry_node.name)
        # Entry point is also connected to END
        graph.add_edge(entry_node.name, END)
    except StopIteration:
        print("Warning: No entry point node specified in configuration")
    except Exception as e:
        print(f"Error adding entry point edges: {e}")
    
    # Add all other edges defined in the configuration
    for edge in config.edges:
        try:
            graph.add_edge(edge.from_, edge.to)
        except Exception as e:
            print(f"Error adding edge from '{edge.from_}' to '{edge.to}': {e}")
    
    # Compile and return the graph
    return graph.compile(store=InMemoryStore()) # TODO: Use proper store



class GraphState(TypedDict):
    """State for the graph execution."""
    messages: Annotated[list[AnyMessage], add_messages]
    config: GraphConfig
    initialized_node_ids: Dict[str, str]  # Maps node names to registry IDs

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

# Global registry for nodes - using Any for the value type
node_registry: Dict[str, Any] = {}

# Function to create a planner node
def create_planner_node(config_id: str, version_id: str) -> Any:
    """Create a new planner node with the specified configuration."""
    # Here you would use config_id and version_id to determine how to configure the node
    # For simplicity, we're creating a basic node that just echoes messages
    node = (
        StateGraph(MessagesState)
        .add_node('echo', planner_node_handler)
        .add_edge(START, "echo")
        .add_edge("echo", END)
        .compile(store=InMemoryStore())
    )
    return node

# Function to get or create a node
def get_or_create_node(config_id: str, version_id: str) -> str:
    """Get or create a node and return its registry ID."""
    node_id = f"{config_id}:{version_id}"
    if node_id not in node_registry:
        node_registry[node_id] = create_planner_node(config_id, version_id)
    return node_id

# Handler for the planner node
def planner_node_handler(state: Annotated[MessagesState, InjectedState]):
    """A simple handler that just echoes the last message."""
    messages = state["messages"]
        
    echo_response = {"role": "assistant", "content": f"<agent_config>{graph_config}</agent_config>"}
    return {"messages": messages + [echo_response]}


class PsiState(TypedDict):
    messages: List[Any]  # Simple list of any message type
    planner_node_id: str  # Store ID instead of actual graph object
    initialized_node_ids: Dict[Tuple[str, str], str]  # Store IDs, not objects

def task_handler(state: Annotated[PsiState, InjectedState]):
    """A simple agent that processes messages using a planner node from the registry."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Get or create planner node if not already in state
    if "planner_node_id" not in state or not state["planner_node_id"]:
        config_id = "default"
        version_id = "v1"
        planner_node_id = get_or_create_node(config_id, version_id)
        
        # Initialize the node registry dictionary if not present
        initialized_node_ids = state.get("initialized_node_ids", {})
        initialized_node_ids[(config_id, version_id)] = planner_node_id
        
        updated_state = {
            "messages": messages,
            "planner_node_id": planner_node_id,
            "initialized_node_ids": initialized_node_ids
        }
    else:
        updated_state = state
    
    # Get the planner node from registry
    planner_node_id = updated_state["planner_node_id"]
    planner = node_registry[planner_node_id]
    
    # Process the message with the planner
    planner_result = planner.invoke({"messages": [last_message]})
    
    # From debug output, we see the structure is {'messages': [...]}
    print("DEBUG - Planner Result Structure:", planner_result)
    
    # Get the last message from the result
    planner_response = planner_result["messages"][-1]
    
    # Handle both dictionary and Message object types for the response
    if hasattr(planner_response, "content"):
        # Use regex to extract the config from the response
        match = re.search(r'<agent_config>(.*?)</agent_config>', planner_response.content, re.DOTALL)
        if match:
            config = match.group(1)
        else:
            print(f"Warning: Could not find agent_config in response: {planner_response.content[:100]}...")
            # Use default config or the graph_config variable as a fallback
            config = graph_config
    else:
        match = re.search(r'<agent_config>(.*?)</agent_config>', planner_response["content"], re.DOTALL)
        if match:
            config = match.group(1)
        else:
            print(f"Warning: Could not find agent_config in response: {planner_response['content'][:100]}...")
            # Use default config or the graph_config variable as a fallback
            config = graph_config

    # Add a response to the message history
    response = {"role": "assistant", "content": "I'm analyzing your request..."}
    
    try:
        parsed_config = parse_graph_config(config)
        print("Successfully parsed the agent configuration")

        # Create the graph
        graph = create_graph(parsed_config)
        result = graph.invoke({"messages": [last_message]})
        response = {
            "role": "assistant",
            "content": result["messages"][-1].content
        }

        print("Result:", response)
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        response = {
            "role": "assistant",
            "content": "I processed your request but encountered issues with my agent configuration system."
        }
    
    # Return updated state with our response
    updated_state["messages"] = messages + [response]
    return updated_state


store = InMemoryStore()

# Redefining the supervisor graph with a clear structure
supervisor = (
    StateGraph(PsiState)
    .add_node('task_handler', task_handler)
    .add_edge(START, "task_handler")
    .add_edge("task_handler", END)
    .compile(store=InMemoryStore())
)

# Initial state with empty initialized_node_ids dictionary
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
        }
    ],
    "initialized_node_ids": {}
}

# Run the supervisor directly instead of streaming
print("\nRunning the supervisor agent:")
try:
    # Note: supervisor.invoke() returns the final state directly, not wrapped in node output
    final_state = supervisor.invoke(initial_state)
    
    print(f"Final state keys: {list(final_state.keys())}")
    
    # Print final messages in a readable format
    if "messages" in final_state:
        print("\nFinal Message History:")
        for message in final_state["messages"]:
            if isinstance(message, dict):
                print(f"Role: {message.get('role', 'unknown')}")
                print(f"Content: {message.get('content', '')}")
            else:
                # Try pretty_print if available, otherwise fall back to str representation
                try:
                    message.pretty_print()
                except AttributeError:
                    print(str(message))
            print("-" * 40)
    else:
        print("No messages found in the final state")
    
    # Print planner node info
    print("\nState Management Information:")
    print(f"Planner Node ID: {final_state.get('planner_node_id', 'None')}")
    print(f"Initialized Node IDs: {final_state.get('initialized_node_ids', {})}")
except Exception as e:
    print(f"Error running the supervisor: {e}")

# Print the registry size for verification
print(f"Node Registry Size: {len(node_registry)} node(s)")
for node_id in node_registry:
    print(f"  - {node_id}")