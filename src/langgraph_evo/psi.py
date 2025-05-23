from langgraph_evo.core.config import ConfigRecord, GraphConfig, parse_graph_config
from langgraph_evo.core.registry import PLANNER_NODE_ID, SUPERVISOR_NODE_ID
from langgraph_evo.core.store import BaseStore
from langgraph_evo.core.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from typing import Annotated, Dict, List, Optional, Union, Any
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE
import re

node_registry: Dict[str, CompiledGraph] = {}

class PlannerMixin:
    SYSTEM_PROMPT = """You are an AI agent configurator that helps set up and run AI agent systems.
    
    When a user asks a question:
    1. Retrieve all the agent configurations using the get_configs tool
    2. For each configuration, analyze whether this configuration can solve the user's query
    3. If it can, return the configuration string along with other information

    Output format:
    - If it can, return the configuration infromation with the following format:
    <agent_config>
    Config string here
    </agent_config>
    <config_name>
    Config name here
    </config_name>
    <config_version>
    Config version here
    </config_version>
    <config_description>
    Config description here
    </config_description>
    - If it cannot, return "No configuration found"

    Example output:
    <agent_config>
    Config string here
    </agent_config>
    <config_name>
    Config name here
    </config_name>
    <config_version>
    Config version here
    </config_version>
    <config_description>
    Config description here
    </config_description>

    Only do the following but not more than that:
    - Retrieve the agent configurations
    - Analyze whether each configuration can solve the user's query
    - Return the configuration information if it can, otherwise return "No configuration found"
    """

    @staticmethod
    def _get_or_create_planner_node(store) -> str:
        """Get or create a node and return its registry ID.
        
        Args:
            store: The store to use for node creation
            
        Returns:
            str: The registry ID for the node
        """
        
        node_id = PLANNER_NODE_ID
        if node_id not in node_registry:
            node_registry[node_id] = ("planner", PlannerMixin._create_planner_node(store))
        return node_id

    @staticmethod
    def _create_planner_node(store: BaseStore, model: str = "openai:gpt-4"):
        """Create a new planner node with the React agent."""

        @tool
        def get_configs(
            store: Annotated[BaseStore, InjectedStore]
        ) -> List[ConfigRecord]:
            """Retrieve the agent configuration.
            
            This tool retrieves all agent configurations from the store.
            Output is a list of ConfigRecord objects.
            """
            configs = store.search(AGENT_CONFIGS_NAMESPACE)
            if configs:
                return list(map(lambda c: c.value, configs))
            return []

        # Create the React agent
        react_agent = create_react_agent(
            model=model,
            tools=[get_configs],
            prompt=PlannerMixin.SYSTEM_PROMPT
        )
        
        # Create the node using the injected store and React agent
        node = (
            StateGraph(MessagesState)
            .add_node('react_planner', react_agent)
            .add_edge(START, "react_planner")
            .add_edge("react_planner", END)
            .compile(store=store)
        )
        return node

class GraphCreatorMixin:

    @staticmethod
    def _create_graph(config: GraphConfig) -> StateGraph:
        """Create a graph based on the configuration.
        
        Args:
            config: The graph configuration to use
            
        Returns:
            A compiled graph that can be executed
        """
        # Import here to avoid circular imports
        from langgraph_evo.components.tools import create_handoff_tool, name_of_transfer_tool
        from langgraph.prebuilt import create_react_agent
        from langgraph_evo.core.tool_registry import get_tool, has_tool, resolve_tool
        
        # Create a mapping of tool names to actual tool objects
        tool_map = {}
        if config.edges is None:
            config.edges = []
        
        # Process tool configurations
        if config.tools:
            for tool_config in config.tools:
                if hasattr(tool_config, 'name') and hasattr(tool_config, 'script'):
                    # Skip if already in map
                    if tool_config.name in tool_map:
                        continue
                        
                    # Try to get from registry or resolve
                    tool_obj = None
                    
                    # First try the registry
                    if has_tool(tool_config.name):
                        tool_obj = get_tool(tool_config.name)
                        
                    # Then try to resolve by script name
                    if tool_obj is None and tool_config.script:
                        tool_obj = resolve_tool(tool_config.script)
                        
                    if tool_obj is not None:
                        tool_map[tool_config.name] = tool_obj
                    else:
                        print(f"Warning: Could not resolve tool '{tool_config.name}' with script '{tool_config.script}'")
        
        # Create handoff tools for node transitions
        handoff_tools = {}
        for edge in config.edges:
            tool_name = name_of_transfer_tool(edge.to)
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
                    if edge.from_ == node.name and name_of_transfer_tool(edge.to) in handoff_tools:
                        node_tools.append(handoff_tools[name_of_transfer_tool(edge.to)])
                
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
        
        return graph


class SupervisorMixin:

    SUPERVISOR_PROMPT_TEMPLATE = """
    You are a helpful assistant performing a task that requires managing multiple agents.
    Analyze the user's request and decide the steps to take to complete the task.
    If you need to use an agent, use the handoff tool to delegate the task to the appropriate agent.
    If you are not able to complete the task with the available agents, just reply with "I'm sorry, I'm not able to complete the task."

    TERMINATION CRITERIA:
    1. When the user's question has been FULLY answered with all requested information
    2. When all calculations, research, or data gathering needed for the response are complete
    3. When you've provided a final, comprehensive response that requires no further agent work
    4. When an agent has provided a complete response and no other agent input is needed

    INSTRUCTIONS:
    - When managing agents, evaluate whether their responses fully address the user's question
    - After receiving an agent response, determine if:
    a) Another agent needs to process the information (use handoff)
    b) The response needs clarification (ask the same agent again)
    c) The task is complete (provide final answer to user)
    - Once you determine the task is complete, provide a final conclusive response and DO NOT use any more handoff tools
    - For multi-part questions, ensure ALL parts are addressed before considering the task complete

    Note: You are not allowed to use the handoff tool to transfer to yourself. You don't perform any tasks, you only manage the agents.

    Below is the list of agents and their descriptions:
    {children_nodes_str}
    """
    @staticmethod
    def _create_supervisor(children_nodes: list[str], store: BaseStore):
        """Create a supervisor agent that can manage multiple agents."""

        from langgraph_evo.components.tools import create_handoff_tool

        handoff_tools = []

        valid_children_nodes = [node for node in children_nodes if not node.startswith("__")]

        children_node_descriptions = []

        for child_node in valid_children_nodes:
            description, node = node_registry[child_node]
            handoff_tools.append(create_handoff_tool(
                agent_name=node.name,
                description=description,
            ))
            children_node_descriptions.append(f"- {node.name}: {description}")

        children_nodes_str = "\n".join(children_node_descriptions)

        prompt = SupervisorMixin.SUPERVISOR_PROMPT_TEMPLATE.format(children_nodes_str=children_nodes_str)

        supervisor_agent = create_react_agent(
            model="openai:gpt-4",
            tools=handoff_tools,
            prompt=prompt,
            name="__supervisor",
        )

        destinations = tuple(valid_children_nodes) + (END,)

        graph = (
            StateGraph(MessagesState)
            # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
            .add_node(supervisor_agent, destinations=destinations)
            .add_edge(START, "__supervisor")
            .add_edge("__supervisor", END)
        )

        for child_node in valid_children_nodes:
            graph.add_node(node_registry[child_node][1])
            graph.add_edge(child_node, "__supervisor")



        return graph.compile(name="supervisor", store=store)


class PsiGraph(StateGraph, PlannerMixin, GraphCreatorMixin, SupervisorMixin):
    def __init__(self):
        # Initialize StateGraph with the state schema
        super().__init__(state_schema=GraphState)
        from langgraph_evo.core.tool_registry import register_standard_tools
        register_standard_tools()

        # Flow:
        # 1. Attempts to process the user's question in supervisor node
        #       - If successful, return the final answer (finalize node)
        #       - If not successful and it's the first attempt, continue to planner node
        #       - If not successful and it's not the first attempt, return the final answer (finalize node), reply "I'm sorry, I'm not able to complete the task."
        # 2. Planner node:
        #       - Get the agent config
        #       - Create the graph
        #       - Create/update the supervisor
        #       - If successful, go to supervisor node
        #       - If not successful, go to finalize node
        # 3. Finalize node:
        #       - Return the final answer

        # Add nodes using RunnableCallable for logic
        self.add_node("process", self._create_process_node())
        self.add_node("finalize", self._create_finalize_node())
        self.add_edge("process", "finalize")
        self.add_edge("finalize", END)
        
        # Set entry point
        self.set_entry_point("process")

    def _create_process_node(self) -> RunnableCallable:
        """Private method to create the processing node."""
        return RunnableCallable(PsiGraph.task_handler)

    def _create_finalize_node(self) -> RunnableCallable:
        """Private method to create the finalizing node."""
        return RunnableCallable(lambda state: state)

    def get_graph(self,
        checkpointer: Checkpointer = None,
        *,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        debug: bool = False,
        name: Optional[str] = None,
    ) -> "CompiledStateGraph":
        return self.compile(checkpointer=checkpointer, store=store, interrupt_before=interrupt_before, interrupt_after=interrupt_after, debug=debug, name=name)


    @staticmethod
    def _supervisor_can_handle_task(state, store, config):
        """Check if the supervisor can handle the task."""

        messages = state["messages"]
        
        updated_state = {
            "messages": messages,
        }

        record = node_registry.get(SUPERVISOR_NODE_ID)
        if not record:
            return False, {}
        _, supervisor_graph = record
        # Check if the supervisor can handle the task
        result = supervisor_graph.invoke({"messages": state["messages"]})
        response = AIMessage(content=result["messages"][-1].content)
        updated_state["messages"] = messages + [response]

        print("Result:", response)
        if "I'm not able to complete the task." in result["messages"][-1].content:
            return False, {}
        return True, updated_state


    @staticmethod
    def task_handler(
        state: GraphState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ):
        """A node handler that processes messages using the agent system.
        
        Args:
            state: The current state
            store: The store to use
            config: Optional configuration parameters
            
        Returns:
            Updated state with task results
        """

        can_handle_task, updated_state = PsiGraph._supervisor_can_handle_task(state, store, config)
        if can_handle_task:
            return updated_state

        messages = state["messages"]
        
        # Extract metadata from config if available
        metadata = {}
        if config and "metadata" in config:
            metadata = config["metadata"]
            print(f"Received metadata in config: {metadata.keys()}")
        
        # Get or create planner node if not already in state
        if "planner_node_id" not in state or not state["planner_node_id"]:
            # Initialize the node registry dictionary if not present
            planner_node_id = PsiGraph._get_or_create_planner_node(store)
            
            initialized_node_ids = state.get("initialized_node_ids", set())
            initialized_node_ids.add(PLANNER_NODE_ID)
            
            updated_state = {
                "messages": messages,
                "planner_node_id": planner_node_id,
                "initialized_node_ids": initialized_node_ids
            }
        else:
            updated_state = state
        
        # Get the planner node from registry
        planner_node_id = updated_state["planner_node_id"]
        planner = node_registry[planner_node_id][1]
        
        try:
            # Find the last user message for fallback processing if needed
            last_user_message = None
            for msg in reversed(messages):
                # Handle both dict and Message object types
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_message = msg
                    break
                elif hasattr(msg, "type") and msg.type == "human":
                    # Convert to dict format if it's a Message object
                    last_user_message = HumanMessage(content=msg.content)
                    break
            
            if not last_user_message and messages:
                # If no user message found, use the last message regardless of type
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    last_user_message = last_msg
                else:
                    last_user_message = HumanMessage(content=last_msg.content)
            
            # Process with the planner, passing the full conversation history
            # This allows the planner to understand context from previous interactions
            # which is essential for follow-up questions
            planner_result = planner.invoke({"messages": messages}, debug=True)

            planner_response = planner_result["messages"][-1]

            content = planner_response.content if hasattr(planner_response, "content") else planner_response["content"]
        except (AttributeError, TypeError, KeyError) as e:
            print(f"Warning: Could not extract content from planner response: {e}")
            content = ""

        # Search for agent configuration
        match = re.search(r'<agent_config>(.*?)</agent_config>', content, re.DOTALL)
        if match:
            agent_config_str = match.group(1)
        else:
            print(f"Warning: Could not find agent_config in response: {content[:100]}...")
            response = AIMessage(content="Sorry, I'm unable to create an agent to perform the task.")
            updated_state["messages"] = messages + [response]
            return updated_state

        # Extract additional configuration fields
        config_name_match = re.search(r'<config_name>(.*?)</config_name>', content, re.DOTALL)
        config_name = config_name_match.group(1).strip() if config_name_match else "default"
        
        config_version_match = re.search(r'<config_version>(.*?)</config_version>', content, re.DOTALL)
        config_version = config_version_match.group(1).strip() if config_version_match else "v1"
        
        config_description_match = re.search(r'<config_description>(.*?)</config_description>', content, re.DOTALL)
        config_description = config_description_match.group(1).strip() if config_description_match else ""
        
        # Print extracted config metadata
        print(f"Config Name: {config_name}")
        print(f"Config Version: {config_version}")
        print(f"Config Description: {config_description[:50]}..." if len(config_description) > 50 else f"Config Description: {config_description}")

        # Add a response to the message history
        response = AIMessage(content="I'm analyzing your request...")
        supervisor_graph = None
        
        try:
            parsed_config = parse_graph_config(agent_config_str)
            print("Successfully parsed the agent configuration")

            # Create the graph using the registry (no need to pass tools explicitly)
            graph_raw = PsiGraph._create_graph(parsed_config)
            graph = graph_raw.compile(name=f"{config_name}__{config_version}", store=store)
            updated_state["initialized_node_ids"].add(graph.name)

            node_registry[graph.name] = (config_description, graph)
            supervisor_graph = PsiGraph._create_supervisor([name for name in updated_state["initialized_node_ids"] if not name.startswith("__")], store)
            node_registry[SUPERVISOR_NODE_ID] = ("__supervisor", supervisor_graph)

            # Pass the full message history to the graph for proper context in follow-up questions
            result = supervisor_graph.invoke({"messages": messages}, debug=True)
            response = AIMessage(content=result["messages"][-1].content)

            print("Result:", response)
        except Exception as e:
            print(f"Error creating/running agent: {e}")
            response = AIMessage(content="I processed your request but encountered issues with my agent configuration system.")
        
        # Return updated state with our response
        updated_state["messages"] = messages + [response]

        return updated_state

