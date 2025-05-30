from langgraph_evo.core.config import ConfigRecord, GraphConfig, parse_graph_config
from langgraph_evo.core.registry import PLANNER_NODE_ID, SUPERVISOR_NODE_ID
from langgraph_evo.core.store import BaseStore
from langgraph_evo.core.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ChatMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from typing import Annotated, Dict, List, Optional, Union, Any
from typing_extensions import Literal
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE
import re

class ParentMessage(ChatMessage):
    role: Literal["developer"] = "developer"
    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """Create a ParentMessage.

        Args:
            content: The string contents of the message.
            **kwargs: Additional fields.
        """
        super().__init__(content=content, role="developer", **kwargs)



class ChildMessage(ChatMessage):
    role: Literal["tool"] = "tool"
    child_id: str
    def __init__(
        self, content: Union[str, list[Union[str, dict]]], child_id: str, **kwargs: Any
    ) -> None:
        """Create a ChildMessage.

        Args:
            content: The string contents of the message.
            child_id: The ID of the child node that sent the message.
            **kwargs: Additional fields.
        """
        super().__init__(content=content, role="tool", child_id=child_id, **kwargs)


node_registry: Dict[str, CompiledGraph] = {}

class PlannerMixin:
    SYSTEM_PROMPT = """You are an AI agent configurator that helps set up and run AI agent systems.
    
    When a user asks a question:
    1. Retrieve all the agent configurations using the get_configs tool
    2. For each configuration, analyze whether this configuration can solve the user's query
    3. If it can, return the configuration string along with other information

    IMPORTANT: The configuration string MUST be valid YAML format. Follow these YAML rules:
    - Use proper indentation (2 spaces per level)
    - Quote string values that contain special characters, colons, or newlines
    - Ensure all mapping keys end with a colon followed by a space
    - Use proper list syntax with dashes
    - Escape any special characters in string values

    Output format:
    <agent_config>
    Valid YAML configuration string here
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

        # Add nodes for the new workflow
        self.add_node("supervisor", self._create_supervisor_node())
        self.add_node("planner", self._create_planner_node())
        self.add_node("finalize", self._create_finalize_node())
        
        # Set entry point to supervisor
        self.add_edge(START, "supervisor")
        
        # Add conditional edges from supervisor
        self.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "finalize_success": "finalize",
                "first_attempt_failed": "planner", 
                "multiple_attempts_failed": "finalize"
            }
        )
        
        # Add conditional edges from planner
        self.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "planner_success": "supervisor",
                "planner_failed": "finalize"
            }
        )
        
        # Finalize node goes to END
        self.add_edge("finalize", END)

    def _create_supervisor_node(self) -> RunnableCallable:
        """Private method to create the supervisor node."""
        return RunnableCallable(PsiGraph.supervisor_handler)

    def _create_planner_node(self) -> RunnableCallable:
        """Private method to create the planner node."""
        return RunnableCallable(PsiGraph.planner_handler)

    def _create_finalize_node(self) -> RunnableCallable:
        """Private method to create the finalizing node."""
        return RunnableCallable(PsiGraph.finalize_handler)

    @staticmethod
    def _get_call_path(config: Dict[str, Any]) -> tuple[str, ...]:
        config = PsiGraph._ensure_call_path(config)
        return config['metadata']['call_path']

    @staticmethod
    def _ensure_call_path(config: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure metadata exists with call_path
        if 'metadata' not in config:
            config['metadata'] = {
                'call_path': ()
            }
        elif 'call_path' not in config['metadata']:
            config['metadata']['call_path'] = ()

        return config

    @staticmethod
    def _extend_call_path(config: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        config = PsiGraph._ensure_call_path(config)
        # Ensure metadata exists with call_path
        config['metadata']['call_path'] = config['metadata']['call_path'] + (node_id,)
        return config

    @staticmethod
    def _get_parent_state(state: GraphState, config: Dict[str, Any]) -> GraphState:
        config = PsiGraph._ensure_call_path(config)
        call_path = config['metadata']['call_path']
        if len(call_path) < 1:
            raise ValueError("Call path is too short to get parent state")
        parent_call_path = call_path[:-1]
        if parent_call_path == ():
            return state
        return state["children_states"][parent_call_path]

    @staticmethod
    def _get_sub_state(state: GraphState, config: Dict[str, Any]) -> GraphState:
        config = PsiGraph._ensure_call_path(config)
        call_path = config['metadata']['call_path']
        return state["children_states"].get(call_path, {"messages": []})

    @staticmethod
    def _format_message(message: BaseMessage) -> str:
        if isinstance(message, ChildMessage):
            return f"Child Agent [{message.child_id}]: {message.content}"
        elif isinstance(message, ParentMessage):
            return f"Parent Agent: {message.content}"
        elif isinstance(message, HumanMessage):
            return f"User: {message.content}"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def _get_relevant_messages(state: GraphState) -> str:
        messages = []
        for message in reversed(state["messages"]):
            if isinstance(message, ChildMessage):
                messages.append(message)
            elif isinstance(message, ParentMessage):
                messages.append(message)
                break
            elif isinstance(message, HumanMessage):
                messages.append(message)
                break
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        return "\n".join([PsiGraph._format_message(message) for message in reversed(messages)])

    @staticmethod
    def _route_from_supervisor(state: GraphState) -> Literal["finalize_success", "first_attempt_failed", "multiple_attempts_failed"]:
        """Route from supervisor node based on success and attempt count."""
        if state.get("supervisor_success", False):
            return "finalize_success"
        
        attempt_count = state.get("attempt_count", 0)
        if attempt_count <= 1:
            return "first_attempt_failed"
        else:
            return "multiple_attempts_failed"

    @staticmethod 
    def _route_from_planner(state: GraphState) -> Literal["planner_success", "planner_failed"]:
        """Route from planner node based on success."""
        if state.get("planner_success", False):
            return "planner_success"
        else:
            return "planner_failed"

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
    def _supervisor_can_handle_task(state: GraphState, store: BaseStore, config: Dict[str, Any]):
        """Check if the supervisor can handle the task."""

        parent_state = PsiGraph._get_parent_state(state, config)
        sub_state = PsiGraph._get_sub_state(state, config)

        sub_state["messages"].append(ParentMessage(content=f"I have received the following messages from the parent agent:\n\n{PsiGraph._get_relevant_messages(parent_state)}"))

        record = node_registry.get(SUPERVISOR_NODE_ID)
        if not record:
            return False, {}
        _, supervisor_graph = record
        # Check if the supervisor can handle the task
        result = supervisor_graph.invoke(sub_state)

        updated_state = {}
        if (call_path:= PsiGraph._get_call_path(config)) == (): # root node
            updated_state = {
                "messages": [ChildMessage(content=result["messages"][-1].content, child_id=SUPERVISOR_NODE_ID)]
            }
        elif len(call_path) == 1:
            updated_state = {
                "messages": [ChildMessage(content=result["messages"][-1].content, child_id=SUPERVISOR_NODE_ID)],
                "children_states": {
                    call_path: result
                }
            }
        else:
            # non-root node
            updated_state = {
                "children_states": {
                    call_path[:-1]: {
                        "messages": [ChildMessage(content=result["messages"][-1].content, child_id=SUPERVISOR_NODE_ID)]
                    },
                    call_path: result
                }
            }

        if "I'm not able to complete the task." in result["messages"][-1].content:
            return False, updated_state
        return True, updated_state




    @staticmethod
    def supervisor_handler(
        state: GraphState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ):
        """Supervisor node handler that attempts to process user questions.
        
        Args:
            state: The current state
            store: The store to use
            config: Optional configuration parameters
            
        Returns:
            Updated state with supervisor results and success status
        """
        config = PsiGraph._extend_call_path(config, "supervisor")

        # Initialize attempt count if not present
        attempt_count = state.get("attempt_count", 0) + 1
        
        # Try to handle task with existing supervisor
        can_handle_task, updated_state = PsiGraph._supervisor_can_handle_task(state, store, config)
        
        updated_state.update(
            attempt_count=attempt_count,
            supervisor_success=can_handle_task                
        )
        
        return updated_state

    @staticmethod
    def planner_handler(
        state: GraphState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ):
        """Planner node handler that gets agent config and creates/updates supervisor.
        
        Args:
            state: The current state
            store: The store to use
            config: Optional configuration parameters
            
        Returns:
            Updated state with planner results and success status
        """

        config = PsiGraph._extend_call_path(config, "planner")

        parent_state = PsiGraph._get_parent_state(state, config)
        sub_state = PsiGraph._get_sub_state(state, config)

        sub_state["messages"].append(ParentMessage(content=f"I have received the following messages from the parent agent:\n\n{PsiGraph._get_relevant_messages(parent_state)}"))

        planner_success = False

        updated_state = {
            "messages": [],
            "planner_node_id": None,
            "initialized_node_ids": set(),
            "planner_success": False,
            "children_states": {
                PsiGraph._get_call_path(config): sub_state
            }
        }
        
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
            
            updated_state["planner_node_id"] = planner_node_id
            updated_state["initialized_node_ids"] = initialized_node_ids
        else:
            updated_state["planner_node_id"] = state["planner_node_id"]
            updated_state["initialized_node_ids"] = state["initialized_node_ids"]
        
        # Get the planner node from registry
        planner_node_id = updated_state["planner_node_id"]
        planner = node_registry[planner_node_id][1]
        
        try:
            # Process with the planner, passing the full conversation history
            planner_result = planner.invoke(sub_state, debug=True)
            planner_response = planner_result["messages"][-1]
            content = planner_response.content if hasattr(planner_response, "content") else planner_response["content"]
            updated_state["children_states"][PsiGraph._get_call_path(config)]= planner_result
        except (AttributeError, TypeError, KeyError) as e:
            print(f"Warning: Could not extract content from planner response: {e}")
            content = ""

        # Search for agent configuration
        match = re.search(r'<agent_config>(.*?)</agent_config>', content, re.DOTALL)
        if match:
            agent_config_str = match.group(1).strip()
            print(f"Extracted agent config:\n{agent_config_str}")
        else:
            print(f"Warning: Could not find agent_config in response: {content[:100]}...")
            response = ChildMessage(content="Sorry, I'm unable to create an agent to perform the task.", child_id=PLANNER_NODE_ID)
            updated_state["messages"].append(response)
            updated_state["planner_success"] = False
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

        try:
            parsed_config = parse_graph_config(agent_config_str)
            print("Successfully parsed the agent configuration")

            # Create the graph using the registry
            graph_raw = PsiGraph._create_graph(parsed_config)
            graph = graph_raw.compile(name=f"{config_name}__{config_version}", store=store)
            updated_state["initialized_node_ids"].add(graph.name)

            node_registry[graph.name] = (config_description, graph)
            supervisor_graph = PsiGraph._create_supervisor([name for name in updated_state["initialized_node_ids"] if not name.startswith("__")], store)
            node_registry[SUPERVISOR_NODE_ID] = ("__supervisor", supervisor_graph)

            planner_success = True
            response = ChildMessage(content="I've successfully set up the agents and am ready to process your request.", child_id=PLANNER_NODE_ID)
            print("Planner succeeded in creating supervisor")
        except Exception as e:
            print(f"Error creating/running agent: {e}")
            print(f"Failed YAML content:\n{agent_config_str}")
            response = ChildMessage(content=f"I encountered a configuration error: {str(e)}. Please check the agent configuration format.", child_id=PLANNER_NODE_ID)
            planner_success = False
        
        updated_state["messages"].append(response)
        updated_state["planner_success"] = planner_success
        
        # Preserve other state fields
        for key in ["attempt_count", "supervisor_success"]:
            if key in state:
                updated_state[key] = state[key]
        
        return updated_state

    @staticmethod
    def finalize_handler(
        state: GraphState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ):
        """Finalize node handler that returns final answers.
        
        Args:
            state: The current state
            store: The store to use
            config: Optional configuration parameters
            
        Returns:
            Final state with appropriate messages
        """
        messages = state["messages"]
        attempt_count = state.get("attempt_count", 0)
        supervisor_success = state.get("supervisor_success", False)
        
        # If we reach finalize after multiple failed attempts, add error message
        if attempt_count > 1 and not supervisor_success:
            error_message = AIMessage(content="I'm sorry, I'm not able to complete the task.")
            messages = messages + [error_message]
            print("Finalize: Multiple attempts failed, returning error message")
        else:
            print(f"Finalize: Returning final answer (supervisor_success: {supervisor_success})")
        
        # Return final state
        final_state = dict(state)
        final_state["messages"] = messages
        return final_state

