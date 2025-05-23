
from langgraph_evo.core.registry import PLANNER_NODE_ID, SUPERVISOR_NODE_ID, _get_or_create_node, node_registry
from langgraph_evo.core.store import BaseStore
from langgraph_evo.core.state import GraphState
from langgraph_evo.components.handlers import task_handler_wrapped
from langgraph_evo.core.config import parse_graph_config
from langgraph_evo.core.builder import create_graph
from langgraph_evo.components.supervisor import create_supervisor
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import Dict, Optional, Union, Any
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.utils.runnable import RunnableCallable
import re


def create_psi_graph(store: BaseStore):

    from langgraph_evo.core.tool_registry import register_standard_tools
    register_standard_tools()

    psi = (
        StateGraph(GraphState)
        .add_node('task_handler', task_handler_wrapped)
        .add_edge(START, "task_handler")
        .add_edge("task_handler", END)
        .compile(store=store)
    )
    return psi


class PsiGraph(StateGraph):
    def __init__(self):
        # Initialize StateGraph with the state schema
        super().__init__(state_schema=GraphState)
        from langgraph_evo.core.tool_registry import register_standard_tools
        register_standard_tools()
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
            planner_node_id = _get_or_create_node(store)
            
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
            planner_result = planner.invoke({"messages": messages})

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
            graph = create_graph(config_name, config_version, parsed_config, store)
            updated_state["initialized_node_ids"].add(graph.name)

            node_registry[graph.name] = (config_description, graph)
            supervisor_graph = create_supervisor([name for name in updated_state["initialized_node_ids"] if not name.startswith("__")], store)
            node_registry[SUPERVISOR_NODE_ID] = ("__supervisor", supervisor_graph)

            # Pass the full message history to the graph for proper context in follow-up questions
            result = supervisor_graph.invoke({"messages": messages})
            response = AIMessage(content=result["messages"][-1].content)

            print("Result:", response)
        except Exception as e:
            print(f"Error creating/running agent: {e}")
            response = AIMessage(content="I processed your request but encountered issues with my agent configuration system.")
        
        # Return updated state with our response
        updated_state["messages"] = messages + [response]

        return updated_state

