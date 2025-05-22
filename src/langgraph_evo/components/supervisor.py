from langgraph_evo.components.tools import create_handoff_tool
from langgraph_evo.core.registry import SUPERVISOR_NODE_ID, node_registry
from langgraph_evo.core.store import BaseStore
from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState


def create_supervisor(children_nodes: list[str], store: BaseStore):
    """Create a supervisor agent that can manage multiple agents."""


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

    prompt = f"""
    You are a helpful assistant performing a task that requires managing multiple agents.
    Analyze the user's request and decide the steps to take to complete the task.
    If you need to use an agent, use the handoff tool to delegate the task to the appropriate agent.
    If you are not able to complete the task with the available agents, just reply with "I'm sorry, I'm not able to complete the task."
    Below is the list of agents and their descriptions:
    {children_nodes_str}
    """

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
        # always return back to the supervisor
        .add_edge("__supervisor", "__supervisor")
    )

    for child_node in valid_children_nodes:
        graph.add_node(node_registry[child_node][1])
        graph.add_edge(child_node, "__supervisor")



    return graph.compile()
