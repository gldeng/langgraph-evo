
from langgraph_evo.core.store import BaseStore
from langgraph_evo.core.state import PsiState
from langgraph_evo.components.handlers import task_handler_wrapped
from langgraph.graph import StateGraph, START, END


def create_psi_graph(store: BaseStore):

    from langgraph_evo.core.tool_registry import register_standard_tools
    register_standard_tools()

    psi = (
        StateGraph(PsiState)
        .add_node('task_handler', task_handler_wrapped)
        .add_edge(START, "task_handler")
        .add_edge("task_handler", END)
        .compile(store=store)
    )
    return psi