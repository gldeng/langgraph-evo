"""Tool definitions and creation functions."""
from typing import Annotated, Dict, Any, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_core.messages import ToolMessage


def name_of_transfer_tool(agent_name: str):
    raw_name = f"transfer_to_{agent_name}"
    # Replace any non-alphanumeric characters with underscores
    return ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in raw_name)

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Create a tool that transfers control to another agent.
    
    Args:
        agent_name: The name of the agent to transfer to
        description: Optional custom description for the tool
        
    Returns:
        A handoff tool function
    """
    name = name_of_transfer_tool(agent_name)
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool

# Utility tool functions
def add(a: float, b: float):
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b 