"""Base agent implementations for LangGraph Evolution System.

This module provides base agent classes and interfaces for the LangGraph Evolution System.
"""

from typing import Any, Dict, List, Optional, Protocol, Union
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool

# Set up logging
logger = logging.getLogger(__name__)


class AgentOutput(Protocol):
    """Protocol for agent outputs."""
    output: str


class BaseAgent(Protocol):
    """Base agent protocol for LangGraph Evolution System.
    
    This defines the interface that all agents in the system must implement.
    """
    
    def invoke(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Invoke the agent with the given input data.
        
        Args:
            input_data: Input data for the agent, including 'input' and optional 'history'
            
        Returns:
            An AgentOutput object containing the agent's response
        """
        ...


class ReactAgent:
    """A basic ReAct pattern agent implementation.
    
    This agent uses the ReAct pattern (Reason, Act) for handling tasks.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        system_message: Optional[str] = None
    ):
        """Initialize the ReactAgent.
        
        Args:
            model: The LLM to use for the agent
            tools: Optional tools available to the agent
            system_message: Optional system message for the agent
        """
        self.model = model
        self.tools = tools or []
        self.system_message = system_message or (
            "You are a helpful AI assistant that solves tasks step-by-step. "
            "Think carefully about the problem, break it down, and provide detailed solutions."
        )
        logger.info(f"Initialized ReactAgent with {len(self.tools)} tools")
    
    def invoke(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Invoke the agent with the given input data.
        
        Args:
            input_data: Input data for the agent, including 'input' and optional 'history'
            
        Returns:
            An AgentOutput object containing the agent's response
        """
        input_text = input_data.get("input", "")
        history = input_data.get("history", [])
        
        # Create the messages list
        messages = []
        
        # Add system message
        messages.append(SystemMessage(content=self.system_message))
        
        # Add history
        messages.extend(history)
        
        # Add the current input as a human message
        messages.append(HumanMessage(content=input_text))
        
        # Invoke the model
        try:
            response = self.model.invoke(messages)
            return AgentOutput(output=response.content)
        except Exception as e:
            logger.error(f"Error invoking ReactAgent: {e}")
            return AgentOutput(output=f"Error: {str(e)}")


class ToolUsingAgent:
    """An agent that can use tools to solve tasks.
    
    This agent has access to tools and can decide when to use them.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        tools: List[BaseTool],
        system_message: Optional[str] = None,
        max_iterations: int = 10
    ):
        """Initialize the ToolUsingAgent.
        
        Args:
            model: The LLM to use for the agent
            tools: Tools available to the agent
            system_message: Optional system message for the agent
            max_iterations: Maximum number of tool-using iterations
        """
        self.model = model
        self.tools = tools
        self.system_message = system_message or (
            "You are a helpful AI assistant with access to tools. "
            "Use these tools appropriately to solve tasks."
        )
        self.max_iterations = max_iterations
        
        # Create tool descriptions
        self.tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools
        )
        
        # Create tool map for lookup
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        logger.info(f"Initialized ToolUsingAgent with {len(self.tools)} tools")
    
    def _extract_tool_calls(self, message_content: str) -> List[Dict[str, str]]:
        """Extract tool calls from message content.
        
        This is a simple extraction based on text patterns. In a real system,
        this would use more robust parsing based on the model's structured output format.
        
        Args:
            message_content: The message content to parse
            
        Returns:
            List of tool calls, each with 'name' and 'arguments' keys
        """
        tool_calls = []
        
        # Very simple parsing - in a real system this would be more robust
        content = message_content.lower()
        for tool_name in self.tool_map:
            if f"use {tool_name}" in content or f"use the {tool_name}" in content:
                # Extract arguments - this is a simplistic approach
                start_idx = content.find(tool_name) + len(tool_name)
                end_idx = content.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = len(content)
                    
                args_text = content[start_idx:end_idx].strip()
                
                tool_calls.append({
                    "name": tool_name,
                    "arguments": args_text
                })
        
        return tool_calls
    
    def invoke(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Invoke the agent with the given input data.
        
        Args:
            input_data: Input data for the agent, including 'input' and optional 'history'
            
        Returns:
            An AgentOutput object containing the agent's response
        """
        input_text = input_data.get("input", "")
        history = input_data.get("history", [])
        
        # Create the initial messages list
        messages = []
        
        # Add system message with tool descriptions
        full_system_message = (
            f"{self.system_message}\n\n"
            f"You have access to the following tools:\n{self.tool_descriptions}\n\n"
            "To use a tool, clearly state 'I'll use the TOOL_NAME' followed by the necessary arguments."
        )
        messages.append(SystemMessage(content=full_system_message))
        
        # Add history
        messages.extend(history)
        
        # Add the current input as a human message
        messages.append(HumanMessage(content=input_text))
        
        # Conversation tracking
        conversation = []
        
        # Tool use loop
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            
            try:
                # Invoke the model
                response = self.model.invoke(messages)
                conversation.append(response)
                
                # Check for tool calls
                tool_calls = self._extract_tool_calls(response.content)
                
                if not tool_calls:
                    # No tool calls, return the response
                    return AgentOutput(output=response.content)
                
                # Process tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    if tool_name in self.tool_map:
                        tool = self.tool_map[tool_name]
                        
                        # Execute the tool (in a real system, this would properly parse arguments)
                        try:
                            tool_result = tool.invoke(tool_args)
                            
                            # Add tool result to messages
                            messages.append(AIMessage(content=response.content))
                            messages.append(
                                HumanMessage(
                                    content=f"Tool {tool_name} returned: {tool_result}"
                                )
                            )
                            
                            # Add to conversation
                            conversation.append(
                                HumanMessage(
                                    content=f"Tool {tool_name} returned: {tool_result}"
                                )
                            )
                        except Exception as e:
                            # Add error message
                            messages.append(AIMessage(content=response.content))
                            messages.append(
                                HumanMessage(
                                    content=f"Error using tool {tool_name}: {str(e)}"
                                )
                            )
                            
                            # Add to conversation
                            conversation.append(
                                HumanMessage(
                                    content=f"Error using tool {tool_name}: {str(e)}"
                                )
                            )
            except Exception as e:
                logger.error(f"Error in ToolUsingAgent: {e}")
                return AgentOutput(output=f"Error: {str(e)}")
        
        # If we've reached the maximum iterations, return a summary of the conversation
        final_output = "I've reached the maximum number of iterations. Here's a summary of what I found:\n\n"
        for msg in conversation:
            if isinstance(msg, AIMessage):
                final_output += f"{msg.content}\n\n"
        
        return AgentOutput(output=final_output) 