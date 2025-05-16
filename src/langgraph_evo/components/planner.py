"""Planner implementation for LangGraph Evolution System.

This module provides a Planner component that analyzes tasks and creates appropriate
graph configurations to handle them. The Planner uses a supervisor pattern to
coordinate multiple agents.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, cast
import json
import logging
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from ..core.state import TaskState, merge_state
from ..core.agents import BaseAgent

# Set up logging
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be included in a task graph."""
    PROCESSOR = "processor"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    EXECUTOR = "executor"


class PlannerConfig:
    """Configuration for the Planner component."""
    
    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        agent_configs: Optional[Dict[AgentType, Dict[str, Any]]] = None,
        max_iterations: int = 10,
        graph_type: str = "supervisor",
        default_tools: Optional[List[BaseTool]] = None
    ):
        """Initialize the PlannerConfig.
        
        Args:
            model: LLM to use for creating agents
            agent_configs: Configurations for different agent types
            max_iterations: Maximum number of iterations for the supervisor graph
            graph_type: Type of graph to create (supervisor, pipeline, etc.)
            default_tools: Default tools to provide to agents
        """
        self.model = model
        self.agent_configs = agent_configs or {}
        self.max_iterations = max_iterations
        self.graph_type = graph_type
        self.default_tools = default_tools or []


class Planner:
    """A planner component that analyzes tasks and creates appropriate graph configurations.
    
    The Planner takes in a task, analyzes it, and creates a graph with appropriate agents
    to handle the task efficiently.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize the Planner.
        
        Args:
            config: Optional configuration for the Planner
        """
        self.config = config or PlannerConfig()
        logger.info(f"Initialized Planner with graph type: {self.config.graph_type}")
    
    def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a task to determine the optimal graph configuration.
        
        Args:
            task: The task to analyze
            
        Returns:
            A dictionary containing analysis results, including recommended agents and graph structure
        """
        # This would use LLM to analyze the task
        # For now, return a default analysis based on task type or complexity
        task_type = task.get("task_type", "general")
        task_description = task.get("description", "")
        
        # Example analysis logic (in a real implementation, this would use LLM)
        analysis = {
            "recommended_agents": [],
            "complexity": "medium",
            "graph_type": self.config.graph_type,
        }
        
        # Simple rule-based analysis (placeholder for LLM-based analysis)
        if "research" in task_description.lower():
            analysis["recommended_agents"].append(AgentType.RESEARCHER.value)
        
        if "execute" in task_description.lower() or "run" in task_description.lower():
            analysis["recommended_agents"].append(AgentType.EXECUTOR.value)
        
        # Always include a processor agent
        if AgentType.PROCESSOR.value not in analysis["recommended_agents"]:
            analysis["recommended_agents"].append(AgentType.PROCESSOR.value)
        
        # 50% of the time, add a critic for quality control
        if "evaluate" in task_description.lower() or "review" in task_description.lower():
            analysis["recommended_agents"].append(AgentType.CRITIC.value)
        
        return analysis
    
    def create_agent(self, agent_type: AgentType, tools: Optional[List[BaseTool]] = None) -> BaseAgent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            tools: Optional tools to provide to the agent
            
        Returns:
            A configured agent of the specified type
        """
        tools = tools or self.config.default_tools
        
        # Get agent-specific configuration
        agent_config = self.config.agent_configs.get(agent_type, {})
        
        # Agent system messages based on type
        system_messages = {
            AgentType.PROCESSOR: """You are a processor agent responsible for processing and solving tasks.
You analyze the task and provide a clear solution.
Always think step by step and provide detailed solutions.""",
            
            AgentType.RESEARCHER: """You are a researcher agent responsible for gathering information.
Your goal is to find relevant information to help solve the task.
Be thorough and cite your sources.""",
            
            AgentType.CRITIC: """You are a critic agent responsible for evaluating solutions.
Carefully review the proposed solution and provide constructive feedback.
Point out potential issues and suggest improvements.""",
            
            AgentType.EXECUTOR: """You are an executor agent responsible for executing plans.
Your goal is to implement the solution efficiently and accurately.
Follow the plan exactly and report any issues encountered."""
        }
        
        # Create the agent using the system message for the specified type
        system_message = system_messages.get(agent_type, system_messages[AgentType.PROCESSOR])
        
        # If using the react pattern
        agent = create_react_agent(
            llm=self.config.model,
            tools=tools,
            system_message=system_message
        )
        
        return agent
    
    def create_supervisor_graph(self, agents: Dict[str, BaseAgent], task: Dict[str, Any]) -> StateGraph:
        """Create a supervisor graph with the specified agents.
        
        This implements a supervisor pattern where a central agent coordinates multiple worker agents.
        
        Args:
            agents: Dictionary mapping agent IDs to agent instances
            task: The task to handle
            
        Returns:
            A configured StateGraph
        """
        # Create the supervisor prompt
        supervisor_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are the supervisor agent that coordinates other agents to solve a task.
Your job is to:
1. Break down the task into subtasks
2. Assign subtasks to appropriate agents
3. Review their work and provide feedback
4. Integrate their results into a final solution
5. Determine when the task is complete

You have access to the following agents:
{agent_descriptions}

To use an agent, specify 'agent: AGENT_ID' in your response, followed by the task for that agent.
After receiving an agent's response, determine the next steps or mark the task as complete."""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="Task: {task_description}\n\nThink about how to solve this task by coordinating the available agents.")
        ])
        
        # Define the graph state
        class SupervisorState(TaskState):
            """State for the supervisor graph."""
            
            # Add any supervisor-specific state fields
            current_agent: Optional[str] = None
            subtasks: List[Dict[str, Any]] = []
            completed_subtasks: List[Dict[str, Any]] = []
            
        # Create the graph
        graph = StateGraph(SupervisorState)
        
        # Add supervisor node
        def _run_supervisor(state: SupervisorState) -> Dict[str, Any]:
            """Run the supervisor agent to coordinate other agents."""
            history = state.history or []
            
            # Create agent descriptions for the prompt
            agent_descriptions = "\n".join([
                f"- {agent_id.upper()}: {agent.__class__.__name__}" 
                for agent_id, agent in agents.items()
            ])
            
            # Run the supervisor
            response = self.config.model.invoke(
                supervisor_prompt.format(
                    agent_descriptions=agent_descriptions,
                    task_description=task.get("description", ""),
                    history=history
                )
            )
            
            # Add supervisor's response to history
            updated_state = {"history": history + [response]}
            
            # Parse the response to extract agent assignment
            # This is a simplistic version; a real implementation would use more robust parsing
            content = response.content
            if "agent:" in content.lower():
                agent_id_line = [line for line in content.split("\n") 
                                if "agent:" in line.lower()][0]
                agent_id = agent_id_line.split("agent:")[1].strip().split()[0].lower()
                
                # If valid agent ID, set as current agent
                if agent_id in agents:
                    updated_state["current_agent"] = agent_id
                    
                    # Extract subtask description
                    subtask_desc = content[content.lower().find("agent:") + len("agent:") + len(agent_id):].strip()
                    updated_state["subtasks"] = state.subtasks + [{
                        "agent_id": agent_id,
                        "description": subtask_desc
                    }]
            
            # Check if task is complete
            if "complete" in content.lower() or "finished" in content.lower():
                updated_state["status"] = "completed"
            
            return updated_state
        
        graph.add_node("supervisor", _run_supervisor)
        
        # Add agent nodes
        for agent_id, agent in agents.items():
            def _run_agent(state: SupervisorState, agent_instance=agent, aid=agent_id) -> Dict[str, Any]:
                """Run the specified agent on the current subtask."""
                history = state.history or []
                subtasks = state.subtasks or []
                
                # Find the current subtask for this agent
                current_subtask = next(
                    (st for st in subtasks if st["agent_id"] == aid and 
                     st not in state.completed_subtasks),
                    None
                )
                
                if not current_subtask:
                    # No subtask found, return unchanged state
                    return {}
                
                # Get the last message from the supervisor
                last_message = history[-1] if history else None
                
                # Run the agent
                response = agent_instance.invoke({
                    "input": current_subtask["description"],
                    "history": history,
                })
                
                # Add agent's response to history
                updated_state = {
                    "history": history + [
                        AIMessage(content=f"[{aid.upper()}]: {response.output}")
                    ],
                    "completed_subtasks": state.completed_subtasks + [current_subtask],
                    "current_agent": None  # Reset current agent
                }
                
                return updated_state
            
            # Create node for this agent
            graph.add_node(agent_id, _run_agent)
        
        # Add edge from START to supervisor
        graph.add_edge(START, "supervisor")
        
        # Add conditional edges
        def _route_to_agent(state: SupervisorState) -> str:
            """Route to the appropriate agent based on the supervisor's decision."""
            if state.status == "completed":
                return END
            
            current_agent = state.current_agent
            if current_agent and current_agent in agents:
                return current_agent
            
            return "supervisor"
        
        # Add edge from supervisor to agents or END
        graph.add_conditional_edges("supervisor", _route_to_agent)
        
        # Add edges from agents back to supervisor
        for agent_id in agents:
            graph.add_edge(agent_id, "supervisor")
        
        # Set the entry point
        graph.set_entry_point("supervisor")
        
        return graph.compile()
    
    def create_pipeline_graph(self, agents: Dict[str, BaseAgent], task: Dict[str, Any]) -> StateGraph:
        """Create a pipeline graph where agents process the task in sequence.
        
        Args:
            agents: Dictionary mapping agent IDs to agent instances
            task: The task to handle
            
        Returns:
            A configured StateGraph
        """
        # Define the graph state
        graph = StateGraph(TaskState)
        
        # Add agent nodes in sequence
        agent_ids = list(agents.keys())
        
        for i, agent_id in enumerate(agent_ids):
            agent = agents[agent_id]
            
            def _run_agent(state: TaskState, agent_instance=agent) -> Dict[str, Any]:
                """Run the agent on the current task state."""
                history = state.history or []
                
                # For the first agent, use the original task description
                if not history:
                    input_text = task.get("description", "")
                else:
                    # For subsequent agents, include the history
                    input_text = f"Task: {task.get('description', '')}\n\nPrevious work:\n"
                    for msg in history:
                        input_text += f"{msg.content}\n\n"
                
                # Run the agent
                response = agent_instance.invoke({
                    "input": input_text,
                    "history": history,
                })
                
                # Add agent's response to history
                return {
                    "history": history + [
                        AIMessage(content=f"[{agent_id.upper()}]: {response.output}")
                    ]
                }
            
            # Create node for this agent
            graph.add_node(agent_id, _run_agent)
            
            # Add edge from previous agent or START
            if i == 0:
                graph.add_edge(START, agent_id)
            else:
                graph.add_edge(agent_ids[i-1], agent_id)
        
        # Add edge from last agent to END
        if agent_ids:
            graph.add_edge(agent_ids[-1], END)
        
        return graph.compile()
    
    def create_graph(self, task: Dict[str, Any]) -> StateGraph:
        """Create a graph to handle the specified task.
        
        Args:
            task: The task to create a graph for
            
        Returns:
            A compiled StateGraph
        """
        # Analyze the task
        analysis = self.analyze_task(task)
        
        # Determine which agents to include
        agent_types = [AgentType(agent_type) for agent_type in analysis["recommended_agents"]]
        
        # Create agents
        agents = {}
        for agent_type in agent_types:
            agent_id = agent_type.value
            agents[agent_id] = self.create_agent(agent_type)
        
        # Create the appropriate type of graph
        graph_type = analysis.get("graph_type", self.config.graph_type)
        
        if graph_type == "supervisor":
            graph = self.create_supervisor_graph(agents, task)
        elif graph_type == "pipeline":
            graph = self.create_pipeline_graph(agents, task)
        else:
            # Default to supervisor
            logger.warning(f"Unknown graph type: {graph_type}. Defaulting to supervisor.")
            graph = self.create_supervisor_graph(agents, task)
        
        logger.info(f"Created {graph_type} graph with {len(agents)} agents for task: {task.get('description', '')[:50]}...")
        
        return graph
    
    def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and create a graph to handle the task.
        
        Args:
            task: The task to plan for
            
        Returns:
            A dictionary containing the graph and planning information
        """
        # Analyze the task
        analysis = self.analyze_task(task)
        
        # Create the graph
        graph = self.create_graph(task)
        
        # Return the results
        return {
            "graph": graph,
            "analysis": analysis,
            "task": task
        } 