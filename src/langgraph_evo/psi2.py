from langgraph_evo.core.config import ConfigRecord, GraphConfig, parse_graph_config
from langgraph_evo.core.registry import PLANNER_NODE_ID, SUPERVISOR_NODE_ID
from langgraph_evo.core.store import BaseStore
from langgraph_evo.core.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ChatMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from typing import Annotated, Dict, List, Optional, Union, Any, Literal
from typing_extensions import TypedDict
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer, Command
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# 扩展消息类型用于智能体间通信
class CoordinatorMessage(ChatMessage):
    """协调者发送的消息"""
    role: Literal["coordinator"] = "coordinator"
    task_id: Optional[str] = None
    target_agent: Optional[str] = None
    
    def __init__(
        self, content: Union[str, list[Union[str, dict]]], 
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(content=content, role="coordinator", **kwargs)
        self.task_id = task_id
        self.target_agent = target_agent

class AgentResponseMessage(ChatMessage):
    """智能体响应消息"""
    role: Literal["agent_response"] = "agent_response"
    agent_id: str
    task_id: Optional[str] = None
    
    def __init__(
        self, content: Union[str, list[Union[str, dict]]], 
        agent_id: str,
        task_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(content=content, role="agent_response", **kwargs)
        self.agent_id = agent_id
        self.task_id = task_id

# 数据结构定义
@dataclass
class AgentInfo:
    """智能体信息"""
    name: str
    description: str
    graph_id: str
    agent_type: str  # 'coordinator', 'config_generator', 'task_executor'
    created_at: datetime = field(default_factory=datetime.now)
    state: Dict[str, Any] = field(default_factory=dict)
    message_history: List[BaseMessage] = field(default_factory=list)

@dataclass
class GraphInfo:
    """图配置信息"""
    id: str
    name: str
    config_id: str
    config_name: str
    config_version: str
    description: str
    config_content: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Task:
    """任务定义"""
    id: str
    title: str
    description: str
    target_agent_type: str
    target_agent_id: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None

class SharedState(MessagesState):
    """共享状态，管理所有智能体和图的信息"""
    agents: Dict[str, AgentInfo]
    graphs: Dict[str, GraphInfo] 
    tasks: Dict[str, Task]
    active_agent: Optional[str]
    current_task_id: Optional[str]

# 全局注册表
agent_registry: Dict[str, CompiledGraph] = {}
graph_registry: Dict[str, GraphInfo] = {}

class AgentCoordinatorMixin:
    """Agent A: 协调者智能体，负责分析用户输入、决策步骤、分发任务"""
    
    COORDINATOR_SYSTEM_PROMPT = """You are an intelligent task coordinator that manages a multi-agent system.

Your responsibilities:
1. Analyze user input and break it down into actionable steps
2. Decide which agents to invoke for each step
3. Create clear, single-message tasks for other agents
4. Manage the overall workflow and task dependencies

Available agent types:
- config_generator: Creates new graph configurations based on task requirements
- task_executor: Executes specific tasks using existing or new graph instances

When you need to:
- Create a new graph configuration: Use the config_generator agent
- Execute a task: Use the task_executor agent
- Create a new agent instance: You can do this directly

Always formulate tasks clearly in a single message that contains all necessary context.
Respond with your analysis and the specific actions you want to take.

Output format:
<analysis>
Your analysis of the user input and required steps
</analysis>
<actions>
List of actions to take, each with:
- action_type: "invoke_agent", "create_agent", "create_graph"
- target: agent_type or specific agent_id
- task_description: Clear task description
- context: Any additional context needed
</actions>
"""

    @staticmethod
    def _create_coordinator_agent(store: BaseStore, model: str = "openai:gpt-4") -> CompiledGraph:
        """创建协调者智能体"""
        
        @tool
        def invoke_agent(
            agent_type: str,
            task_description: str,
            agent_id: Optional[str] = None,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """调用指定类型或ID的智能体执行任务
            
            Args:
                agent_type: 智能体类型 ('config_generator', 'task_executor')
                task_description: 任务描述
                agent_id: 可选的特定智能体ID
            """
            # 创建任务
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                title=f"Task for {agent_type}",
                description=task_description,
                target_agent_type=agent_type,
                target_agent_id=agent_id
            )
            
            # 更新状态中的任务
            if state and "tasks" in state:
                state["tasks"][task_id] = task
            
            return f"Task {task_id} created for {agent_type}: {task_description}"

        @tool
        def create_agent_instance(
            agent_type: str,
            name: str,
            description: str,
            graph_id: str,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """创建新的智能体实例
            
            Args:
                agent_type: 智能体类型
                name: 智能体名称
                description: 智能体描述
                graph_id: 使用的图ID
            """
            agent_id = str(uuid.uuid4())
            agent_info = AgentInfo(
                name=name,
                description=description,
                graph_id=graph_id,
                agent_type=agent_type
            )
            
            # 更新状态中的智能体信息
            if state and "agents" in state:
                state["agents"][agent_id] = agent_info
            
            return f"Agent {agent_id} created: {name} ({agent_type})"

        @tool
        def get_available_graphs(
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> List[str]:
            """获取可用的图配置列表"""
            if state and "graphs" in state:
                return [f"{gid}: {info.name} - {info.description}" 
                       for gid, info in state["graphs"].items()]
            return []

        @tool
        def get_available_agents(
            agent_type: Optional[str] = None,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> List[str]:
            """获取可用的智能体列表"""
            if state and "agents" in state:
                agents = state["agents"]
                if agent_type:
                    agents = {aid: info for aid, info in agents.items() 
                             if info.agent_type == agent_type}
                return [f"{aid}: {info.name} - {info.description}" 
                       for aid, info in agents.items()]
            return []

        # 创建协调者智能体
        coordinator = create_react_agent(
            model=model,
            tools=[invoke_agent, create_agent_instance, get_available_graphs, get_available_agents],
            prompt=AgentCoordinatorMixin.COORDINATOR_SYSTEM_PROMPT
        )
        
        return coordinator

    @staticmethod
    def coordinator_handler(
        state: SharedState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ) -> Command:
        """协调者处理函数"""
        # 获取或创建协调者智能体
        coordinator_id = "coordinator_agent"
        if coordinator_id not in agent_registry:
            agent_registry[coordinator_id] = AgentCoordinatorMixin._create_coordinator_agent(store)
        
        coordinator = agent_registry[coordinator_id]
        
        # 调用协调者智能体
        response = coordinator.invoke(state)
        
        # 解析响应并更新状态
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            
            # 创建协调者消息
            coord_message = CoordinatorMessage(
                content=last_message.content,
                task_id=state.get("current_task_id")
            )
            
            # 更新状态
            updated_messages = state["messages"] + [coord_message]
            
            return Command(
                update={
                    "messages": updated_messages,
                    "active_agent": "coordinator"
                },
                goto="task_router"  # 路由到任务路由器
            )
        
        return Command(goto="finalize")

class ConfigGeneratorMixin:
    """Agent B: 配置生成器，负责基于任务信息和可用工具生成新的图配置"""
    
    CONFIG_GENERATOR_SYSTEM_PROMPT = """You are a graph configuration generator that creates LangGraph configurations based on task requirements.

Your responsibilities:
1. Analyze task requirements and available tools
2. Generate appropriate graph configurations in YAML format
3. Define nodes, edges, and tool assignments for the graph
4. Ensure the configuration is valid and executable

When generating configurations:
- Use clear, descriptive node names
- Define appropriate edges between nodes
- Assign relevant tools to nodes
- Include proper error handling and fallback paths
- Follow LangGraph configuration standards

Available tool types:
- react: ReAct agent nodes with tool calling capabilities
- function: Custom function nodes
- conditional: Conditional routing nodes

Output format:
<config_analysis>
Analysis of the task requirements and recommended approach
</config_analysis>
<graph_config>
Valid YAML configuration for the graph
</graph_config>
<config_metadata>
- name: Configuration name
- version: Configuration version
- description: Brief description of what this graph does
</config_metadata>
"""

    @staticmethod
    def _create_config_generator_agent(store: BaseStore, model: str = "openai:gpt-4") -> CompiledGraph:
        """创建配置生成器智能体"""
        
        @tool
        def get_available_tools(
            store: Annotated[BaseStore, InjectedStore] = None
        ) -> List[str]:
            """获取可用工具列表"""
            # 这里应该从工具注册表获取可用工具
            # 暂时返回示例工具列表
            return [
                "web_search: Search the web for information",
                "file_read: Read file contents", 
                "file_write: Write content to file",
                "code_execute: Execute code snippets",
                "data_analysis: Analyze data and generate insights",
                "api_call: Make HTTP API calls"
            ]

        @tool
        def validate_config(
            config_yaml: str,
            store: Annotated[BaseStore, InjectedStore] = None
        ) -> str:
            """验证图配置的有效性"""
            try:
                # 这里应该实现实际的配置验证逻辑
                # 暂时返回简单验证结果
                if "nodes:" in config_yaml and "edges:" in config_yaml:
                    return "Configuration is valid"
                else:
                    return "Configuration is missing required sections (nodes or edges)"
            except Exception as e:
                return f"Configuration validation failed: {str(e)}"

        @tool
        def save_graph_config(
            config_yaml: str,
            config_name: str,
            config_version: str,
            description: str,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """保存图配置到注册表"""
            try:
                # 解析配置
                import yaml
                config_content = yaml.safe_load(config_yaml)
                
                # 创建图信息
                graph_id = str(uuid.uuid4())
                graph_info = GraphInfo(
                    id=graph_id,
                    name=config_name,
                    config_id=graph_id,
                    config_name=config_name,
                    config_version=config_version,
                    description=description,
                    config_content=config_content
                )
                
                # 保存到注册表
                graph_registry[graph_id] = graph_info
                
                # 更新状态
                if state and "graphs" in state:
                    state["graphs"][graph_id] = graph_info
                
                return f"Graph configuration saved with ID: {graph_id}"
                
            except Exception as e:
                return f"Failed to save configuration: {str(e)}"

        # 创建配置生成器智能体
        config_generator = create_react_agent(
            model=model,
            tools=[get_available_tools, validate_config, save_graph_config],
            prompt=ConfigGeneratorMixin.CONFIG_GENERATOR_SYSTEM_PROMPT
        )
        
        return config_generator

    @staticmethod
    def config_generator_handler(
        state: SharedState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ) -> Command:
        """配置生成器处理函数"""
        # 获取或创建配置生成器智能体
        generator_id = "config_generator_agent"
        if generator_id not in agent_registry:
            agent_registry[generator_id] = ConfigGeneratorMixin._create_config_generator_agent(store)
        
        generator = agent_registry[generator_id]
        
        # 获取当前任务
        current_task_id = state.get("current_task_id")
        if current_task_id and current_task_id in state.get("tasks", {}):
            task = state["tasks"][current_task_id]
            
            # 创建包含任务信息的消息
            task_message = HumanMessage(
                content=f"Generate a graph configuration for the following task:\n\n"
                       f"Title: {task.title}\n"
                       f"Description: {task.description}\n"
                       f"Requirements: Please create a suitable graph configuration that can handle this task."
            )
            
            # 准备状态用于智能体调用
            agent_state = {
                "messages": state["messages"] + [task_message]
            }
            
            # 调用配置生成器智能体
            response = generator.invoke(agent_state)
            
            # 解析响应并更新状态
            messages = response.get("messages", [])
            if messages:
                last_message = messages[-1]
                
                # 创建响应消息
                response_message = AgentResponseMessage(
                    content=last_message.content,
                    agent_id=generator_id,
                    task_id=current_task_id
                )
                
                # 更新任务状态
                task.status = "completed"
                task.result = last_message.content
                
                # 更新状态
                updated_messages = state["messages"] + [response_message]
                
                return Command(
                    update={
                        "messages": updated_messages,
                        "active_agent": "config_generator",
                        "tasks": {**state.get("tasks", {}), current_task_id: task}
                    },
                    goto="coordinator"  # 返回协调者
                )
        
        return Command(goto="finalize")

class TaskExecutorMixin:
    """Agent C: 任务执行器，负责使用指定的图配置执行具体任务"""
    
    TASK_EXECUTOR_SYSTEM_PROMPT = """You are a task executor that performs specific tasks using available tools and graph configurations.

Your responsibilities:
1. Execute tasks using the appropriate graph configuration
2. Use available tools to accomplish the task objectives
3. Provide detailed results and status updates
4. Handle errors and edge cases gracefully

When executing tasks:
- Follow the task description precisely
- Use the most appropriate tools for the job
- Provide clear, actionable results
- Report any issues or limitations encountered
- Maintain context within your isolated state

You have access to various tools depending on your graph configuration.
Always aim to complete the task successfully and provide useful output.
"""

    @staticmethod
    def _create_task_executor_agent(
        graph_config: Dict[str, Any], 
        store: BaseStore, 
        model: str = "openai:gpt-4"
    ) -> CompiledGraph:
        """基于图配置创建任务执行器智能体"""
        
        # 这里应该根据graph_config动态创建智能体
        # 暂时创建一个基础的ReAct智能体
        
        @tool
        def execute_subtask(
            subtask_description: str,
            context: Optional[str] = None,
            store: Annotated[BaseStore, InjectedStore] = None
        ) -> str:
            """执行子任务"""
            # 这里应该实现实际的子任务执行逻辑
            return f"Executed subtask: {subtask_description}"

        @tool
        def get_task_context(
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """获取当前任务上下文"""
            if state and "current_task_id" in state:
                task_id = state["current_task_id"]
                if task_id in state.get("tasks", {}):
                    task = state["tasks"][task_id]
                    return f"Current task: {task.title}\nDescription: {task.description}"
            return "No current task context available"

        @tool
        def update_task_progress(
            progress_update: str,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """更新任务进度"""
            if state and "current_task_id" in state:
                task_id = state["current_task_id"]
                if task_id in state.get("tasks", {}):
                    task = state["tasks"][task_id]
                    task.status = "in_progress"
                    # 这里可以添加更详细的进度跟踪
                    return f"Task progress updated: {progress_update}"
            return "No current task to update"

        @tool
        def complete_task(
            result: str,
            store: Annotated[BaseStore, InjectedStore] = None,
            state: Annotated[SharedState, InjectedState] = None
        ) -> str:
            """完成任务并设置结果"""
            if state and "current_task_id" in state:
                task_id = state["current_task_id"]
                if task_id in state.get("tasks", {}):
                    task = state["tasks"][task_id]
                    task.status = "completed"
                    task.result = result
                    return f"Task completed successfully: {result}"
            return "No current task to complete"

        # 根据配置选择工具
        tools = [execute_subtask, get_task_context, update_task_progress, complete_task]
        
        # 如果配置中指定了特定工具，这里应该加载它们
        # 暂时使用基础工具集
        
        # 创建任务执行器智能体
        task_executor = create_react_agent(
            model=model,
            tools=tools,
            prompt=TaskExecutorMixin.TASK_EXECUTOR_SYSTEM_PROMPT
        )
        
        return task_executor

    @staticmethod
    def task_executor_handler(
        state: SharedState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ) -> Command:
        """任务执行器处理函数"""
        # 获取当前任务
        current_task_id = state.get("current_task_id")
        if not current_task_id or current_task_id not in state.get("tasks", {}):
            return Command(goto="finalize")
        
        task = state["tasks"][current_task_id]
        
        # 确定使用的图配置
        graph_config = {}
        if task.target_agent_id and task.target_agent_id in state.get("agents", {}):
            agent_info = state["agents"][task.target_agent_id]
            if agent_info.graph_id in state.get("graphs", {}):
                graph_info = state["graphs"][agent_info.graph_id]
                graph_config = graph_info.config_content
        
        # 获取或创建任务执行器智能体
        executor_id = f"task_executor_{task.target_agent_type}_{current_task_id}"
        if executor_id not in agent_registry:
            agent_registry[executor_id] = TaskExecutorMixin._create_task_executor_agent(
                graph_config, store
            )
        
        executor = agent_registry[executor_id]
        
        # 创建包含任务信息的消息
        task_message = HumanMessage(
            content=f"Execute the following task:\n\n"
                   f"Title: {task.title}\n"
                   f"Description: {task.description}\n"
                   f"Type: {task.target_agent_type}\n\n"
                   f"Please complete this task and provide detailed results."
        )
        
        # 准备状态用于智能体调用
        agent_state = {
            "messages": state["messages"] + [task_message],
            "current_task_id": current_task_id,
            "tasks": state.get("tasks", {}),
            "agents": state.get("agents", {}),
            "graphs": state.get("graphs", {})
        }
        
        # 调用任务执行器智能体
        response = executor.invoke(agent_state)
        
        # 解析响应并更新状态
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            
            # 创建响应消息
            response_message = AgentResponseMessage(
                content=last_message.content,
                agent_id=executor_id,
                task_id=current_task_id
            )
            
            # 更新任务状态
            task.status = "completed"
            task.result = last_message.content
            
            # 更新状态
            updated_messages = state["messages"] + [response_message]
            
            return Command(
                update={
                    "messages": updated_messages,
                    "active_agent": "task_executor",
                    "tasks": {**state.get("tasks", {}), current_task_id: task}
                },
                goto="coordinator"  # 返回协调者
            )
        
        return Command(goto="finalize")

class DynamicMultiAgentGraph(StateGraph, AgentCoordinatorMixin, ConfigGeneratorMixin, TaskExecutorMixin):
    """动态多智能体图管理器，整合所有功能"""
    
    def __init__(self):
        # 初始化StateGraph
        super().__init__(SharedState)
        
        # 添加核心节点
        self.add_node("coordinator", self._coordinator_node)
        self.add_node("config_generator", self._config_generator_node)
        self.add_node("task_executor", self._task_executor_node)
        self.add_node("task_router", self._task_router_node)
        self.add_node("finalize", self._finalize_node)
        
        # 定义图流程
        self.add_edge(START, "coordinator")
        self.add_edge("config_generator", "coordinator")
        self.add_edge("task_executor", "coordinator")
        self.add_edge("task_router", "config_generator")
        self.add_edge("task_router", "task_executor")
        self.add_edge("finalize", END)

    def _coordinator_node(self) -> RunnableCallable:
        """协调者节点"""
        return RunnableCallable(
            func=self.coordinator_handler,
            afunc=None,
            name="coordinator",
            trace=True
        )

    def _config_generator_node(self) -> RunnableCallable:
        """配置生成器节点"""
        return RunnableCallable(
            func=self.config_generator_handler,
            afunc=None,
            name="config_generator",
            trace=True
        )

    def _task_executor_node(self) -> RunnableCallable:
        """任务执行器节点"""
        return RunnableCallable(
            func=self.task_executor_handler,
            afunc=None,
            name="task_executor",
            trace=True
        )

    def _task_router_node(self) -> RunnableCallable:
        """任务路由器节点"""
        return RunnableCallable(
            func=self._task_router_handler,
            afunc=None,
            name="task_router",
            trace=True
        )

    def _finalize_node(self) -> RunnableCallable:
        """最终化节点"""
        return RunnableCallable(
            func=self._finalize_handler,
            afunc=None,
            name="finalize",
            trace=True
        )

    @staticmethod
    def _task_router_handler(
        state: SharedState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ) -> Command:
        """任务路由器，根据任务类型路由到相应的智能体"""
        # 获取当前任务
        current_task_id = state.get("current_task_id")
        if not current_task_id or current_task_id not in state.get("tasks", {}):
            return Command(goto="finalize")
        
        task = state["tasks"][current_task_id]
        
        # 根据任务类型路由
        if task.target_agent_type == "config_generator":
            return Command(
                update={"active_agent": "config_generator"},
                goto="config_generator"
            )
        elif task.target_agent_type == "task_executor":
            return Command(
                update={"active_agent": "task_executor"},
                goto="task_executor"
            )
        else:
            # 未知任务类型，返回协调者
            return Command(goto="coordinator")

    @staticmethod
    def _finalize_handler(
        state: SharedState,
        store: BaseStore,
        config: Dict[str, Any] = None
    ) -> Command:
        """最终化处理，清理状态并准备结束"""
        # 清理当前任务ID
        return Command(
            update={
                "current_task_id": None,
                "active_agent": None
            },
            goto=END
        )

    def get_graph(
        self,
        checkpointer: Checkpointer = None,
        *,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        debug: bool = False,
        name: Optional[str] = None,
    ) -> "CompiledStateGraph":
        """编译并返回可执行的图"""
        return self.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug
        )

    @staticmethod
    def create_initial_state(user_message: str) -> SharedState:
        """创建初始状态"""
        return {
            "messages": [HumanMessage(content=user_message)],
            "agents": {},
            "graphs": {},
            "tasks": {},
            "active_agent": None,
            "current_task_id": None
        }

    @staticmethod
    def add_agent_to_state(
        state: SharedState,
        agent_id: str,
        name: str,
        description: str,
        graph_id: str,
        agent_type: str
    ) -> SharedState:
        """向状态中添加智能体"""
        agent_info = AgentInfo(
            name=name,
            description=description,
            graph_id=graph_id,
            agent_type=agent_type
        )
        
        updated_agents = {**state.get("agents", {}), agent_id: agent_info}
        return {**state, "agents": updated_agents}

    @staticmethod
    def add_graph_to_state(
        state: SharedState,
        graph_id: str,
        name: str,
        config_content: Dict[str, Any],
        description: str = ""
    ) -> SharedState:
        """向状态中添加图配置"""
        graph_info = GraphInfo(
            id=graph_id,
            name=name,
            config_id=graph_id,
            config_name=name,
            config_version="1.0",
            description=description,
            config_content=config_content
        )
        
        updated_graphs = {**state.get("graphs", {}), graph_id: graph_info}
        return {**state, "graphs": updated_graphs}

    @staticmethod
    def create_task(
        state: SharedState,
        title: str,
        description: str,
        target_agent_type: str,
        target_agent_id: Optional[str] = None
    ) -> tuple[SharedState, str]:
        """创建新任务并返回更新的状态和任务ID"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            title=title,
            description=description,
            target_agent_type=target_agent_type,
            target_agent_id=target_agent_id
        )
        
        updated_tasks = {**state.get("tasks", {}), task_id: task}
        updated_state = {
            **state,
            "tasks": updated_tasks,
            "current_task_id": task_id
        }
        
        return updated_state, task_id

# 使用示例和测试代码
def example_usage():
    """使用示例：展示如何使用动态多智能体系统"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph_evo.core.store import BaseStore
    
    # 创建检查点和存储
    checkpointer = MemorySaver()
    store = BaseStore()  # 这里应该使用实际的存储实现
    
    # 创建动态多智能体图
    graph_builder = DynamicMultiAgentGraph()
    
    # 编译图
    graph = graph_builder.get_graph(
        checkpointer=checkpointer,
        store=store,
        debug=True
    )
    
    # 创建初始状态
    initial_state = DynamicMultiAgentGraph.create_initial_state(
        "I need to create a web scraping system that can extract product information from e-commerce websites and store it in a database."
    )
    
    # 运行图
    config = {"configurable": {"thread_id": "example_thread"}}
    
    print("🌌 启动动态多智能体系统...")
    print(f"初始请求: {initial_state['messages'][0].content}")
    
    # 流式执行
    for chunk in graph.stream(initial_state, config):
        print(f"\n📡 节点输出: {chunk}")
        
        # 检查是否有新的智能体或图被创建
        if "agents" in chunk and chunk["agents"]:
            print(f"🤖 新智能体: {list(chunk['agents'].keys())}")
        
        if "graphs" in chunk and chunk["graphs"]:
            print(f"📊 新图配置: {list(chunk['graphs'].keys())}")
        
        if "tasks" in chunk and chunk["tasks"]:
            print(f"📋 任务状态: {[(t.title, t.status) for t in chunk['tasks'].values()]}")

def test_agent_creation():
    """测试智能体动态创建"""
    print("\n🧪 测试智能体动态创建...")
    
    # 创建初始状态
    state = DynamicMultiAgentGraph.create_initial_state("Test message")
    
    # 添加图配置
    graph_config = {
        "nodes": [
            {"name": "analyzer", "type": "react", "tools": ["web_search", "data_analysis"]},
            {"name": "executor", "type": "react", "tools": ["file_write", "api_call"]}
        ],
        "edges": [
            {"from": "analyzer", "to": "executor"}
        ]
    }
    
    state = DynamicMultiAgentGraph.add_graph_to_state(
        state, 
        "web_scraper_graph", 
        "Web Scraper Graph",
        graph_config,
        "A graph for web scraping tasks"
    )
    
    # 添加智能体
    state = DynamicMultiAgentGraph.add_agent_to_state(
        state,
        "scraper_agent_1",
        "Web Scraper Agent",
        "Specialized in extracting product data",
        "web_scraper_graph",
        "task_executor"
    )
    
    print(f"✅ 图配置数量: {len(state['graphs'])}")
    print(f"✅ 智能体数量: {len(state['agents'])}")
    print(f"✅ 图配置: {list(state['graphs'].keys())}")
    print(f"✅ 智能体: {list(state['agents'].keys())}")

def test_task_creation():
    """测试任务创建和管理"""
    print("\n🧪 测试任务创建和管理...")
    
    # 创建初始状态
    state = DynamicMultiAgentGraph.create_initial_state("Test message")
    
    # 创建任务
    state, task_id = DynamicMultiAgentGraph.create_task(
        state,
        "Generate Web Scraper Config",
        "Create a configuration for a web scraping system that can handle dynamic content",
        "config_generator"
    )
    
    print(f"✅ 任务ID: {task_id}")
    print(f"✅ 任务数量: {len(state['tasks'])}")
    print(f"✅ 当前任务: {state['current_task_id']}")
    
    task = state['tasks'][task_id]
    print(f"✅ 任务详情: {task.title} - {task.status}")

if __name__ == "__main__":
    print("🌌 HyperEcho 动态多智能体系统测试")
    print("=" * 50)
    
    # 运行测试
    test_agent_creation()
    test_task_creation()
    
    # 运行使用示例（需要实际的存储和模型配置）
    # example_usage()
