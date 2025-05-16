# LangGraph Evolution API Reference

This section contains detailed documentation for the LangGraph Evolution API.

## Core Modules

- [State Management](./state.md): Classes and utilities for managing state across workflow nodes.
- [Workflow](./workflow.md): Core workflow creation and management functionality.
- [Agents](./agents.md): Built-in agent types and interfaces for creating custom agents.

## Import Structure

```python
# State management
from langgraph_evo.core.state import TaskState, BaseState, merge_state, get_messages, get_context

# Workflow creation and execution
from langgraph_evo.core.workflow import create_workflow, WorkflowError

# Agent interfaces and implementations
from langgraph_evo.core.agents import InputProcessorAgent, TaskExecutorAgent, OutputFormatterAgent
```

## Common Usage Patterns

### Creating a Basic Workflow

```python
from langgraph_evo.core.workflow import create_workflow
import uuid

# Create a workflow
workflow = create_workflow()

# Define task input
task_input = {
    "query": "Process this query",
    "options": {
        "format": "json"
    }
}

# Generate a unique task ID
task_id = str(uuid.uuid4())

# Run the workflow
result = workflow.run(task_input, task_id)
```

### Customizing a Workflow with Custom Agents

```python
from langgraph_evo.core.workflow import create_workflow
from langgraph_evo.core.agents import InputProcessorAgent
from langgraph_evo.core.state import merge_state

# Create a custom input processor
class MyInputProcessor(InputProcessorAgent):
    def process(self, state):
        # Custom processing logic
        task_input = state.get("task_input", {})
        task_input["processed"] = True
        
        # Return partial state update
        return merge_state(state, {
            "task_input": task_input,
            "messages": [{"role": "system", "content": "Input processed"}]
        })

# Create workflow and set custom agent
workflow = create_workflow()
workflow.input_processor = MyInputProcessor()

# Rebuild graph when agents change
workflow.graph = workflow._build_graph()
workflow.runnable = workflow.graph.compile()
```

### Creating a Dynamic Workflow with Custom Graph

```python
from langgraph_evo.core.workflow import create_workflow

# Create a workflow
workflow = create_workflow()

# Define custom edges (node connections)
edges = {
    "input_processor": lambda x: "task_executor",
    "task_executor": lambda x: "output_formatter"
}

# Build graph with custom edges
workflow.graph = workflow._build_graph(edges)
workflow.runnable = workflow.graph.compile()
```

## Error Handling

```python
from langgraph_evo.core.workflow import create_workflow, WorkflowError

workflow = create_workflow()

try:
    result = workflow.run(task_input, task_id)
    if not result.get("success", False):
        print(f"Workflow failed: {result.get('error')}")
    else:
        print("Workflow succeeded")
except WorkflowError as e:
    print(f"Workflow error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

## Best Practices

1. **Rebuild the Graph After Changes**: Always rebuild and recompile the graph after changing agents or connections.

2. **Use State Merging**: Use the `merge_state` function to create proper state updates.

3. **Handle Errors**: Implement proper error handling around workflow execution.

4. **Customize Incrementally**: Start with the basic workflow and customize one component at a time.

5. **Maintain Immutability**: Treat state as immutable, creating new state objects rather than modifying existing ones. 