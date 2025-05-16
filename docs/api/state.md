# State Management API Reference

This document provides a detailed reference for the state management components in LangGraph Evolution.

## Module: `langgraph_evo.core.state`

The state module provides classes and utilities for managing state across workflow nodes.

### Classes

#### `BaseState`

```python
class BaseState(TypedDict):
```

Base state for all workflows. This represents the fundamental state object that gets passed between nodes in the workflow graph.

**Attributes:**
- `metadata` (`Dict[str, Any]`): A dictionary containing metadata about the state.

#### `TaskState`

```python
class TaskState(BaseState):
```

State for task processing workflows. Contains information about the task being processed and its current state.

**Attributes:**
- `task_id` (`str`): Unique identifier for the task
- `task_input` (`Dict[str, Any]`): Input data for the task
- `task_output` (`Optional[Dict[str, Any]]`): Output data from the task, if available
- `messages` (`List[Dict[str, Any]]`): List of messages, representing the history of communication
- `steps` (`List[Dict[str, Any]]`): List of steps, tracking each agent's processing in sequence
- `context` (`Dict[str, Any]`): Dictionary holding intermediate data needed between processing steps
- `error` (`Optional[str]`): Error field for tracking any errors during processing

### Functions

#### `merge_state`

```python
def merge_state(old_state: TaskState, new_partial_state: Dict[str, Any]) -> TaskState:
```

Merge a partial state update into the full state. This utility function helps update the state with partial updates from nodes, handling nested merges and list appends appropriately.

**Arguments:**
- `old_state` (`TaskState`): The current complete state
- `new_partial_state` (`Dict[str, Any]`): Partial updates to apply

**Returns:**
- `TaskState`: The merged state with updates applied

**Behavior:**
- For new keys not in `old_state`, the value is simply added
- For dictionary values, dictionaries are merged recursively
- For list values, lists are appended
- For other values, the new value overrides the old value

#### `get_messages`

```python
def get_messages(state: TaskState) -> List[Dict[str, Any]]:
```

Extract messages from the state for passing to LLM agents.

**Arguments:**
- `state` (`TaskState`): The current workflow state

**Returns:**
- `List[Dict[str, Any]]`: The messages list from the state

#### `get_context`

```python
def get_context(state: TaskState) -> Dict[str, Any]:
```

Extract the context dictionary from the state.

**Arguments:**
- `state` (`TaskState`): The current workflow state

**Returns:**
- `Dict[str, Any]`: The context dictionary from the state

## Example Usage

### Creating and Updating State

```python
from langgraph_evo.core.state import TaskState, merge_state

# Create an initial state
initial_state = TaskState(
    task_id="task-123",
    task_input={"query": "Process this data"},
    task_output=None,
    messages=[{"role": "system", "content": "Initial message"}],
    steps=[],
    context={},
    error=None,
    metadata={"created_at": "2023-07-10T12:00:00Z"}
)

# Create a partial update
partial_update = {
    "messages": [{"role": "user", "content": "New message"}],
    "context": {"key": "value"}
}

# Merge the update into the state
updated_state = merge_state(initial_state, partial_update)
```

### Extracting State Components

```python
from langgraph_evo.core.state import get_messages, get_context

# Extract messages from state
messages = get_messages(updated_state)

# Extract context from state
context = get_context(updated_state)
```

## Best Practices

1. **Immutability**: Treat state objects as immutable. Use `merge_state` to create new state objects with updates.

2. **Type Safety**: Use type hints and TypedDict to ensure state objects have the expected structure.

3. **Minimal Updates**: Only include the fields that need to be updated in partial updates.

4. **State Selection**: Use state selection functions like `get_messages` and `get_context` to extract specific parts of the state.

5. **Error Tracking**: Use the `error` field to track errors during processing. 