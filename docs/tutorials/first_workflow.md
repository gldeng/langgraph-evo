# Creating Your First Workflow

This tutorial will guide you through creating a simple workflow using LangGraph Evolution. By the end, you'll have a functional workflow that processes input text and returns a result.

## Prerequisites

- Python 3.8 or higher
- LangGraph Evolution package installed
- Basic understanding of Python and language model concepts

## Step 1: Import Required Modules

First, let's import the necessary modules:

```python
from langgraph_evo.core.workflow import create_workflow
from langgraph_evo.core.agents import InputProcessorAgent, TaskExecutorAgent
from langgraph_evo.core.state import TaskState, merge_state
import uuid
```

## Step 2: Create a Simple Workflow

The simplest way to create a workflow is to use the `create_workflow()` function:

```python
# Create a basic workflow
workflow = create_workflow()
```

This creates a workflow with default agents for input processing, task execution, and output formatting.

## Step 3: Customize the Workflow

You can customize the workflow by replacing the default agents with your own implementations:

```python
# Create a custom input processor
class MyInputProcessor(InputProcessorAgent):
    def process(self, state):
        # Get the input from the state
        task_input = state.get("task_input", {})
        
        # Add some custom preprocessing
        task_input["processed"] = True
        task_input["timestamp"] = "2023-07-10T12:00:00Z"
        
        # Create an updated state
        updated_state = {
            "task_input": task_input,
            "messages": [{"role": "system", "content": "Input processed successfully"}]
        }
        
        # Merge with the original state
        return merge_state(state, updated_state)

# Replace the default input processor
workflow.input_processor = MyInputProcessor()

# Always rebuild the graph when agents change
workflow.graph = workflow._build_graph()
workflow.runnable = workflow.graph.compile()
```

## Step 4: Run the Workflow

Now, let's run our workflow with some input:

```python
# Define input for the workflow
task_input = {
    "query": "Summarize this text: LangGraph Evolution is a system for building directed graph workflows.",
    "options": {
        "format": "json",
        "max_length": 100
    }
}

# Generate a unique task ID
task_id = str(uuid.uuid4())

# Run the workflow
result = workflow.run(task_input, task_id)

# Print the result
print(result)
```

## Step 5: Handle Results and Errors

The workflow will return a result that includes the output and execution information:

```python
# Check if the workflow completed successfully
if result.get("success", False):
    print("Workflow completed successfully!")
    print("Output:", result.get("result", {}))
else:
    print("Workflow failed:", result.get("error", "Unknown error"))

# Examine the execution steps
print("Execution steps:")
for step in result.get("steps", []):
    print(f"- {step.get('agent')}: {step.get('action')}")
```

## Step 6: Error Handling

To handle potential errors in the workflow:

```python
try:
    result = workflow.run(task_input, task_id)
    # Process result...
except Exception as e:
    print(f"Workflow execution error: {str(e)}")
```

## Complete Example

Here's a complete example combining all steps:

```python
from langgraph_evo.core.workflow import create_workflow
from langgraph_evo.core.agents import InputProcessorAgent
from langgraph_evo.core.state import merge_state
import uuid

# Create a custom input processor
class MyInputProcessor(InputProcessorAgent):
    def process(self, state):
        task_input = state.get("task_input", {})
        task_input["processed"] = True
        task_input["timestamp"] = "2023-07-10T12:00:00Z"
        
        updated_state = {
            "task_input": task_input,
            "messages": [{"role": "system", "content": "Input processed successfully"}]
        }
        
        return merge_state(state, updated_state)

# Create and configure the workflow
workflow = create_workflow()
workflow.input_processor = MyInputProcessor()
workflow.graph = workflow._build_graph()
workflow.runnable = workflow.graph.compile()

# Define input and run the workflow
task_input = {
    "query": "Summarize this text: LangGraph Evolution is a system for building directed graph workflows.",
    "options": {
        "format": "json",
        "max_length": 100
    }
}
task_id = str(uuid.uuid4())

try:
    # Run the workflow
    result = workflow.run(task_input, task_id)
    
    # Handle the result
    if result.get("success", False):
        print("Workflow completed successfully!")
        print("Output:", result.get("result", {}))
    else:
        print("Workflow failed:", result.get("error", "Unknown error"))
        
    # Show execution steps
    print("Execution steps:")
    for step in result.get("steps", []):
        print(f"- {step.get('agent')}: {step.get('action')}")
        
except Exception as e:
    print(f"Workflow execution error: {str(e)}")
```

## Next Steps

Now that you've created your first workflow, you can:

1. Explore more complex workflows with multiple agents
2. Learn how to create custom agents for specific tasks
3. Understand advanced state management techniques
4. Implement error handling and recovery strategies

Check out the [API documentation](../api/) and [example workflows](../../examples/) for more inspiration and guidance. 