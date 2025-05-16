# LangGraph Evolution Workflow Diagrams

This document provides visual representations of LangGraph Evolution workflows to help understand their structure and execution flow.

## Basic Workflow Structure

```
  ┌───────────────────────────────────────────────────┐
  │                   Workflow                         │
  │                                                    │
  │   ┌───────────┐     ┌───────────┐     ┌────────┐  │
  │   │   Input   │     │    Task   │     │ Output │  │
  │   │ Processor │────▶│  Executor │────▶│Formatter│  │
  │   └───────────┘     └───────────┘     └────────┘  │
  │                                                    │
  └───────────────────────────────────────────────────┘
```

The basic workflow consists of three main components:
1. **Input Processor**: Processes and validates input data
2. **Task Executor**: Performs the main task processing
3. **Output Formatter**: Formats the output into the desired structure

## Workflow Execution Flow

```
  ┌────────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
  │            │      │            │      │            │      │            │
  │  Initial   │      │ Processed  │      │  Executed  │      │   Final    │
  │   State    │─────▶│   State    │─────▶│   State    │─────▶│   State    │
  │            │      │            │      │            │      │            │
  └────────────┘      └────────────┘      └────────────┘      └────────────┘
        │                   │                   │                   │
        ▼                   ▼                   ▼                   ▼
  ┌────────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
  │ task_id    │      │ task_id    │      │ task_id    │      │ task_id    │
  │ task_input │      │ task_input │      │ task_input │      │ task_input │
  │ messages:[]│      │ messages:[1]│      │ messages:[2]│      │ messages:[3]│
  │ steps:[]   │      │ steps:[1]   │      │ steps:[2]   │      │ steps:[3]   │
  │ context:{} │      │ context:{} │      │ context:{a} │      │ context:{a,b}│
  │ error:null │      │ error:null │      │ error:null │      │ error:null │
  └────────────┘      └────────────┘      └────────────┘      └────────────┘
```

The execution flow shows how the state evolves as it passes through the workflow:
1. **Initial State**: Contains task ID and input
2. **Processed State**: After input processing, with first message and step
3. **Executed State**: After task execution, with additional messages and context data
4. **Final State**: After output formatting, with complete messages and context

## State Merging Process

```
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │  Current State │     │ Partial Update │     │  Updated State │
  │                │     │                │     │                │
  │ task_id: "123" │     │                │     │ task_id: "123" │
  │ messages: [m1] │  +  │ messages: [m2] │  =  │ messages: [m1,m2] │
  │ context: {a:1} │     │ context: {b:2} │     │ context: {a:1,b:2} │
  │                │     │                │     │                │
  └────────────────┘     └────────────────┘     └────────────────┘
```

State merging shows how partial updates are applied:
- Lists (like messages) are appended
- Dictionaries (like context) are merged
- Other values override existing values

## Custom Workflow with Multiple Agents

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         Custom Workflow                              │
  │                                                                      │
  │   ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌────────┐  │
  │   │   Input   │     │  Research │     │  Content  │     │ Output │  │
  │   │ Processor │────▶│   Agent   │────▶│   Agent   │────▶│Formatter│  │
  │   └───────────┘     └───────────┘     └───────────┘     └────────┘  │
  │         │                                                            │
  │         │           ┌───────────┐                                    │
  │         └──────────▶│   Error   │                                    │
  │                     │  Handler  │                                    │
  │                     └───────────┘                                    │
  └─────────────────────────────────────────────────────────────────────┘
```

A custom workflow can include additional agents and branching logic:
1. **Input Processor**: Handles initial input processing
2. **Research Agent**: Gathers additional information
3. **Content Agent**: Creates content using the research
4. **Output Formatter**: Formats the final output
5. **Error Handler**: Manages errors from any component

## Message Flow Between Agents

```
  ┌─────────────┐                             ┌─────────────┐
  │   Agent A   │                             │   Agent B   │
  │             │                             │             │
  │  ┌───────┐  │         ┌─────────┐         │  ┌───────┐  │
  │  │Process│  │         │ Message │         │  │Process│  │
  │  │ State │◀─┼─────────┤ Passing │─────────┼─▶│ State │  │
  │  └───────┘  │         └─────────┘         │  └───────┘  │
  │      │      │                             │      │      │
  │      ▼      │                             │      ▼      │
  │  ┌───────┐  │                             │  ┌───────┐  │
  │  │Update │  │                             │  │Update │  │
  │  │ State │  │                             │  │ State │  │
  │  └───────┘  │                             │  └───────┘  │
  └─────────────┘                             └─────────────┘
```

The message flow diagram illustrates how agents communicate:
1. Each agent processes the incoming state
2. Messages are passed according to the workflow graph edges
3. Each agent can update the state before passing it on

## Error Handling Flow

```
                ┌──────────────┐
                │    Normal    │
                │  Execution   │
                └──────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │  Error Check  │
               └───────┬───────┘
                       │
          ┌────────────────────────┐
          │                        │
          ▼                        ▼
  ┌───────────────┐        ┌───────────────┐
  │  No Error:    │        │  Error:       │
  │ Continue Flow │        │ Handle Error  │
  └───────┬───────┘        └───────┬───────┘
          │                        │
          ▼                        ▼
  ┌───────────────┐        ┌───────────────┐
  │  Next Node    │        │ Error Handler │
  └───────────────┘        └───────────────┘
                                  │
                                  ▼
                           ┌───────────────┐
                           │ Return Result │
                           │  With Error   │
                           └───────────────┘
```

The error handling flow shows how errors are managed:
1. Normal execution flows through the workflow
2. Error checks are performed at each stage
3. If no error is detected, execution continues to the next node
4. If an error is detected, the error handler is invoked
5. The error handler processes the error and returns a result with error information

## Using These Diagrams

These diagrams can help you:
1. Understand the structure of LangGraph Evolution workflows
2. Visualize how state flows through the system
3. Design custom workflows with multiple agents
4. Implement error handling strategies

Feel free to adapt these diagrams for your own documentation or presentations. 