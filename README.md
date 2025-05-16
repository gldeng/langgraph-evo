# LangGraph-Evo

An evolutionary framework for LangGraph-based multi-agent workflows.

## Overview

LangGraph Evolution System (LangGraph-Evo) is an advanced framework built on top of LangGraph that enables the dynamic creation, execution, and evolution of multi-agent workflows. The system leverages LangGraph's directed graph architecture and state management capabilities to implement a continuous improvement loop where agent workflows automatically evolve based on performance evaluations.

## Features

- **Automated Configuration and Optimization**: Evolve multi-agent workflows using LangGraph's graph-based architecture
- **Comprehensive Tracing**: Extended evaluation mechanisms for LangGraph workflows
- **Evolutionary Improvements**: Performance-based evolution of agent configurations
- **LangGraph Compatibility**: Seamless integration with existing LangGraph components

## Installation

```bash
pip install langgraph-evo
```

## Quick Start

```python
from langgraph_evo import create_evolving_graph
from langgraph_evo.components import Planner, Factory, Evaluator

# Create an evolving graph
evolving_graph = create_evolving_graph(
    planner=Planner(),
    factory=Factory(),
    evaluator=Evaluator()
)

# Run an evolution cycle
evolved_graph = evolving_graph.evolve(iterations=5)

# Use the evolved graph
result = evolved_graph.invoke({
    "task": "Solve this problem: ..."
})
```

## Project Structure

- `src/langgraph_evo/`: Main package
  - `core/`: Core functionality
  - `components/`: Implementation of framework components
  - `utils/`: Utility functions
  - `database/`: Database interaction logic

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/your-username/langgraph-evo.git
cd langgraph-evo

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 