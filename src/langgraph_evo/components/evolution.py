"""Evolution implementation for LangGraph Evolution System.

This module provides an Evolution component that evolves graph configurations
through a genetic algorithm approach to optimize task performance.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Tuple, Callable
import logging
import random
import copy
from enum import Enum

from langgraph.graph import StateGraph
from ..core.state import TaskState

# Set up logging
logger = logging.getLogger(__name__)

# Type for the graph
GraphType = TypeVar('GraphType', bound=StateGraph)


class MutationType(Enum):
    """Types of mutations that can be applied to graphs."""
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    CHANGE_EDGE = "change_edge"
    SWAP_NODES = "swap_nodes"
    MODIFY_NODE_CONFIG = "modify_node_config"


class Evolution:
    """Evolution system for LangGraph graphs.
    
    This component evolves graph configurations using genetic algorithm
    principles to optimize performance on tasks.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        mutation_operators: Optional[Dict[MutationType, Callable]] = None
    ):
        """Initialize the Evolution component.
        
        Args:
            config: Optional configuration dictionary for evolution parameters
            mutation_operators: Optional custom mutation operators
        """
        # Default configuration
        self.config = {
            "population_size": 10,
            "generations": 5,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "elite_size": 2,
            **(config or {})
        }
        
        # Set up mutation operators
        self.mutation_operators = mutation_operators or {}
        
        # Generate default mutation operators for any not provided
        self._init_default_mutation_operators()
        
        logger.info(f"Initialized Evolution with config: {self.config}")
    
    def _init_default_mutation_operators(self):
        """Initialize default mutation operators if not provided."""
        # Add default operators for any missing mutation types
        for mutation_type in MutationType:
            if mutation_type not in self.mutation_operators:
                # Map enum values to method names
                method_name = f"_default_{mutation_type.value}"
                if hasattr(self, method_name):
                    self.mutation_operators[mutation_type] = getattr(self, method_name)
                else:
                    logger.warning(f"No default operator found for {mutation_type}")
    
    def _default_add_node(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default mutation: Add a node to the graph.
        
        Args:
            graph_config: The graph configuration dictionary
            
        Returns:
            Modified graph configuration
        """
        # This is a placeholder implementation
        # In a real system, this would have knowledge of available node types
        # and how to properly insert them into the graph
        
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Example placeholder logic:
        if "nodes" in new_config:
            # In a real system, we would add a valid node with appropriate config
            new_node = {"type": "processor", "config": {"name": f"added_node_{random.randint(1000, 9999)}"}}
            new_config["nodes"].append(new_node)
            
            # We would also need to update edges to connect this node
            # This is highly simplified and would need real implementation
            if "edges" in new_config:
                # Connect to a random existing node
                if len(new_config["nodes"]) > 1:
                    target_idx = random.randint(0, len(new_config["nodes"]) - 2)
                    new_edge = {
                        "source": len(new_config["nodes"]) - 1,  # New node index
                        "target": target_idx,
                        "condition": "default"
                    }
                    new_config["edges"].append(new_edge)
        
        return new_config
    
    def _default_remove_node(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default mutation: Remove a node from the graph.
        
        Args:
            graph_config: The graph configuration dictionary
            
        Returns:
            Modified graph configuration
        """
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Example placeholder logic:
        if "nodes" in new_config and len(new_config["nodes"]) > 2:  # Keep at least 2 nodes
            # Select a node to remove (avoid removing critical nodes)
            removable_indices = list(range(1, len(new_config["nodes"]) - 1))  # Skip first and last
            if removable_indices:
                remove_idx = random.choice(removable_indices)
                
                # Remove the node
                new_config["nodes"].pop(remove_idx)
                
                # Update edges to maintain a valid graph
                if "edges" in new_config:
                    # Remove edges connected to the removed node
                    new_config["edges"] = [
                        edge for edge in new_config["edges"]
                        if edge["source"] != remove_idx and edge["target"] != remove_idx
                    ]
                    
                    # Update indices for remaining edges
                    for edge in new_config["edges"]:
                        if edge["source"] > remove_idx:
                            edge["source"] -= 1
                        if edge["target"] > remove_idx:
                            edge["target"] -= 1
        
        return new_config
    
    def _default_change_edge(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default mutation: Change an edge in the graph.
        
        Args:
            graph_config: The graph configuration dictionary
            
        Returns:
            Modified graph configuration
        """
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Example placeholder logic:
        if "edges" in new_config and new_config["edges"]:
            # Select an edge to modify
            edge_idx = random.randint(0, len(new_config["edges"]) - 1)
            
            # Modify the edge (target or condition)
            edge = new_config["edges"][edge_idx]
            
            # 50% chance to change target, 50% to change condition
            if random.random() < 0.5 and "nodes" in new_config and len(new_config["nodes"]) > 1:
                # Change target to another valid node
                current_target = edge["target"]
                possible_targets = [i for i in range(len(new_config["nodes"])) 
                                    if i != current_target and i != edge["source"]]
                if possible_targets:
                    edge["target"] = random.choice(possible_targets)
            else:
                # Change condition (in a real system, we'd have valid conditions)
                conditions = ["default", "success", "failure", "retry"]
                edge["condition"] = random.choice(conditions)
        
        return new_config
    
    def _default_swap_nodes(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default mutation: Swap two nodes in the graph.
        
        Args:
            graph_config: The graph configuration dictionary
            
        Returns:
            Modified graph configuration
        """
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Example placeholder logic:
        if "nodes" in new_config and len(new_config["nodes"]) > 2:
            # Select two nodes to swap
            idx1 = random.randint(1, len(new_config["nodes"]) - 2)  # Avoid swapping start/end
            idx2 = random.randint(1, len(new_config["nodes"]) - 2)
            
            # Ensure they're different
            while idx1 == idx2:
                idx2 = random.randint(1, len(new_config["nodes"]) - 2)
            
            # Swap the nodes
            new_config["nodes"][idx1], new_config["nodes"][idx2] = \
                new_config["nodes"][idx2], new_config["nodes"][idx1]
            
            # Update edges to maintain correct connections
            if "edges" in new_config:
                for edge in new_config["edges"]:
                    if edge["source"] == idx1:
                        edge["source"] = idx2
                    elif edge["source"] == idx2:
                        edge["source"] = idx1
                        
                    if edge["target"] == idx1:
                        edge["target"] = idx2
                    elif edge["target"] == idx2:
                        edge["target"] = idx1
        
        return new_config
    
    def _default_modify_node_config(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default mutation: Modify a node's configuration.
        
        Args:
            graph_config: The graph configuration dictionary
            
        Returns:
            Modified graph configuration
        """
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Example placeholder logic:
        if "nodes" in new_config and new_config["nodes"]:
            # Select a node to modify
            node_idx = random.randint(0, len(new_config["nodes"]) - 1)
            node = new_config["nodes"][node_idx]
            
            # Modify configuration (in a real system, this would be type-aware)
            if "config" in node:
                # Example: add or modify a parameter
                param_keys = ["temperature", "max_tokens", "timeout", "retries"]
                param_key = random.choice(param_keys)
                
                if param_key == "temperature":
                    node["config"][param_key] = round(random.uniform(0.1, 1.0), 1)
                elif param_key == "max_tokens":
                    node["config"][param_key] = random.randint(100, 2000)
                elif param_key == "timeout":
                    node["config"][param_key] = random.randint(10, 60)
                elif param_key == "retries":
                    node["config"][param_key] = random.randint(0, 3)
        
        return new_config
    
    def mutate(self, graph_config: Dict[str, Any], mutation_rate: Optional[float] = None) -> Dict[str, Any]:
        """Apply random mutations to a graph configuration.
        
        Args:
            graph_config: The graph configuration to mutate
            mutation_rate: Optional override for mutation rate
            
        Returns:
            Mutated graph configuration
        """
        mutation_rate = mutation_rate or self.config["mutation_rate"]
        
        # Deep copy to avoid modifying the original
        new_config = copy.deepcopy(graph_config)
        
        # Apply mutations based on mutation rate
        if random.random() < mutation_rate:
            # Select a random mutation type
            mutation_type = random.choice(list(MutationType))
            
            # Apply the mutation if we have an operator for it
            if mutation_type in self.mutation_operators:
                logger.info(f"Applying mutation: {mutation_type.value}")
                new_config = self.mutation_operators[mutation_type](new_config)
            else:
                logger.warning(f"No operator found for mutation type: {mutation_type}")
        
        return new_config
    
    def crossover(
        self, 
        parent1_config: Dict[str, Any], 
        parent2_config: Dict[str, Any],
        crossover_rate: Optional[float] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parent graph configurations.
        
        Args:
            parent1_config: First parent configuration
            parent2_config: Second parent configuration
            crossover_rate: Optional override for crossover rate
            
        Returns:
            Tuple of (child1_config, child2_config)
        """
        crossover_rate = crossover_rate or self.config["crossover_rate"]
        
        # If random value exceeds crossover rate, return copies of parents
        if random.random() > crossover_rate:
            return copy.deepcopy(parent1_config), copy.deepcopy(parent2_config)
        
        # Deep copy parents
        child1 = copy.deepcopy(parent1_config)
        child2 = copy.deepcopy(parent2_config)
        
        # Implement a basic crossover (this is a simplified placeholder)
        # In a real system, this would need to ensure the resulting graphs are valid
        
        # Example: swap some node configurations between parents
        if "nodes" in child1 and "nodes" in child2:
            # Find common nodes that can be swapped safely
            min_nodes = min(len(child1["nodes"]), len(child2["nodes"]))
            
            if min_nodes > 2:  # Need at least 3 nodes for meaningful crossover
                # Select crossover points (avoid start/end nodes)
                crossover_point = random.randint(1, min_nodes - 2)
                
                # Swap node configurations at the crossover point
                for i in range(crossover_point, min_nodes - 1):
                    # We only swap the config, not the entire node
                    # to maintain graph structure
                    if "config" in child1["nodes"][i] and "config" in child2["nodes"][i]:
                        child1["nodes"][i]["config"], child2["nodes"][i]["config"] = \
                            child2["nodes"][i]["config"], child1["nodes"][i]["config"]
        
        logger.info("Performed crossover between parent configurations")
        return child1, child2
    
    def select_parents(
        self, 
        population: List[Dict[str, Any]], 
        fitness_scores: List[float]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select two parents for reproduction using tournament selection.
        
        Args:
            population: List of graph configurations
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Tuple of (parent1_config, parent2_config)
        """
        # Tournament selection
        def select_one():
            # Select random candidates for tournament
            tournament_size = min(3, len(population))
            candidates = random.sample(range(len(population)), tournament_size)
            
            # Find the candidate with highest fitness
            best_candidate = candidates[0]
            for candidate in candidates:
                if fitness_scores[candidate] > fitness_scores[best_candidate]:
                    best_candidate = candidate
            
            return population[best_candidate]
        
        # Select two parents
        parent1 = select_one()
        parent2 = select_one()
        
        # Ensure they're different (if possible)
        attempts = 0
        while parent1 == parent2 and attempts < 5 and len(population) > 1:
            parent2 = select_one()
            attempts += 1
        
        return parent1, parent2
    
    def evolve_generation(
        self, 
        population: List[Dict[str, Any]], 
        fitness_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Evolve a population to create a new generation.
        
        Args:
            population: Current generation of graph configurations
            fitness_scores: Fitness scores for each configuration
            
        Returns:
            New generation of graph configurations
        """
        # Sort population by fitness
        sorted_pairs = sorted(zip(population, fitness_scores), 
                              key=lambda x: x[1], reverse=True)
        sorted_population, _ = zip(*sorted_pairs)
        sorted_population = list(sorted_population)
        
        new_generation = []
        
        # Elitism: Keep the best performers
        elite_size = self.config["elite_size"]
        for i in range(min(elite_size, len(sorted_population))):
            new_generation.append(copy.deepcopy(sorted_population[i]))
        
        # Fill the rest with children from crossover and mutation
        while len(new_generation) < self.config["population_size"]:
            # Select parents
            parent1, parent2 = self.select_parents(sorted_population, fitness_scores)
            
            # Perform crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Perform mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new generation
            new_generation.append(child1)
            if len(new_generation) < self.config["population_size"]:
                new_generation.append(child2)
        
        # Ensure we don't exceed population size
        return new_generation[:self.config["population_size"]]
    
    def run_evolution(
        self,
        initial_population: List[Dict[str, Any]],
        fitness_function: Callable[[Dict[str, Any]], float],
        generations: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Run the full evolutionary process.
        
        Args:
            initial_population: Starting population of graph configurations
            fitness_function: Function to evaluate fitness of a configuration
            generations: Optional override for number of generations
            
        Returns:
            Tuple of (best_config, best_fitness, final_population)
        """
        generations = generations or self.config["generations"]
        
        # Ensure population is of correct size
        population = initial_population[:self.config["population_size"]]
        
        # Fill population if needed
        while len(population) < self.config["population_size"]:
            # Take random member and mutate it
            if population:
                template = random.choice(population)
                new_member = self.mutate(template, mutation_rate=1.0)  # Force mutation
                population.append(new_member)
            else:
                logger.error("Cannot evolve: empty initial population and no template")
                raise ValueError("Empty initial population with no template to generate from")
        
        # Track best solution across all generations
        global_best_config = None
        global_best_fitness = float('-inf')
        
        # Run for specified number of generations
        for generation in range(generations):
            # Evaluate fitness for current population
            fitness_scores = [fitness_function(config) for config in population]
            
            # Find best in this generation
            best_idx = fitness_scores.index(max(fitness_scores))
            best_config = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Update global best
            if best_fitness > global_best_fitness:
                global_best_config = copy.deepcopy(best_config)
                global_best_fitness = best_fitness
                
            logger.info(f"Generation {generation+1}/{generations}: "
                        f"Best fitness = {best_fitness}, "
                        f"Avg fitness = {sum(fitness_scores)/len(fitness_scores)}")
            
            # Evolve to create next generation (except for last iteration)
            if generation < generations - 1:
                population = self.evolve_generation(population, fitness_scores)
        
        return global_best_config, global_best_fitness, population 