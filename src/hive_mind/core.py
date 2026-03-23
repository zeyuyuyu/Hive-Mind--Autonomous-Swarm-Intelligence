"""Core implementation of Hive Mind swarm intelligence."""

from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class SwarmAgent:
    """Individual agent in the swarm."""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float

class HiveMind:
    """Manages collective swarm intelligence and behavior."""
    
    def __init__(self, 
                 num_agents: int,
                 dimensions: int,
                 objective_fn: Callable,
                 bounds: tuple,
                 topology: str = 'dynamic'):
        self.num_agents = num_agents
        self.dimensions = dimensions
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.topology = topology
        
        # Initialize swarm
        self.agents: List[SwarmAgent] = []
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.initialize_swarm()
        
        # Adaptive parameters
        self.inertia = 0.9
        self.cognitive_factor = 2.0
        self.social_factor = 2.0
        self.adaptation_rate = 0.01
    
    def initialize_swarm(self) -> None:
        """Initialize swarm agents with random positions and velocities."""
        for i in range(self.num_agents):
            position = np.random.uniform(self.bounds[0], self.bounds[1], 
                                       self.dimensions)
            velocity = np.zeros(self.dimensions)
            score = self.objective_fn(position)
            
            agent = SwarmAgent(
                id=i,
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_score=score
            )
            
            self.agents.append(agent)
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = position.copy()
    
    def get_neighborhood(self, agent_id: int) -> List[SwarmAgent]:
        """Get neighboring agents based on topology."""
        if self.topology == 'global':
            return self.agents
        elif self.topology == 'ring':
            left = (agent_id - 1) % self.num_agents
            right = (agent_id + 1) % self.num_agents
            return [self.agents[left], self.agents[right]]
        elif self.topology == 'dynamic':
            # Dynamic topology based on spatial proximity
            distances = []
            for other in self.agents:
                if other.id != agent_id:
                    dist = np.linalg.norm(self.agents[agent_id].position - 
                                         other.position)
                    distances.append((dist, other))
            distances.sort()
            return [agent for _, agent in distances[:3]]
    
    def adapt_parameters(self) -> None:
        """Dynamically adapt swarm parameters based on performance."""
        if self.global_best_score > 0:
            self.inertia *= (1 - self.adaptation_rate)
            self.cognitive_factor *= (1 + self.adaptation_rate)
        else:
            self.inertia *= (1 + self.adaptation_rate)
            self.social_factor *= (1 + self.adaptation_rate)
            
        # Enforce bounds
        self.inertia = np.clip(self.inertia, 0.4, 0.9)
        self.cognitive_factor = np.clip(self.cognitive_factor, 1.5, 2.5)
        self.social_factor = np.clip(self.social_factor, 1.5, 2.5)
    
    def update(self) -> None:
        """Update swarm positions and velocities for one iteration."""
        for agent in self.agents:
            # Get neighborhood best
            neighborhood = self.get_neighborhood(agent.id)
            neighborhood_best = min(neighborhood, 
                                  key=lambda x: x.best_score)
            
            # Update velocity
            cognitive = np.random.random(self.dimensions) * self.cognitive_factor
            social = np.random.random(self.dimensions) * self.social_factor
            
            agent.velocity = (self.inertia * agent.velocity + 
                            cognitive * (agent.best_position - agent.position) +
                            social * (neighborhood_best.best_position - 
                                     agent.position))
            
            # Update position
            agent.position += agent.velocity
            agent.position = np.clip(agent.position, 
                                    self.bounds[0], 
                                    self.bounds[1])
            
            # Evaluate new position
            score = self.objective_fn(agent.position)
            
            # Update personal best
            if score < agent.best_score:
                agent.best_score = score
                agent.best_position = agent.position.copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = agent.position.copy()
        
        # Adapt parameters
        self.adapt_parameters()
    
    def optimize(self, max_iterations: int) -> tuple:
        """Run swarm optimization for specified iterations."""
        for _ in range(max_iterations):
            self.update()
        return self.global_best_position, self.global_best_score
