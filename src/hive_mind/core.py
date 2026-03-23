"""Core swarm intelligence implementation with dynamic topology and role adaptation."""
import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import networkx as nx

@dataclass
class SwarmAgent:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    role: str = 'worker'
    fitness: float = 0.0
    
class HiveMind:
    def __init__(self, n_agents: int, dimensions: int):
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.agents = []
        self.topology = nx.Graph()
        self.roles = ['scout', 'worker', 'coordinator']
        self.role_behaviors: Dict[str, Callable] = {
            'scout': self._scout_behavior,
            'worker': self._worker_behavior,
            'coordinator': self._coordinator_behavior
        }
        self._initialize_swarm()
        
    def _initialize_swarm(self):
        for i in range(self.n_agents):
            agent = SwarmAgent(
                id=i,
                position=np.random.uniform(-1, 1, self.dimensions),
                velocity=np.random.uniform(-0.1, 0.1, self.dimensions)
            )
            self.agents.append(agent)
            self.topology.add_node(i)
            
    def update_topology(self):
        """Dynamically update swarm communication topology based on spatial proximity."""
        self.topology.clear_edges()
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(
                    self.agents[i].position - self.agents[j].position
                )
                if dist < 0.3:  # Proximity threshold
                    self.topology.add_edge(i, j, weight=1.0/dist)
                    
    def adapt_roles(self):
        """Dynamically assign roles based on agent positions and network centrality."""
        centrality = nx.eigenvector_centrality(self.topology, max_iter=1000)
        for agent in self.agents:
            if centrality[agent.id] > 0.8:
                agent.role = 'coordinator'
            elif np.linalg.norm(agent.position) > 0.8:
                agent.role = 'scout'
            else:
                agent.role = 'worker'
                
    def _scout_behavior(self, agent: SwarmAgent):
        """Explore the environment periphery."""
        exploration_vector = np.random.uniform(-1, 1, self.dimensions)
        return exploration_vector * 0.1
        
    def _worker_behavior(self, agent: SwarmAgent):
        """Follow local consensus and maintain cohesion."""
        neighbors = list(self.topology.neighbors(agent.id))
        if not neighbors:
            return np.zeros(self.dimensions)
            
        center = np.mean([self.agents[i].position for i in neighbors], axis=0)
        cohesion = center - agent.position
        return cohesion * 0.05
        
    def _coordinator_behavior(self, agent: SwarmAgent):
        """Guide nearby agents and maintain swarm structure."""
        neighbors = list(self.topology.neighbors(agent.id))
        if not neighbors:
            return np.zeros(self.dimensions)
            
        # Influence neighbors while maintaining optimal spacing
        spacing_vector = np.zeros(self.dimensions)
        for neighbor_id in neighbors:
            neighbor = self.agents[neighbor_id]
            diff = agent.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                spacing_vector += diff/dist * (0.2 - dist)
        return spacing_vector * 0.1
        
    def step(self):
        """Execute one time step of swarm behavior."""
        self.update_topology()
        self.adapt_roles()
        
        # Update all agents
        for agent in self.agents:
            # Get behavior vector based on role
            behavior_vector = self.role_behaviors[agent.role](agent)
            
            # Update velocity with behavior vector and inertia
            agent.velocity = agent.velocity * 0.9 + behavior_vector
            
            # Update position
            agent.position += agent.velocity
            
            # Bound position
            agent.position = np.clip(agent.position, -1, 1)
            
    def get_swarm_state(self) -> Dict:
        """Return current state of the swarm."""
        return {
            'positions': np.array([a.position for a in self.agents]),
            'velocities': np.array([a.velocity for a in self.agents]),
            'roles': [a.role for a in self.agents],
            'topology': self.topology.edges()
        }