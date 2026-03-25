import numpy as np
from typing import List, Dict, Set, Optional
import networkx as nx

class SwarmAgent:
    def __init__(self, agent_id: int, position: np.ndarray):
        self.id = agent_id
        self.position = position
        self.velocity = np.zeros_like(position)
        self.neighbors: Set[int] = set()
        self.state: Dict = {}

class AdaptiveSwarm:
    def __init__(self, n_agents: int, dimensions: int = 2, communication_range: float = 10.0):
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.communication_range = communication_range
        self.agents: Dict[int, SwarmAgent] = {}
        self.topology = nx.Graph()
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        for i in range(self.n_agents):
            position = np.random.uniform(-50, 50, self.dimensions)
            self.agents[i] = SwarmAgent(i, position)
            self.topology.add_node(i)
    
    def update_topology(self):
        """Dynamically update network topology based on agent positions"""
        # Reset edges
        self.topology.clear_edges()
        
        # Update neighbor connections based on proximity
        for i in self.agents:
            self.agents[i].neighbors.clear()
            for j in self.agents:
                if i != j:
                    distance = np.linalg.norm(
                        self.agents[i].position - self.agents[j].position
                    )
                    if distance <= self.communication_range:
                        self.agents[i].neighbors.add(j)
                        self.topology.add_edge(i, j, weight=distance)

    def get_local_centroid(self, agent_id: int) -> np.ndarray:
        """Calculate centroid of local neighborhood"""
        if not self.agents[agent_id].neighbors:
            return self.agents[agent_id].position
            
        positions = [self.agents[n].position for n in self.agents[agent_id].neighbors]
        return np.mean(positions, axis=0)

    def apply_cohesion(self, agent_id: int, factor: float = 0.1) -> np.ndarray:
        """Force moving agent toward local neighborhood centroid"""
        centroid = self.get_local_centroid(agent_id)
        return factor * (centroid - self.agents[agent_id].position)

    def apply_separation(self, agent_id: int, min_distance: float = 5.0, factor: float = 0.2) -> np.ndarray:
        """Force pushing agents away from too-close neighbors"""
        separation = np.zeros(self.dimensions)
        for neighbor_id in self.agents[agent_id].neighbors:
            diff = self.agents[agent_id].position - self.agents[neighbor_id].position
            distance = np.linalg.norm(diff)
            if distance < min_distance:
                separation += (diff / distance) * (min_distance - distance)
        return factor * separation

    def apply_alignment(self, agent_id: int, factor: float = 0.1) -> np.ndarray:
        """Force aligning agent velocity with neighbors"""
        if not self.agents[agent_id].neighbors:
            return np.zeros(self.dimensions)
            
        velocities = [self.agents[n].velocity for n in self.agents[agent_id].neighbors]
        avg_velocity = np.mean(velocities, axis=0)
        return factor * (avg_velocity - self.agents[agent_id].velocity)

    def step(self, dt: float = 0.1):
        """Advance simulation by one timestep"""
        self.update_topology()
        
        # Calculate new velocities
        new_velocities = {}
        for agent_id in self.agents:
            cohesion = self.apply_cohesion(agent_id)
            separation = self.apply_separation(agent_id)
            alignment = self.apply_alignment(agent_id)
            
            new_velocity = self.agents[agent_id].velocity + cohesion + separation + alignment
            # Limit velocity magnitude
            speed = np.linalg.norm(new_velocity)
            if speed > 10.0:
                new_velocity = (new_velocity / speed) * 10.0
            new_velocities[agent_id] = new_velocity
        
        # Update positions and velocities
        for agent_id in self.agents:
            self.agents[agent_id].velocity = new_velocities[agent_id]
            self.agents[agent_id].position += self.agents[agent_id].velocity * dt

    def get_network_metrics(self) -> Dict:
        """Calculate key network topology metrics"""
        return {
            'avg_degree': np.mean([d for _, d in self.topology.degree()]),
            'clustering': nx.average_clustering(self.topology),
            'components': nx.number_connected_components(self.topology),
            'density': nx.density(self.topology)
        }

    def get_positions(self) -> np.ndarray:
        """Return array of all agent positions"""
        return np.array([agent.position for agent in self.agents.values()])
