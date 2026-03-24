import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx

class HiveMind:
    def __init__(self, n_agents: int = 10):
        self.n_agents = n_agents
        self.positions = np.zeros((n_agents, 2))  # 2D positions
        self.velocities = np.zeros((n_agents, 2))
        self.topology = nx.Graph()
        self._build_initial_topology()
        
    def _build_initial_topology(self) -> None:
        """Initialize the communication topology between agents"""
        self.topology.add_nodes_from(range(self.n_agents))
        # Initially connect to k-nearest neighbors
        for i in range(self.n_agents):
            dists = np.linalg.norm(self.positions - self.positions[i], axis=1)
            k_nearest = np.argsort(dists)[1:4]  # Connect to 3 nearest neighbors
            for j in k_nearest:
                self.topology.add_edge(i, j)

    def optimize_topology(self, objective_fn) -> None:
        """Dynamically optimize the swarm topology based on objective function"""
        current_score = objective_fn(self.topology)
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                # Try flipping edge status
                if self.topology.has_edge(i, j):
                    self.topology.remove_edge(i, j)
                else:
                    self.topology.add_edge(i, j)
                    
                new_score = objective_fn(self.topology)
                
                # Keep change only if it improves score
                if new_score <= current_score:
                    if self.topology.has_edge(i, j):
                        self.topology.remove_edge(i, j)
                    else:
                        self.topology.add_edge(i, j)

    def update_positions(self, dt: float = 0.1) -> None:
        """Update agent positions based on swarm dynamics"""
        # Compute forces between connected agents
        forces = np.zeros_like(self.positions)
        
        for i in range(self.n_agents):
            neighbors = list(self.topology.neighbors(i))
            if not neighbors:
                continue
                
            # Cohesion
            center = np.mean([self.positions[j] for j in neighbors], axis=0)
            forces[i] += (center - self.positions[i]) * 0.5
            
            # Separation
            for j in neighbors:
                diff = self.positions[i] - self.positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1.0:  # Minimum separation distance
                    forces[i] += diff / (dist * dist)
            
            # Alignment
            avg_vel = np.mean([self.velocities[j] for j in neighbors], axis=0)
            forces[i] += (avg_vel - self.velocities[i]) * 0.1

        # Update velocities and positions
        self.velocities += forces * dt
        # Limit velocities
        speed = np.linalg.norm(self.velocities, axis=1)
        mask = speed > 2.0
        self.velocities[mask] *= 2.0 / speed[mask, np.newaxis]
        
        self.positions += self.velocities * dt

    def get_state(self) -> Dict:
        """Return current state of the swarm"""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'topology': self.topology.copy()
        }

    def set_state(self, positions: Optional[np.ndarray] = None,
                  velocities: Optional[np.ndarray] = None) -> None:
        """Set swarm state explicitly"""
        if positions is not None:
            assert positions.shape == (self.n_agents, 2)
            self.positions = positions.copy()
        if velocities is not None:
            assert velocities.shape == (self.n_agents, 2)
            self.velocities = velocities.copy()
        self._build_initial_topology()  # Rebuild topology for new positions