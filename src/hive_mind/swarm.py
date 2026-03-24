from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class FormationType(Enum):
    CIRCLE = 'circle'
    GRID = 'grid' 
    V_FORMATION = 'v_formation'
    DYNAMIC = 'dynamic'

@dataclass
class SwarmAgent:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    neighbors: List[int] = None
    
class SwarmFormation:
    def __init__(self, num_agents: int, formation_type: FormationType):
        self.num_agents = num_agents
        self.formation_type = formation_type
        self.agents: List[SwarmAgent] = []
        self.separation_weight = 1.0
        self.cohesion_weight = 1.0
        self.alignment_weight = 1.0
        
    def initialize_swarm(self) -> None:
        """Initialize swarm agents in specified formation"""
        for i in range(self.num_agents):
            pos = self._get_initial_position(i)
            vel = np.random.randn(2) * 0.1
            self.agents.append(SwarmAgent(id=i, position=pos, velocity=vel))
            
    def _get_initial_position(self, index: int) -> np.ndarray:
        if self.formation_type == FormationType.CIRCLE:
            theta = (2 * np.pi * index) / self.num_agents
            x = 10 * np.cos(theta)
            y = 10 * np.sin(theta)
            return np.array([x, y])
        elif self.formation_type == FormationType.GRID:
            side = int(np.ceil(np.sqrt(self.num_agents)))
            x = (index % side) * 2
            y = (index // side) * 2
            return np.array([x, y])
        elif self.formation_type == FormationType.V_FORMATION:
            x = index * 2
            y = abs(index - self.num_agents//2) * 2
            return np.array([x, y])
        else:
            return np.random.randn(2) * 10

    def update(self, dt: float = 0.1) -> None:
        """Update swarm positions using flocking behavior rules"""
        for agent in self.agents:
            separation = self._separation_force(agent)
            cohesion = self._cohesion_force(agent)
            alignment = self._alignment_force(agent)
            
            # Combined force
            force = (separation * self.separation_weight + 
                    cohesion * self.cohesion_weight +
                    alignment * self.alignment_weight)
            
            # Update velocity and position
            agent.velocity += force * dt
            # Limit velocity magnitude
            speed = np.linalg.norm(agent.velocity)
            if speed > 5.0:
                agent.velocity = (agent.velocity / speed) * 5.0
            agent.position += agent.velocity * dt

    def _separation_force(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate separation force to avoid crowding"""
        force = np.zeros(2)
        for other in self.agents:
            if other.id != agent.id:
                diff = agent.position - other.position
                dist = np.linalg.norm(diff)
                if dist < 2.0:  # Separation radius
                    force += diff / (dist * dist)
        return force

    def _cohesion_force(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate cohesion force to move toward center of mass"""
        center = np.zeros(2)
        count = 0
        for other in self.agents:
            if other.id != agent.id:
                dist = np.linalg.norm(other.position - agent.position)
                if dist < 5.0:  # Cohesion radius
                    center += other.position
                    count += 1
        if count > 0:
            center /= count
            return (center - agent.position) * 0.1
        return np.zeros(2)

    def _alignment_force(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate alignment force to match velocities"""
        avg_velocity = np.zeros(2)
        count = 0
        for other in self.agents:
            if other.id != agent.id:
                dist = np.linalg.norm(other.position - agent.position)
                if dist < 5.0:  # Alignment radius
                    avg_velocity += other.velocity
                    count += 1
        if count > 0:
            avg_velocity /= count
            return (avg_velocity - agent.velocity) * 0.1
        return np.zeros(2)

    def get_positions(self) -> List[Tuple[float, float]]:
        """Return current positions of all agents"""
        return [(agent.position[0], agent.position[1]) for agent in self.agents]