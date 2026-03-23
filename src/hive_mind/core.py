import random
import numpy as np
from typing import List, Tuple

class SwarmAgent:
    def __init__(self, id: str, position: Tuple[float, float], sensor_range: float, communication_range: float):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range
        self.communication_range = communication_range
        self.neighbors = []
        self.shared_data = {}

    def sense_environment(self, environment: 'Environment'):
        self.neighbors = environment.get_nearby_agents(self.position, self.sensor_range)

    def communicate(self):
        for neighbor in self.neighbors:
            if np.linalg.norm(np.array(self.position) - np.array(neighbor.position)) <= self.communication_range:
                self.shared_data.update(neighbor.shared_data)
                neighbor.shared_data.update(self.shared_data)

    def make_decision(self) -> List[float]:
        # Implement distributed decision making logic here
        # utilizing the shared data from neighboring agents
        return [random.uniform(-1, 1), random.uniform(-1, 1)]

    def update_position(self, action: List[float]):
        self.position = (self.position[0] + action[0], self.position[1] + action[1])

class Environment:
    def __init__(self, size: Tuple[float, float]):
        self.size = size
        self.agents: List[SwarmAgent] = []

    def add_agent(self, agent: SwarmAgent):
        self.agents.append(agent)

    def get_nearby_agents(self, position: Tuple[float, float], range: float) -> List[SwarmAgent]:
        return [agent for agent in self.agents if np.linalg.norm(np.array(position) - np.array(agent.position)) <= range]

    def update(self):
        for agent in self.agents:
            agent.sense_environment(self)
            agent.communicate()
            action = agent.make_decision()
            agent.update_position(action)
