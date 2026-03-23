import numpy as np
from typing import List, Tuple

class SwarmCoordinator:
    def __init__(self, num_agents: int, communication_radius: float):
        self.num_agents = num_agents
        self.communication_radius = communication_radius
        self.agent_positions = np.zeros((num_agents, 2))
        self.agent_decisions = np.zeros(num_agents, dtype=int)

    def update_agent_positions(self, positions: np.ndarray):
        self.agent_positions = positions

    def get_connected_agents(self, agent_id: int) -> List[int]:
        connected_agents = []
        agent_pos = self.agent_positions[agent_id]
        for i in range(self.num_agents):
            if i != agent_id and np.linalg.norm(self.agent_positions[i] - agent_pos) <= self.communication_radius:
                connected_agents.append(i)
        return connected_agents

    def make_distributed_decision(self) -> Tuple[np.ndarray, np.ndarray]:
        for agent_id in range(self.num_agents):
            connected_agents = self.get_connected_agents(agent_id)
            local_decisions = self.agent_decisions[connected_agents]
            self.agent_decisions[agent_id] = np.argmax(np.bincount(local_decisions))

        return self.agent_positions, self.agent_decisions
