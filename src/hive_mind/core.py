import os
import numpy as np
from collections import deque
from .agent import Agent
from .governance import GovernanceProtocol

class SwarmManager:
    def __init__(self, num_agents, initial_state, reward_function):
        self.agents = [Agent(initial_state) for _ in range(num_agents)]
        self.governance = GovernanceProtocol()
        self.reward_function = reward_function
        self.state_history = deque(maxlen=1000)

    def step(self):
        for agent in self.agents:
            agent.update(self.governance.get_state())
            reward = self.reward_function(agent.state)
            agent.learn(reward)
            self.state_history.append(agent.state)

        self.governance.update(self.state_history)

if __name__ == '__main__':
    swarm = SwarmManager(num_agents=100, initial_state=np.random.rand(10), reward_function=lambda x: np.sum(x))
    while True:
        swarm.step()