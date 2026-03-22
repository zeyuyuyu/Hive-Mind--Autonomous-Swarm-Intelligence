import numpy as np

class SwarmOptimizer:
    def __init__(self, num_agents, search_space, objective_function, c1=2.0, c2=2.0, w=0.5):
        self.num_agents = num_agents
        self.search_space = search_space
        self.objective_function = objective_function
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.agents = self._initialize_agents()
        self.global_best = None
        self.global_best_fitness = float('-inf')

    def _initialize_agents(self):
        agents = []
        for _ in range(self.num_agents):
            agent = np.random.uniform(low=self.search_space[:, 0], high=self.search_space[:, 1])
            agents.append(agent)
        return agents

    def _update_agent(self, agent, best_agent, global_best):
        r1 = np.random.uniform(0, 1, size=len(agent))
        r2 = np.random.uniform(0, 1, size=len(agent))
        velocity = self.w * agent + self.c1 * r1 * (best_agent - agent) + self.c2 * r2 * (global_best - agent)
        new_agent = agent + velocity
        new_agent = np.clip(new_agent, self.search_space[:, 0], self.search_space[:, 1])
        return new_agent

    def optimize(self, max_iterations):
        for _ in range(max_iterations):
            for i, agent in enumerate(self.agents):
                fitness = self.objective_function(agent)
                if fitness > self.global_best_fitness:
                    self.global_best = agent
                    self.global_best_fitness = fitness
                self.agents[i] = self._update_agent(agent, agent, self.global_best)
        return self.global_best, self.global_best_fitness
