import numpy as np
from typing import List, Tuple, Optional, Dict
import random

class Agent:
    def __init__(self, position: Tuple[float, float], agent_id: int):
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.best_position = self.position.copy()
        self.best_score = float('-inf')
        self.id = agent_id
        self.pheromone_strength = 1.0

class PheromoneGrid:
    def __init__(self, width: int, height: int, evaporation_rate: float = 0.1):
        self.grid = np.zeros((height, width))
        self.evaporation_rate = evaporation_rate
    
    def deposit(self, position: Tuple[int, int], strength: float):
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            self.grid[y, x] += strength
    
    def evaporate(self):
        self.grid *= (1 - self.evaporation_rate)

class HiveMind:
    def __init__(self, num_agents: int, bounds: Tuple[float, float, float, float]):
        self.agents: List[Agent] = []
        self.num_agents = num_agents
        self.bounds = bounds
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.pheromone_grid = PheromoneGrid(
            width=100,
            height=100,
            evaporation_rate=0.05
        )
        self.initialize_agents()

    def initialize_agents(self):
        for i in range(self.num_agents):
            position = (
                random.uniform(self.bounds[0], self.bounds[1]),
                random.uniform(self.bounds[2], self.bounds[3])
            )
            self.agents.append(Agent(position, i))

    def update_agent(self, agent: Agent, objective_func) -> None:
        # Social influence parameters
        inertia = 0.8
        cognitive = 1.5
        social = 1.5
        pheromone_influence = 0.3

        # Calculate pheromone gradient
        grid_x = int((agent.position[0] - self.bounds[0]) * 99 / (self.bounds[1] - self.bounds[0]))
        grid_y = int((agent.position[1] - self.bounds[2]) * 99 / (self.bounds[3] - self.bounds[2]))
        
        pheromone_gradient = np.zeros(2)
        if 0 <= grid_x < 99 and 0 <= grid_y < 99:
            dx = self.pheromone_grid.grid[grid_y, grid_x + 1] - self.pheromone_grid.grid[grid_y, grid_x]
            dy = self.pheromone_grid.grid[grid_y + 1, grid_x] - self.pheromone_grid.grid[grid_y, grid_x]
            pheromone_gradient = np.array([dx, dy])

        # Update velocity
        r1, r2 = random.random(), random.random()
        cognitive_velocity = cognitive * r1 * (agent.best_position - agent.position)
        social_velocity = social * r2 * (self.global_best_position - agent.position)
        pheromone_velocity = pheromone_influence * pheromone_gradient

        agent.velocity = (inertia * agent.velocity + 
                        cognitive_velocity + 
                        social_velocity + 
                        pheromone_velocity)

        # Update position
        agent.position += agent.velocity
        
        # Enforce boundaries
        agent.position = np.clip(
            agent.position,
            [self.bounds[0], self.bounds[2]],
            [self.bounds[1], self.bounds[3]]
        )

        # Evaluate new position
        score = objective_func(agent.position)
        
        # Update personal best
        if score > agent.best_score:
            agent.best_score = score
            agent.best_position = agent.position.copy()
            
            # Deposit more pheromone for better solutions
            self.pheromone_grid.deposit(
                (grid_x, grid_y),
                agent.pheromone_strength * (1 + score)
            )

        # Update global best
        if score > self.global_best_score:
            self.global_best_score = score
            self.global_best_position = agent.position.copy()

    def optimize(self, objective_func, iterations: int) -> Tuple[np.ndarray, float]:
        self.global_best_position = self.agents[0].position.copy()
        
        for _ in range(iterations):
            for agent in self.agents:
                self.update_agent(agent, objective_func)
            
            # Evaporate pheromones
            self.pheromone_grid.evaporate()
            
        return self.global_best_position, self.global_best_score