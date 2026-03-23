import random

class HiveMind:
    def __init__(self, num_agents, environment):
        self.agents = [Agent(self, environment) for _ in range(num_agents)]
        self.environment = environment

    def run(self):
        while True:
            for agent in self.agents:
                agent.sense()
                agent.decide()
                agent.act()
            self.environment.update()

class Agent:
    def __init__(self, hive_mind, environment):
        self.hive_mind = hive_mind
        self.environment = environment
        self.position = (random.uniform(-10, 10), random.uniform(-10, 10))
        self.velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.goal = None

    def sense(self):
        # Sense nearby agents and environment
        self.nearby_agents = [agent for agent in self.hive_mind.agents if self.distance(agent) < 5]
        self.nearby_resources = [resource for resource in self.environment.resources if self.distance(resource) < 2]

    def decide(self):
        # Coordinate with nearby agents and decide on a goal
        if not self.goal:
            self.goal = self.find_nearest_resource()
            for agent in self.nearby_agents:
                if agent.goal == self.goal:
                    self.goal = self.find_alternative_resource()
                    break

    def act(self):
        # Move towards the goal
        dx, dy = self.goal[0] - self.position[0], self.goal[1] - self.position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        if distance > 0.1:
            self.velocity = (self.velocity[0] + dx / distance * 0.1, self.velocity[1] + dy / distance * 0.1)
            self.position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])

    def distance(self, other):
        # Calculate the distance to another agent or resource
        dx, dy = self.position[0] - other.position[0], self.position[1] - other.position[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def find_nearest_resource(self):
        # Find the nearest resource
        nearest_resource = min(self.environment.resources, key=self.distance)
        return nearest_resource.position

    def find_alternative_resource(self):
        # Find an alternative resource that is not being targeted by nearby agents
        for resource in self.environment.resources:
            if resource.position != self.goal and all(agent.goal != resource.position for agent in self.nearby_agents):
                return resource.position
        return self.find_nearest_resource()
