import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass

@dataclass
class Agent:
    id: int
    position: np.ndarray
    role: str = 'worker'
    specialization: float = 0.0
    current_task: Optional[str] = None

class SwarmIntelligence:
    def __init__(self, num_agents: int, dimensions: int = 2):
        self.agents = [
            Agent(
                id=i,
                position=np.random.uniform(-1, 1, dimensions)
            ) for i in range(num_agents)
        ]
        self.tasks: Dict[str, Dict] = {}
        self.role_behaviors: Dict[str, Callable] = {
            'worker': self._worker_behavior,
            'scout': self._scout_behavior,
            'specialist': self._specialist_behavior
        }
    
    def add_task(self, task_id: str, position: np.ndarray, complexity: float):
        self.tasks[task_id] = {
            'position': position,
            'complexity': complexity,
            'progress': 0.0,
            'assigned_agents': []
        }

    def update(self, dt: float):
        self._update_roles()
        self._allocate_tasks()
        
        for agent in self.agents:
            if agent.role in self.role_behaviors:
                self.role_behaviors[agent.role](agent, dt)

    def _update_roles(self):
        # Dynamic role adaptation based on task demands
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return

        # Calculate needed distribution
        desired_scouts = max(2, int(0.1 * len(self.agents)))
        desired_specialists = int(0.3 * len(self.agents))
        
        current_roles = {role: sum(1 for a in self.agents if a.role == role)
                        for role in self.role_behaviors.keys()}

        # Adjust roles to match desired distribution
        for agent in self.agents:
            if current_roles['scout'] < desired_scouts:
                if agent.role != 'scout':
                    agent.role = 'scout'
                    current_roles['scout'] += 1
                    current_roles[agent.role] -= 1
            elif current_roles['specialist'] < desired_specialists:
                if agent.role != 'specialist':
                    agent.role = 'specialist'
                    current_roles['specialist'] += 1
                    current_roles[agent.role] -= 1

    def _allocate_tasks(self):
        # Clear current assignments
        for task in self.tasks.values():
            task['assigned_agents'] = []

        # Assign tasks based on agent specialization and proximity
        available_agents = [a for a in self.agents if a.current_task is None]
        
        for task_id, task in self.tasks.items():
            if task['progress'] >= 1.0:
                continue
                
            # Find closest suitable agents
            distances = [np.linalg.norm(a.position - task['position']) 
                        for a in available_agents]
            
            # Assign based on distance and specialization
            num_needed = int(task['complexity'] * 3)  # Scale with complexity
            for _ in range(min(num_needed, len(available_agents))):
                if not distances:
                    break
                best_idx = np.argmin(distances)
                agent = available_agents.pop(best_idx)
                distances.pop(best_idx)
                
                agent.current_task = task_id
                task['assigned_agents'].append(agent.id)

    def _worker_behavior(self, agent: Agent, dt: float):
        if agent.current_task:
            task = self.tasks[agent.current_task]
            # Move toward task
            direction = task['position'] - agent.position
            distance = np.linalg.norm(direction)
            if distance > 0.01:
                agent.position += direction * dt
            else:
                # Contribute to task progress
                task['progress'] = min(1.0, task['progress'] + 0.1 * dt)

    def _scout_behavior(self, agent: Agent, dt: float):
        # Random exploration pattern
        agent.position += np.random.normal(0, 0.1, size=len(agent.position)) * dt
        # Keep within bounds
        agent.position = np.clip(agent.position, -1, 1)

    def _specialist_behavior(self, agent: Agent, dt: float):
        if agent.current_task:
            task = self.tasks[agent.current_task]
            # Specialists are more efficient at task completion
            direction = task['position'] - agent.position
            distance = np.linalg.norm(direction)
            if distance > 0.01:
                agent.position += direction * 1.5 * dt  # Move faster
            else:
                # Contribute more to task progress
                task['progress'] = min(1.0, task['progress'] + 0.2 * dt)
                agent.specialization = min(1.0, agent.specialization + 0.05 * dt)

    def get_swarm_state(self) -> Dict:
        return {
            'agents': [(a.id, a.position.tolist(), a.role) for a in self.agents],
            'tasks': self.tasks
        }