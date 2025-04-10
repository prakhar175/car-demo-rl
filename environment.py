import numpy as np

class GridEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size - 1, self.size - 1])
        return self._get_state()

    def _get_state(self):
        agent = self.agent_pos / self.size
        goal = self.goal_pos / self.size
        return np.concatenate((agent, goal))  # state is [agent_x, agent_y, goal_x, goal_y]

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:  # Up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:  # Down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size - 1:  # Right
            self.agent_pos[0] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.1
        return self._get_state(), reward, done
