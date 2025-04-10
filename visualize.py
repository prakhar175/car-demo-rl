import pygame
import torch
import time
from environment import GridEnv
from dqn_agent import QNetwork
import numpy as np

# === Load Trained Model ===
model = QNetwork(state_size=4, action_size=4)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# === Initialize Environment ===
env = GridEnv(size=10)
grid_size = env.size
cell_size = 60
window_size = grid_size * cell_size

# === Pygame Setup ===
pygame.init()
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("DQN Agent Visualization")
clock = pygame.time.Clock()

# === Colors ===
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)      # Car
GREEN = (0, 255, 0)     # Goal
BLACK = (0, 0, 0)

def draw_grid():
    for x in range(0, window_size, cell_size):
        pygame.draw.line(screen, GRAY, (x, 0), (x, window_size))
    for y in range(0, window_size, cell_size):
        pygame.draw.line(screen, GRAY, (0, y), (window_size, y))

def draw_agent(pos):
    pygame.draw.rect(screen, BLUE, (pos[1]*cell_size+10, pos[0]*cell_size+10, cell_size-20, cell_size-20))

def draw_goal(pos):
    pygame.draw.rect(screen, GREEN, (pos[1]*cell_size+10, pos[0]*cell_size+10, cell_size-20, cell_size-20))

# === Start Visualization ===
state = env.reset()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill(WHITE)
    draw_grid()
    draw_goal((state[2], state[3]))  # goal_x, goal_y
    draw_agent(env.agent_pos)
    # === DQN chooses action ===
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    # === Step in environment ===
    state, reward, done = env.step(action)

    pygame.display.flip()
    clock.tick(3)  # Slow down to 3 FPS so you can see movement
    time.sleep(0.3)

# === Done ===
print("Goal reached!")
time.sleep(2)
pygame.quit()
