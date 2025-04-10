from environment import GridEnv
from dqn_agent import Agent
import matplotlib.pyplot as plt
import torch
env = GridEnv(size=10)
agent = Agent(state_size=4, action_size=4)

episodes = 500
scores = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.train()
    scores.append(total_reward)
    print(f"Episode {e+1}/{episodes} - Score: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
torch.save(agent.qnetwork.state_dict(), "trained_model.pth")

# Plot
plt.plot(scores, label="Score per Episode")
plt.plot([sum(scores[max(0, i-50):i+1])/len(scores[max(0, i-50):i+1]) for i in range(len(scores))], label="Moving Average", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.title("DQN Training Progress")
plt.grid(True)
plt.show()
