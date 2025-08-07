# main.py

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from env.uav_env import UAVEnv
from agents.dqnn_agent import DQNNAgent
from replay_buffer import ReplayBuffer

# === Init ===
env = UAVEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNNAgent(state_dim, action_dim, dueling=True, double_dqn=True)
replay_buffer = ReplayBuffer(capacity=10000)

episodes = 200
batch_size = 64
reward_history = []

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === Training Loop ===
for ep in range(episodes):
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        if done:
            print(f"Episode {ep+1} ended due to: {info['done_reason']}")

        # --- Normalize Reward ---
        reward = reward / 100.0

        # Store and train
        replay_buffer.push(state, action, reward, next_state, done)
        agent.train(replay_buffer, batch_size)
        agent.soft_update_target_network()  # ← Soft update every step

        state = next_state
        ep_reward += reward

    reward_history.append(ep_reward)
    print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward * 100:.2f} | Epsilon: {agent.epsilon:.3f}")

# === Save model ===
torch.save(agent.q_net.state_dict(), "models/dqnn_trained.pth")

# === Save reward plot ===
plt.plot([r * 100 for r in reward_history])
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQNN Training Reward Curve")
plt.grid(True)
plt.savefig("results/reward_curve.png")
plt.show()

# === Evaluation Run ===
state = env.reset()
done = False
eval_reward = 0
agent.epsilon = 0.0  # Pure exploitation

while not done:
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    eval_reward += reward
    env.render()

print(f"Evaluation Reward: {eval_reward * 100:.2f}")
