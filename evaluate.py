from env.uav_env import UAVEnv
from agents.dqnn_agent import DQNNAgent
import torch

env = UAVEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load("models/dqnn_trained.pth"+))
agent.q_net.eval()
agent.epsilon = 0.0  # Pure exploitation

state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Total reward in eval: {total_reward:.2f}")
