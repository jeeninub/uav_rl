import gym
import numpy as np
import os
import matplotlib.pyplot as plt

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        self.grid_size = np.array([20, 20, 5])  # 3D space

        # Load synthetic data
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.users = np.load(os.path.join(data_dir, "users.npy"))
        self.obstacles = np.load(os.path.join(data_dir, "obstacles.npy"), allow_pickle=True)
        self.snr_map = np.load(os.path.join(data_dir, "channel_map.npy"))

        # Action = 7 directions
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.max_steps = 100
        self.reset()

    def reset(self):
        self.uav_pos = np.array([0, 0, 1])
        self.goal = np.array([19, 19, 1])
        self.battery = 1.0
        self.coverage = 0.0
        self.visited = set()
        self.episode_steps = 0
        return self._get_state()

    def _get_state(self):
        x, y, z = self.uav_pos.astype(int)
        snr = self.snr_map[x, y, z]
        dist = np.linalg.norm(self.goal - self.uav_pos) / np.linalg.norm(self.grid_size)

        proximity = 1.0
        if len(self.obstacles) > 0:
            proximity = np.min([
                np.linalg.norm(self.uav_pos - np.array(ob[0]))
                for ob in self.obstacles
            ]) / np.linalg.norm(self.grid_size)

        return np.array([
            *self.uav_pos / self.grid_size,
            self.battery,
            snr,
            self.coverage,
            dist,
            proximity
        ], dtype=np.float32)

    def step(self, action):
        moves = {
            0: np.array([0, 1, 0]),    # ↑
            1: np.array([0, -1, 0]),   # ↓
            2: np.array([-1, 0, 0]),   # ←
            3: np.array([1, 0, 0]),    # →
            4: np.array([1, 1, 0]),    # ↗
            5: np.array([-1, -1, 0]),  # ↘
            6: np.array([0, 0, 0])     # hover
        }

        self.uav_pos = np.clip(self.uav_pos + moves[action], [0, 0, 0], self.grid_size - 1)
        self.battery -= 0.01
        self.episode_steps += 1

        reward = self._compute_reward()

        done = self.battery <= 0 or np.array_equal(self.uav_pos, self.goal) or self.episode_steps >= self.max_steps

        if self.battery <= 0:
            reason = "battery"
        elif np.array_equal(self.uav_pos, self.goal):
            reason = "goal"
        elif self.episode_steps >= self.max_steps:
            reason = "max_steps"
        else:
            reason = "unknown"

        info = {"done_reason": reason}
        return self._get_state(), reward, done, info

    def _compute_reward(self):
        x, y, z = self.uav_pos.astype(int)
        snr = self.snr_map[x, y, z]

        new_pos = tuple(self.uav_pos)
        area_gain = 1.0 if new_pos not in self.visited else 0.0
        self.visited.add(new_pos)
        self.coverage = len(self.visited) / (self.grid_size[0] * self.grid_size[1])

        proximity = np.min([
            np.linalg.norm(self.uav_pos - np.array(ob[0]))
            for ob in self.obstacles
        ]) if len(self.obstacles) > 0 else 999

        collision_risk = 1.0 if proximity < 1.5 else 0.0

        α, β, γ, δ = 1.0, 0.8, 1.5, 0.3
        reward = α * (snr > 0.6) + β * area_gain - γ * collision_risk - δ * 0.01
        return reward

    def render(self):
        print(f"Step {self.episode_steps}: Position {self.uav_pos} | Battery {self.battery:.2f}")

    def render_map(self):
        grid_x, grid_y, _ = self.grid_size
        plt.figure(figsize=(6, 6))
        plt.xlim(0, grid_x)
        plt.ylim(0, grid_y)

        # Users
        ux, uy = self.users[:, 0], self.users[:, 1]
        plt.scatter(ux, uy, c="blue", s=20, label="Users")

        # Obstacles
        ox, oy = zip(*[ob[0][:2] for ob in self.obstacles]) if len(self.obstacles) > 0 else ([], [])
        plt.scatter(ox, oy, c="red", s=30, label="Obstacles")

        # UAV
        x, y = self.uav_pos[0], self.uav_pos[1]
        plt.scatter(x, y, c="green", s=80, marker="X", label="UAV", edgecolors='black')

        # Goal
        gx, gy = self.goal[0], self.goal[1]
        plt.scatter(gx, gy, c="orange", s=80, marker="*", label="Goal")

        plt.title(f"UAV Map - Step {self.episode_steps}")
        plt.legend()
        plt.grid(True)
        plt.show()
