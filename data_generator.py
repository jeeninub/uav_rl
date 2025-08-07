import numpy as np
import os

os.makedirs("data", exist_ok=True)

GRID_X, GRID_Y, GRID_Z = 20, 20, 5

# Users
users = np.random.randint(low=[0, 0], high=[GRID_X, GRID_Y], size=(50, 2))
np.save("data/users.npy", users)

# Obstacles
obstacles = np.random.randint(low=[0, 0, 0], high=[GRID_X, GRID_Y, GRID_Z], size=(30, 3))
np.save("data/obstacles.npy", obstacles)

# Channel map (SNR)
snr_map = np.random.uniform(0.4, 0.95, size=(GRID_X, GRID_Y, GRID_Z))
np.save("data/channel_map.npy", snr_map)

print("✅ Synthetic data generated.")

#checking if it is uploaded on git or not