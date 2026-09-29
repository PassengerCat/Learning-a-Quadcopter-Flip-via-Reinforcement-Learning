import gymnasium as gym
import numpy as np
from quad_velocity_env import QuadcopterVelocityEnv

env_full = QuadcopterVelocityEnv(
    obs_mode="full",
    observation_noise_std={
        "pos": 0.005,
        "quat": 0.001,
        "vel": 0.03,
        "omega": 0.01,
        "motor": 2.0,
    },
    random_wind=True,
    wind_magnitude=2.0,
)

obs, info = env_full.reset(
    seed=0,
    options={"target_velocity": [1.0, 0.0, 0.3]},
)

np.set_printoptions(precision=2, suppress=True, linewidth=150)

for _ in range(1000):
    action = env_full.action_space.sample()
    obs, reward, terminated, truncated, info = env_full.step(action)
    if terminated or truncated:
        break
