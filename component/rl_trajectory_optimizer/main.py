#! /usr/bin/env python3
import numpy as np
from stable_baselines3 import PPO
from trajectory_env import TrajectoryOptimizationEnv
import os, sys

parent_dir = os.path.dirname(os.path.abspath(__file__))


print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir)


from print_path import xMIR, yMIR, xTCP, yTCP

# Lade die Trajektorien
tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T
base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T

# Umgebung erstellen
env = TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory)

# RL-Agent initialisieren
model = PPO("MlpPolicy", env, verbose=1)

# Training starten
print("Starte Training...")
model.learn(total_timesteps=10000)
print("Training abgeschlossen!")

# Modell speichern
model.save("ppo_trajectory_optimizer")
print("Modell gespeichert.")

# Optimierte Trajektorie testen
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
