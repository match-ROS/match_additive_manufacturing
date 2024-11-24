#! /usr/bin/env python3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Lade die Trajektorien
from print_path import xTCP, yTCP, xMIR, yMIR
tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T
base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T

# Umgebung erstellen
env = TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory)
check_env(env)  # Prüft die Umgebung auf Kompatibilität

# RL-Agenten initialisieren
model = PPO("MlpPolicy", env, verbose=1)

# Training
model.learn(total_timesteps=10000)

# Optimierte Trajektorie anwenden
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
