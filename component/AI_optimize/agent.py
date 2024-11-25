#! /usr/bin/env python3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Importiere deine Umgebung
from your_env_file import TrajectoryOptimizationEnv

# Umgebung erstellen
env = TrajectoryOptimizationEnv()
check_env(env)  # Prüft die Umgebung auf Kompatibilität

# Umgebung in einen Vektor-Wrapper packen (wird von Stable-Baselines benötigt)
vec_env = DummyVecEnv([lambda: env])

# RL-Agenten erstellen
model = PPO("MlpPolicy", vec_env, verbose=1)

# Training
model.learn(total_timesteps=10000)

# Modell speichern
model.save("ppo_trajectory_optimizer")
