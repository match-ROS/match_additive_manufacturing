#! /usr/bin/env python3
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
#from trajectory_env import TrajectoryOptimizationEnv
from trajectory_env_pose_interpolation import TrajectoryOptimizationEnv
import os, sys
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import tensorflow as tf
import os
if not os.path.exists("./ppo_tensorboard_logs/"):
    os.makedirs("./ppo_tensorboard_logs/")


parent_dir = os.path.dirname(os.path.abspath(__file__))


print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir)

from stable_baselines3.common.vec_env import SubprocVecEnv

# Funktion zum Erstellen einer neuen Umgebung
def make_env():
    return TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory)


from print_path import xMIR, yMIR, xTCP, yTCP

    
class ResetTrajectoryCallback(BaseCallback):
    def __init__(self, reset_freq, verbose=0):
        super(ResetTrajectoryCallback, self).__init__(verbose)
        self.reset_freq = reset_freq

    def _on_step(self) -> bool:
        # Überprüfe, ob die Umgebung zurückgesetzt werden soll
        if self.num_timesteps % self.reset_freq == 0:
            self.training_env.env_method("reset")
        return True
    

class TrajectoryPlotCallback(BaseCallback):
    def __init__(self, env, tcp_trajectory, plot_freq=1, verbose=0):
        super(TrajectoryPlotCallback, self).__init__(verbose)
        self.env = env
        self.tcp_trajectory = tcp_trajectory
        self.plot_freq = plot_freq

    def _on_step(self) -> bool:
        # Plotten nach jeder `plot_freq` Anzahl von Schritten
        if self.n_calls % self.plot_freq == 0:
            obs = self.env.reset()
            trajectory = []
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                trajectory.append(obs)

            # Trajektorie plotten
            trajectory = np.array(trajectory)
            plt.figure(figsize=(8, 6))
            plt.plot(trajectory[:, 0], trajectory[:, 1], label="Optimierte Trajektorie")
            plt.scatter(self.tcp_trajectory[:, 0], self.tcp_trajectory[:, 1], c='red', label="TCP Trajektorie")
            plt.legend()
            plt.title(f"Trajektorienvergleich nach {self.n_calls} Schritten")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        return True


if __name__ == "__main__":
    # Lade die Trajektorien
    tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T
    base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T

    log_dir = "./ppo_tensorboard_logs/"  # Einheitliches Verzeichnis
    new_logger = configure(log_dir, ["stdout", "tensorboard"])



    env = TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory,time_step=0.1)
    check_env(env, warn=True)

    # obs = env.reset()
    # action = env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action)

    # Anzahl der parallelen Umgebungen
    num_cpu = 8
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    env.max_steps_per_episode = 1000 # Kürzere Episoden für schnelleres Training
    # env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard_logs/")
    # model.learn(total_timesteps=1000)
    #RL-Agenten initialisieren
    # model = PPO("MlpPolicy", vec_env, verbose=2)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     clip_range=0.1,
    #     ent_coef=0.2,
    #     verbose=1,
    #     learning_rate = 0.00001,
    #     tensorboard_log="./ppo_tensorboard_logs/"
    # )
    model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, verbose=1, tensorboard_log="./ppo_tensorboard_logs/")
    model.set_logger(new_logger)

    plot_callback = TrajectoryPlotCallback(vec_env, tcp_trajectory, plot_freq=1)

    # Training starten mit Callback
    reset_callback = ResetTrajectoryCallback(reset_freq=100000)
    model.learn(total_timesteps=5000000, callback=plot_callback)

    # Modell speichern
    model.save("ppo_trajectory_optimizer")
    print("Modell gespeichert.")
