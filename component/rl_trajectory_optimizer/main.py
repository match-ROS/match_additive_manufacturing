#! /usr/bin/env python3
import numpy as np
from stable_baselines3 import PPO
from trajectory_env import TrajectoryOptimizationEnv
import os, sys
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.path.abspath(__file__))


print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir)

from stable_baselines3.common.vec_env import SubprocVecEnv

# Funktion zum Erstellen einer neuen Umgebung
def make_env():
    return TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory)


from print_path import xMIR, yMIR, xTCP, yTCP



class TrajectoryPlotCallback(BaseCallback):
    def __init__(self, env, tcp_trajectory, plot_freq=10000, verbose=0):
        super(TrajectoryPlotCallback, self).__init__(verbose)
        self.env = env
        self.tcp_trajectory = tcp_trajectory
        self.plot_freq = plot_freq

    def _on_step(self) -> bool:
        # Plot after every `plot_freq` steps
        if self.n_calls % self.plot_freq == 0:
            obs = self.env.reset()
            trajectory = []
            done = np.array([False] * self.env.num_envs)  # Initialize 'done' array
            while not np.any(done):  # Continue until all environments are done
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                trajectory.append(obs[0])  # Append trajectory of the first environment
                
            # Selektiere die erste Umgebung und alle Zeitschritte
            trajectory = np.array(trajectory)
            single_env_trajectory = trajectory[:, 0, :]  # Alle Zeitschritte für Umgebung 0
            # Trajectory is recorded for the first environment only
            trajectory = np.array(trajectory)
            plt.plot(single_env_trajectory[:, 0], single_env_trajectory[:, 1], label="Optimierte Trajektorie", linestyle="-", linewidth=2)
            print(f"Trajectory shape: {trajectory.shape}")
            print(f"Trajectory: {trajectory}")
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

    # Anzahl der parallelen Umgebungen
    num_cpu = 24
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    # RL-Agenten initialisieren
    #model = PPO("MlpPolicy", vec_env, verbose=2)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_logs/"
    )

    # Callback erstellen
    plot_callback = TrajectoryPlotCallback(vec_env, tcp_trajectory, plot_freq=1)

    # Training starten mit Callback
    model.learn(total_timesteps=50000, callback=plot_callback)




    # Training starten
    # print("Starte Training...")
    # model.learn(total_timesteps=10000)
    # print("Training abgeschlossen!")

    # Modell speichern
    model.save("ppo_trajectory_optimizer")
    print("Modell gespeichert.")
