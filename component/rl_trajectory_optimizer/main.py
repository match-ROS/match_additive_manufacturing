#! /usr/bin/env python3
import numpy as np
from stable_baselines3 import PPO
from trajectory_env import TrajectoryOptimizationEnv
import os, sys
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env


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
            print(f"Trajectory shape: {trajectory.shape}")
            single_env_trajectory = trajectory[:, 0, :]  # Alle Zeitschritte für Umgebung 0
            # Trajectory is recorded for the first environment only
            trajectory = np.array(trajectory)
            plt.plot(single_env_trajectory[:, 0], single_env_trajectory[:, 1], label="Optimierte Trajektorie", linestyle="-", linewidth=2)
            print(f"Trajectory shape: {trajectory.shape}")
            plt.figure(figsize=(8, 6))
            #plt.plot(trajectory[:, 0], trajectory[:, 1], label="Optimierte Trajektorie")
            plt.scatter(self.tcp_trajectory[:, 0], self.tcp_trajectory[:, 1], c='red', label="TCP Trajektorie")
            plt.scatter(self.env.base_trajectory[:, 0], self.env.base_trajectory[:, 1], c='green', label="Basis Trajektorie")
            plt.legend()
            plt.title(f"Trajektorienvergleich nach {self.n_calls} Schritten")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        return True
    
class ResetTrajectoryCallback(BaseCallback):
    def __init__(self, reset_freq, verbose=0):
        super(ResetTrajectoryCallback, self).__init__(verbose)
        self.reset_freq = reset_freq

    def _on_step(self) -> bool:
        # Überprüfe, ob die Umgebung zurückgesetzt werden soll
        if self.num_timesteps % self.reset_freq == 0:
            self.training_env.env_method("reset")
        return True


if __name__ == "__main__":
    # Lade die Trajektorien
    tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T
    base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T



    print("lenght of tcp_trajectory: ", len(tcp_trajectory))
    print("lenght of base_trajectory: ", len(base_trajectory))


    env = TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory)
    check_env(env, warn=True)

    # obs = env.reset()
    # action = env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action)

    # Anzahl der parallelen Umgebungen
    num_cpu = 1
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    # RL-Agenten initialisieren
    #model = PPO("MlpPolicy", vec_env, verbose=2)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_logs/"
    )

    # Callback erstellen
    plot_callback = TrajectoryPlotCallback(vec_env, tcp_trajectory, plot_freq=1)

    # Training starten mit Callback
    reset_callback = ResetTrajectoryCallback(reset_freq=1000)
    model.learn(total_timesteps=10000, callback=reset_callback)




    # Training starten
    # print("Starte Training...")
    # model.learn(total_timesteps=10000)
    # print("Training abgeschlossen!")

    # Modell speichern
    model.save("ppo_trajectory_optimizer")
    print("Modell gespeichert.")
