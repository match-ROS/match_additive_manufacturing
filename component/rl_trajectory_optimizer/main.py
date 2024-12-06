#! /usr/bin/env python3
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
#from trajectory_env import TrajectoryOptimizationEnv
from trajectory_env_pose_interpolation import TrajectoryOptimizationEnv
import os, sys
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import tensorflow as tf
import optuna
from torch.optim import Adam
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


def calculate_velocity_and_acceleration(trajectory, time_step):
    velocities = np.diff(trajectory, axis=0) / time_step
    accelerations = np.diff(velocities, axis=0) / time_step
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    return velocity_magnitudes, acceleration_magnitudes
    
def calculate_distances(base_trajectory, tcp_trajectory):
    return np.linalg.norm(base_trajectory - tcp_trajectory, axis=1)

class ProfilePlotCallback(BaseCallback):
    def __init__(self, env, tcp_trajectory, plot_freq=1000, verbose=0):
        super(ProfilePlotCallback, self).__init__(verbose)
        self.env = env
        self.tcp_trajectory = tcp_trajectory
        self.plot_freq = plot_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            # Hole die aktuelle optimierte Trajektorie
            base_trajectory = self.env.env_method("get_current_trajectory")[0]
            
            # Berechne Profile
            time_step = self.env.env_method("get_time_step")[0]
            velocities, accelerations = calculate_velocity_and_acceleration(base_trajectory, time_step)
            distances = calculate_distances(base_trajectory, self.tcp_trajectory)

            # Plot Geschwindigkeits- und Beschleunigungsprofil
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(velocities, label="Geschwindigkeit")
            plt.title("Geschwindigkeitsprofil")
            plt.xlabel("Trajektorie-Punkt")
            plt.ylabel("Geschwindigkeit")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(accelerations, label="Beschleunigung", color="orange")
            plt.title("Beschleunigungsprofil")
            plt.xlabel("Trajektorie-Punkt")
            plt.ylabel("Beschleunigung")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            # Plot Abstand
            plt.figure(figsize=(8, 6))
            plt.plot(distances, label="Abstand Basistrajektorie-TCP")
            plt.title("Abstand zwischen optimierter Basistrajektorie und TCP-Trajektorie")
            plt.xlabel("Trajektorie-Punkt")
            plt.ylabel("Abstand")
            plt.legend()
            plt.grid(True)
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
    
def linear_schedule(initial_value):
    """
    Returns a function that computes a linearly decreasing learning rate.
    The learning rate starts at `initial_value` and decreases linearly with the progress.
    """
    def schedule(progress_remaining):
        # Progress remaining: 1.0 -> 0.0
        return progress_remaining * initial_value

    return schedule
    

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir="./ppo_tensorboard_logs/"):
        super(RewardLoggingCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def _on_step(self) -> bool:
        # Logge den Reward, wenn eine Episode endet
        if "episode" in self.locals["infos"][0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            step = self.num_timesteps
            with self.writer.as_default():
                tf.summary.scalar("rollout/ep_rew_mean", episode_reward, step=step)
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()

class RewardBasedLRScheduler:
    def __init__(self, initial_lr=3e-4, patience=10, factor=0.5, min_lr=1e-6, verbose=0):
        """
        Dynamischer Scheduler für die Lernrate basierend auf der Reward-Entwicklung.
        Args:
        - initial_lr: Anfangswert der Lernrate
        - patience: Anzahl der Schritte ohne Verbesserung, bevor die Lernrate angepasst wird
        - factor: Multiplikationsfaktor, um die Lernrate zu verringern (z. B. 0.5 halbiert die Lernrate)
        - min_lr: Minimale Lernrate
        - verbose: Ausgabe von Debugging-Informationen
        """
        self.lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_reward = -np.inf
        self.counter = 0
        self.verbose = verbose

    def step(self, current_reward):
        """
        Prüft, ob die Lernrate angepasst werden muss.
        Args:
        - current_reward: Der aktuelle Reward des Agenten
        Returns:
        - Neue Lernrate
        """
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            new_lr = max(self.min_lr, self.lr * self.factor)
            if self.verbose and new_lr != self.lr:
                print(f"[LRScheduler] Reducing learning rate from {self.lr:.6f} to {new_lr:.6f}")
            self.lr = new_lr
            self.counter = 0

        return self.lr

class RewardBasedLRCallback(BaseCallback):
    def __init__(self, scheduler, verbose=0):
        super(RewardBasedLRCallback, self).__init__(verbose)
        self.scheduler = scheduler
        self.last_reward = -np.inf

    def _on_step(self) -> bool:
        # Hole den letzten episodischen Reward (falls verfügbar)
        reward_logs = self.locals.get("infos", [{}])
        current_reward = reward_logs[-1].get("episode", {}).get("r", None)

        if current_reward is not None:
            # Update der Lernrate basierend auf dem aktuellen Reward
            new_lr = self.scheduler.step(current_reward)
            self.model.lr_schedule = lambda _: new_lr
            if self.verbose:
                print(f"[RewardBasedLRCallback] Updated learning rate to {new_lr:.6f}")

        return True
    
class AdaptiveLRScheduler:
    def __init__(self, initial_lr=3e-4, patience=5, factor_increase=1.5, factor_decrease=0.5, min_lr=1e-6, max_lr=1e-2, verbose=0):
        """
        Adaptiver Scheduler für die Lernrate, der sowohl erhöhen als auch verringern kann.
        Args:
        - initial_lr: Anfangslernrate
        - patience: Anzahl der Schritte ohne Verbesserung, bevor die Lernrate angepasst wird
        - factor_increase: Multiplikator zur Erhöhung der Lernrate (z. B. 1.5)
        - factor_decrease: Multiplikator zur Verringerung der Lernrate (z. B. 0.5)
        - min_lr: Minimale Lernrate
        - max_lr: Maximale Lernrate
        - verbose: Ausgabe von Debugging-Informationen
        """
        self.lr = initial_lr
        self.patience = patience
        self.factor_increase = factor_increase
        self.factor_decrease = factor_decrease
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_reward = -np.inf
        self.counter = 0
        self.verbose = verbose

    def step(self, current_reward):
        """
        Passt die Lernrate dynamisch an.
        Args:
        - current_reward: Der aktuelle Reward des Agenten
        Returns:
        - Neue Lernrate
        """
        if current_reward > self.best_reward:
            # Verbesserung: Aktualisiere besten Reward und setze Zähler zurück
            self.best_reward = current_reward
            self.counter = 0
        else:
            # Keine Verbesserung: Erhöhe den Zähler
            self.counter += 1

        # Anpassung der Lernrate bei zu vielen nicht-verbesserten Schritten
        if self.counter >= self.patience:
            if current_reward < self.best_reward * 0.9:  # Große Verschlechterung: Lernrate verringern
                new_lr = max(self.min_lr, self.lr * self.factor_decrease)
            else:  # Stagnation: Lernrate erhöhen
                new_lr = min(self.max_lr, self.lr * self.factor_increase)

            if self.verbose:
                print(f"[AdaptiveLRScheduler] Adjusting learning rate from {self.lr:.6f} to {new_lr:.6f}")
            self.lr = new_lr
            self.counter = 0

        return self.lr

class AdaptiveLRCallback(BaseCallback):
    def __init__(self, scheduler, verbose=0):
        super(AdaptiveLRCallback, self).__init__(verbose)
        self.scheduler = scheduler

    def _on_step(self) -> bool:
        # Hole den letzten episodischen Reward (falls verfügbar)
        reward_logs = self.locals.get("infos", [{}])
        current_reward = reward_logs[-1].get("episode", {}).get("r", None)

        if current_reward is not None:
            # Update der Lernrate basierend auf dem aktuellen Reward
            new_lr = self.scheduler.step(current_reward)
            self.model.lr_schedule = lambda _: new_lr
            # if self.verbose:
            #     #print(f"[AdaptiveLRCallback] Updated learning rate to {new_lr:.6f}")
            #     

        return True


class TrajectoryPlotCallback(BaseCallback):
    def __init__(self, env, tcp_trajectory, plot_freq=1000, verbose=0):
        super(TrajectoryPlotCallback, self).__init__(verbose)
        self.env = env
        self.tcp_trajectory = tcp_trajectory
        self.plot_freq = plot_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            # Rufe die aktuelle Trajektorie aus der ersten Umgebung ab
            trajectories = self.env.env_method("get_current_trajectory")
            trajectory = trajectories[0]  # Zugriff auf die erste Umgebung

            # Plot
            plt.figure(figsize=(8, 6))
            plt.plot(trajectory[:, 0], trajectory[:, 1], label="Optimierte Trajektorie", linewidth=2)
            plt.scatter(self.tcp_trajectory[:, 0], self.tcp_trajectory[:, 1], c='red', label="TCP Trajektorie", s=10)
            plt.legend()
            plt.title(f"Trajektorienvergleich nach {self.n_calls} Schritten")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.grid(True)
            plt.show()

        return True

if __name__ == "__main__":
    # Lade die Trajektorien
    tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T
    base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T

    log_dir = "./ppo_tensorboard_logs/"  # Einheitliches Verzeichnis
    new_logger = configure(log_dir, ["stdout", "tensorboard"])



    # env = TrajectoryOptimizationEnv(tcp_trajectory, base_trajectory,time_step=0.1)
    # check_env(env, warn=True)

    # Anzahl der parallelen Umgebungen
    num_cpu = 32
    vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    #vec_env.max_steps_per_episode = 1000 # Kürzere Episoden für schnelleres Training

    def lr_schedule(progress_remaining):
        return 3e-2 * progress_remaining 
    
     
    # scheduler = RewardBasedLRScheduler(
    #     initial_lr=3e-4,
    #     patience=50,  # 5 Episoden ohne Verbesserung, bevor die Lernrate reduziert wird
    #     factor=0.5,  # Lernrate wird halbiert
    #     min_lr=1e-6,  # Mindestwert für die Lernrate
    #     verbose=1
    # )

    scheduler = AdaptiveLRScheduler(
            initial_lr=3e-4,
            patience=5,
            factor_increase=1.5,
            factor_decrease=0.5,
            min_lr=1e-6,
            max_lr=1e-3,
            verbose=0
        )

    #model = SAC("MlpPolicy", vec_env, ent_coef='auto',   verbose=0, tensorboard_log="./ppo_tensorboard_logs/")
    model = PPO("MlpPolicy", vec_env, learning_rate=lr_schedule,   verbose=0, tensorboard_log="./ppo_tensorboard_logs/")
    model.set_logger(new_logger)

    # Callbacks initialisieren
    # lr_callback = RewardBasedLRCallback(scheduler, verbose=1)
    lr_callback = AdaptiveLRCallback(scheduler, verbose=0)
    plot_callback = TrajectoryPlotCallback(vec_env, tcp_trajectory, plot_freq=10000)
    profile_plot_callback = ProfilePlotCallback(vec_env, tcp_trajectory, plot_freq=10000)

    # Training starten mit Callback
    reset_callback = ResetTrajectoryCallback(reset_freq=100000)
    reward_callback = RewardLoggingCallback(log_dir="./ppo_tensorboard_logs/")
    model.learn(total_timesteps=1000000) # , callback=lr_callback)

    # Ergebnisse evaluieren
    obs = vec_env.reset()
    action, _ = model.predict(obs)
    obs, reward, terminated, _ = vec_env.step(action)
    print(f"Final Reward: {reward}")
    
    # Modell speichern
    model.save("sac_adaptive_lr_optimizer")
    print("Modell gespeichert.")
