#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt



class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, time_step=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)
        self.scale_factor = 0.1
        self.max_distance = 2.0
        self.step_counter = 0
        self.rewards = []  # Liste für Rewards
        self.initial_trajectory = self.base_trajectory.copy()
        self.time_step = time_step
        self.time_stamps = np.linspace(0, len(self.base_trajectory) - 1, len(self.base_trajectory))
        self.initial_time_stamps = self.time_stamps.copy()
        self.total_duration = self.initial_time_stamps[-1] - self.initial_time_stamps[0]


        self.action_space = spaces.Box(
            low=-1 * self.scale_factor,
            high=1 * self.scale_factor,
            shape=(len(self.time_stamps),),  # Eine Aktion pro Punkt
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.time_stamps),),  # Eine Observation pro Punkt
            dtype=np.float32
        )

        print(f"Action space: {self.action_space}")
        print(f"Shape: {self.action_space.shape}")

    def reset(self, seed=None, options=None):
        """
        Reset die Umgebung und initialisiere den Zufallszahlengenerator.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        # Setze die Trajektorie zurück
        self.current_trajectory = self.base_trajectory.copy()
        
        obs = self.current_trajectory.flatten()[:len(self.base_trajectory)]  # Beispiel: Nur die ersten 964 Werte
        return obs.astype(np.float32), {}

    def calculate_velocities_and_accelerations(self,trajectory):
        # Velocity: First-order difference
        velocities = np.diff(trajectory, axis=0)

        # Acceleration: Second-order difference
        accelerations = np.diff(velocities, axis=0)

        return velocities, accelerations
    
    import matplotlib.pyplot as plt

 
    def step(self, action):
        """
        Passt die Zeitstempel basierend auf den Aktionen an.
        """

        # Plot der Trajektorienänderung
        if self.step_counter % 50000 == 0:
            self.plot_timestamps(self.time_stamps, self.step_counter)
            self.plot_timestamps_difference()
            self.plot_velocity_and_acceleration(self.base_trajectory, self.time_stamps, self.step_counter)


        # Wende die Aktion auf die Zeitstempel an
        new_time_stamps = self.time_stamps.copy()
        for i, a in enumerate(action):
                if i == 0:
                    # Erster Punkt: Nur Änderung der Zeit zum nächsten Punkt
                    new_time_stamps[i] = max(0, new_time_stamps[i] + a)
                elif i == len(self.time_stamps) - 1:
                    # Letzter Punkt: Nur Änderung der Zeit zum vorherigen Punkt
                    new_time_stamps[i] = max(new_time_stamps[i - 1] + 1e-3, new_time_stamps[i] + a)
                else:
                    # Mittlere Punkte: Zeitstempel anpassen
                    new_time_stamps[i] = max(new_time_stamps[i - 1] + 1e-3, new_time_stamps[i] + a)


        new_time_stamps = (
            new_time_stamps - new_time_stamps[0]
        ) * (self.total_duration / (new_time_stamps[-1] - new_time_stamps[0]))


        # Geschwindigkeiten berechnen
        velocities = np.linalg.norm(np.diff(self.base_trajectory, axis=0), axis=1) / np.diff(new_time_stamps)
        accelerations = np.diff(velocities) / np.diff(self.time_stamps[:-1])

        velocity_penalty = np.sum(np.maximum(velocities, 0) ** 2)
        acceleration_penalty = np.sum(np.maximum(accelerations, 0) ** 2)

        # Belohnungsfunktion: Gleichmäßige Geschwindigkeit anstreben
        target_velocity = 1.0  # Zielgeschwindigkeit
        velocity_rmse = np.sqrt(np.mean((velocities) ** 2))

        reward = -velocity_rmse #-velocity_rmse - 0.001 * velocity_penalty - 0.001 * acceleration_penalty

        # Aktualisiere die Zeitstempel
        self.time_stamps = new_time_stamps

        # Schrittzähler erhöhen
        self.step_counter += 1



        # Rückgabe der neuen Beobachtung
        obs = self.time_stamps.astype(np.float32)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}



    def render(self, mode="human"):
        """
        Visualisiert die aktuelle Trajektorie (optional).
        """
        print(f"Current trajectory: {self.current_trajectory}")

    def plot_velocity_and_acceleration(self,base_trajectory, time_stamps, step_counter):
        """
        Visualisiert die Geschwindigkeit und Beschleunigung basierend auf den geänderten Zeitstempeln.
        
        Args:
            base_trajectory (array): Die räumliche Trajektorie der Plattform.
            time_stamps (array): Die aktuellen Zeitstempel.
            step_counter (int): Der aktuelle Schrittzähler.
        """
        # Geschwindigkeiten berechnen
        velocities = np.linalg.norm(np.diff(base_trajectory, axis=0), axis=1) / np.diff(time_stamps)
        
        # Beschleunigungen berechnen
        accelerations = np.diff(velocities) / np.diff(time_stamps[:-1])
        
        # Geschwindigkeit plotten
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time_stamps[:-1], velocities, marker="o", label="Velocity (m/s)")
        plt.title(f"Velocity Profile at Step {step_counter}")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()
        plt.legend()
        
        # Beschleunigung plotten
        plt.subplot(2, 1, 2)
        plt.plot(time_stamps[:-2], accelerations, marker="o", label="Acceleration (m/s²)", color="orange")
        plt.title(f"Acceleration Profile at Step {step_counter}")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()





    def calculate_and_plot_profiles(self,trajectory, step_number):
        # Calculate velocities and accelerations
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Norms for velocities and accelerations
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

        # Plot profiles
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(velocity_magnitudes, label="Geschwindigkeit")
        plt.title(f"Schritt {step_number}: Geschwindigkeitsprofil")
        plt.xlabel("Trajektorie Punkt")
        plt.ylabel("Geschwindigkeit")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(acceleration_magnitudes, label="Beschleunigung", color="orange")
        plt.title(f"Schritt {step_number}: Beschleunigungsprofil")
        plt.xlabel("Trajektorie Punkt")
        plt.ylabel("Beschleunigung")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


    import matplotlib.pyplot as plt

    def plot_timestamps_difference(self):
        time_differences = np.diff(self.time_stamps)
        plt.figure()
        plt.plot(time_differences, marker="o", label="Time Differences")
        plt.axhline(np.mean(time_differences), color="red", linestyle="--", label="Mean Time Difference")
        plt.title("Time Differences Between Points")
        plt.xlabel("Point Index")
        plt.ylabel("Time Difference (s)")
        plt.legend()
        plt.grid()
        plt.show()





    def plot_timestamps(self,timestamps, step_counter):
        """
        Visualisiert die Zeitstempel.
        
        Args:
            timestamps (array): Die aktuellen Zeitstempel.
            step_counter (int): Der aktuelle Schrittzähler.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(timestamps - self.initial_time_stamps, marker="o", linestyle="-")
        plt.title(f"Timestamps Distribution at Step {step_counter}")
        plt.xlabel("Trajectory Point Index")
        plt.ylabel("Timestamp (s)")
        plt.grid()
        plt.show()


    def plot_trajectory_change(self,initial_trajectory, updated_trajectory, step_number):
        plt.figure(figsize=(10, 6))
        # Plot der initialen Trajektorie
        plt.plot(
            initial_trajectory[:, 0],
            initial_trajectory[:, 1],
            label="Initiale Trajektorie",
            linestyle="--",
            color="blue"
        )
        # Plot der aktualisierten Trajektorie
        plt.plot(
            updated_trajectory[:, 0],
            updated_trajectory[:, 1],
            label="Aktualisierte Trajektorie",
            linestyle="-",
            color="red"
        )
        plt.scatter(updated_trajectory[:, 0], updated_trajectory[:, 1], color="red", s=10)  # Punkte der neuen Trajektorie
        plt.scatter(initial_trajectory[:, 0], initial_trajectory[:, 1], color="blue", s=10)  # Punkte der neuen Trajektorie
        plt.title(f"Trajektorienänderung nach Schritt {step_number}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()