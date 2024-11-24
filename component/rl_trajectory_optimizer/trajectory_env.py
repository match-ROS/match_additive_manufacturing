#! /usr/bin/env python3
import numpy as np
import gym
#import gymnasium as gym
from gym import spaces

class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, max_distance=2.0):
        """
        RL-Umgebung für die Optimierung der Basistrajektorie eines mobilen Roboters.
        """
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)  # TCP-Zielpunkte (Manipulatortrajektorie)
        self.base_trajectory = np.array(base_trajectory)  # Basis-Trajektorie
        self.max_distance = max_distance  # Maximale Reichweite des Manipulators

        # Actions: -1 (Punkt nach hinten), 0 (Punkt bleibt), 1 (Punkt nach vorne)
        self.action_space = spaces.MultiDiscrete([3] * len(self.base_trajectory))
        # Observations: x und y Koordinaten der Basis
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.base_trajectory.shape, dtype=np.float32
        )

    def reset(self):
        """
        Reset die Umgebung.
        """
        self.current_trajectory = self.base_trajectory.copy()
        self.current_index = 0
        return self.current_trajectory

    def step(self, action):
        """
        Führt eine Aktion aus und berechnet die neue Trajektorie sowie die Belohnung.
        """
        # Verarbeite die Aktion (verschiebe Punkte entlang des Pfades)
        for i, a in enumerate(action):
            if a == 1 and i < len(self.current_trajectory) - 1:
                self.current_trajectory[i] += (
                    self.current_trajectory[i + 1] - self.current_trajectory[i]
                ) * 0.1
            elif a == -1 and i > 0:
                self.current_trajectory[i] += (
                    self.current_trajectory[i - 1] - self.current_trajectory[i]
                ) * 0.1

        # Geschwindigkeit und Beschleunigung berechnen
        velocities = np.diff(self.current_trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Bestrafungen:
        acceleration_penalty = np.sum(np.abs(accelerations))  # Hohe Beschleunigungen
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
        distance_penalty = np.sum(distances[distances > self.max_distance])  # Überschreiten der Reichweite

        # Belohnung: Gleichmäßige Geschwindigkeit und minimale Beschleunigung
        reward = -acceleration_penalty - distance_penalty

        # Ende der Episode
        done = self.current_index == len(self.current_trajectory) - 1

        self.current_index += 1

        return self.current_trajectory, reward, done, {}

    def render(self, mode="human"):
        """
        Visualisiert die aktuelle Trajektorie (optional).
        """
        print(f"Current trajectory: {self.current_trajectory}")

