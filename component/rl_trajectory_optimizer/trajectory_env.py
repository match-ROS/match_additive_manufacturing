#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces



class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, max_displacement=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)

        # Action space: Flach (1928,)
        self.action_space = spaces.Box(
            low=-max_displacement,
            high=max_displacement,
            shape=(964 * 2,),  # Flach
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.base_trajectory.shape,
            dtype=np.float32
        )

        print(f"Action space: {self.action_space}")
        print(f"Shape: {self.action_space.shape}")

    def reset(self, seed=None, options=None):
        """
        Reset die Umgebung und initialisiere den Zufallszahlengenerator (falls erforderlich).
        """
        if seed is not None:
            # Initialisiere den Zufallszahlengenerator
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_trajectory = self.base_trajectory.copy()  # Zurücksetzen der Trajektorie
        return self.current_trajectory  # Rückgabe der initialen Observation


    def step(self, action):
        # Reshape der Aktion in die ursprüngliche Struktur (964, 2)
        action = action.reshape((964, 2))

        # Wende die Aktion an (z. B. Verschiebung der Punkte)
        self.current_trajectory += action

        # Berechnung der Belohnung
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
        distance_penalty = np.sum((distances[distances > self.max_distance] - self.max_distance) ** 2)
        reward = -distance_penalty

        done = False  # Episode bleibt offen, bis Bedingung erfüllt ist
        return self.current_trajectory, reward, done, {}



    # def step(self, action):
    #     """
    #     Führt eine Aktion aus und berechnet die neue Trajektorie sowie die Belohnung.
    #     """
    #     # Verarbeite die Aktion (verschiebe Punkte entlang des Pfades)
    #     for i, a in enumerate(action):
    #         if a == 1 and i < len(self.current_trajectory) - 1:
    #             self.current_trajectory[i] += (
    #                 self.current_trajectory[i + 1] - self.current_trajectory[i]
    #             ) * 0.1
    #         elif a == -1 and i > 0:
    #             self.current_trajectory[i] += (
    #                 self.current_trajectory[i - 1] - self.current_trajectory[i]
    #             ) * 0.1

    #     # Geschwindigkeit und Beschleunigung berechnen
    #     velocities = np.diff(self.current_trajectory, axis=0)
    #     accelerations = np.diff(velocities, axis=0)

    #     # Bestrafungen:
    #     acceleration_penalty = np.sum(np.abs(accelerations))  # Hohe Beschleunigungen
    #     distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
    #     distance_penalty = np.sum(distances[distances > self.max_distance])  # Überschreiten der Reichweite

    #     # Belohnung: Gleichmäßige Geschwindigkeit und minimale Beschleunigung
    #     reward = -acceleration_penalty - distance_penalty

    #     # Ende der Episode
    #     done = self.current_index == len(self.current_trajectory) - 1

    #     self.current_index += 1

    #     return self.current_trajectory, reward, done, {}

    def render(self, mode="human"):
        """
        Visualisiert die aktuelle Trajektorie (optional).
        """
        print(f"Current trajectory: {self.current_trajectory}")

