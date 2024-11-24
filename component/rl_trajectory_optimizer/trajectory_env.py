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
        self.max_distance = 2.0
        # Action space: Flach (1928,)
        # self.action_space = spaces.Box(
        #     low=-max_displacement,
        #     high=max_displacement,
        #     shape=(964 * 2,),  # Flach
        #     dtype=np.float32
        # )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(964 * 2,),  # Flache Struktur
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(964 * 2,),  # Abgeflachte Struktur
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
        
        # Rückgabe als float32
        return self.current_trajectory.flatten().astype(np.float32), {}



    def step(self, action):
        # Skaliere und reshaped die Aktion
        action = action.reshape((964, 2)) * 0.1

        # Wende die Aktion an
        self.current_trajectory += action

        # Belohnung berechnen (Beispiel)
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
        distance_penalty = np.sum((distances[distances > self.max_distance] - self.max_distance) ** 2)
        reward = -distance_penalty

        # Status: Episode beendet?
        terminated = False  # Setze deine Bedingung hier, z. B. ob das Ziel erreicht wurde
        truncated = False   # Setze dies auf True, wenn eine zeitliche Begrenzung erreicht wurde

        # Rückgabe der Werte
        return self.current_trajectory.flatten().astype(np.float32), reward, terminated, truncated, {}





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

