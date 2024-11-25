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

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len(self.base_trajectory),),  # Eine Aktion pro Punkt
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.base_trajectory),),  # Eine Observation pro Punkt
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
        # Skaliere und reshaped die Aktion
        #action = action.reshape((964, 2)) * 0.005

        new_trajectory = self.current_trajectory.copy()

        for i, a in enumerate(action):
            if i == 0:
                # Erster Punkt: Nur Bewegung in Richtung des nächsten Punktes möglich
                direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]
                new_trajectory[i] += direction_next * a * self.scale_factor
                continue

            if i == len(self.current_trajectory) - 1:
                # Letzter Punkt: Nur Bewegung in Richtung des vorherigen Punktes möglich
                direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
                new_trajectory[i] += direction_prev * abs(a) * self.scale_factor
                continue

            # Mittlere Punkte: Bewegung in Richtung des vorherigen oder nächsten Punktes
            if a > 0:
                # Bewegung in Richtung des nächsten Punktes
                direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]
                new_trajectory[i] += direction_next * a * self.scale_factor
            elif a < 0:
                # Bewegung in Richtung des vorherigen Punktes
                direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
                new_trajectory[i] += direction_prev * abs(a) * self.scale_factor

        # for i, a in enumerate(action):
        #     if i > 0 and i < len(self.current_trajectory) - 1:
                
        #         if i == 0:
        #             direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]
        #             direction_next /= np.linalg.norm(direction_next) + 1e-8
        #             if a > 0:  # Bewegung in Richtung des nächsten Punktes
        #                 new_trajectory[i] += direction_next * a
        #             continue

        #         # Spezialfall: Letzter Punkt (keine Bewegung zum nächsten)
        #         if i == len(self.current_trajectory) - 1:
        #             direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
        #             direction_prev /= np.linalg.norm(direction_prev) + 1e-8
        #             if a < 0:  # Bewegung in Richtung des vorherigen Punktes
        #                 new_trajectory[i] += direction_prev * abs(a)
        #             continue

        #         # Berechne Richtung zum vorherigen Punkt und zum nächsten Punkt
        #         direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
        #         direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]

        #         # Normiere die Richtungen
        #         direction_prev /= np.linalg.norm(direction_prev) + 1e-8
        #         direction_next /= np.linalg.norm(direction_next) + 1e-8

        #         # if both points are the same, set direction to 0
        #         if self.current_trajectory[i - 1][0] == self.current_trajectory[i + 1][0] and self.current_trajectory[i - 1][1] == self.current_trajectory[i + 1][1]:
        #             direction_prev = 0
        #             direction_next = 0

        #         # Bewegung proportional zur Aktion
        #         if a > 0:  # Bewegung in Richtung des nächsten Punktes
        #             new_trajectory[i] += direction_next * a * self.scale_factor
        #         elif a < 0:  # Bewegung in Richtung des vorherigen Punktes
        #             new_trajectory[i] += direction_prev * abs(a) * self.scale_factor

        # Aktualisiere die Trajektorie
        self.current_trajectory = new_trajectory

        # Plot der Trajektorienänderung
        if self.step_counter % 10000 == 0:
            self.plot_trajectory_change(self.initial_trajectory, self.current_trajectory, self.step_counter)
            self.calculate_and_plot_profiles(self.current_trajectory, step_number=self.step_counter)
        # Abstand zur TCP berechnen
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)

        # Punkte, die zu weit entfernt sind, korrigieren
        # exceeds = distances > self.max_distance
        # self.current_trajectory[exceeds] = (
        #     self.tcp_trajectory[exceeds]
        #     + (self.current_trajectory[exceeds] - self.tcp_trajectory[exceeds])
        #     * (self.max_distance / distances[exceeds])[:, np.newaxis]
        # )


        velocities = np.linalg.norm(np.diff(self.current_trajectory, axis=0), axis=1) / self.time_step
        velocity_rmse = np.sqrt(np.mean((velocities) ** 2))


        # speed_changes = np.diff(self.current_trajectory, axis=0)
        # smoothness_penalty = np.sum(np.linalg.norm(speed_changes, axis=1)**2)


        # reward = -np.sum(deviation) - 0.1 * smoothness_penalty # Bestrafe Abweichungen von der Ausgangstrajektorie

        # 1. Strafe für Abweichungen von der TCP-Trajektorie
        distances = np.linalg.norm(self.current_trajectory - self.base_trajectory, axis=1)
        distance_penalty = np.sum(distances)

        # 2. Bestrafe ungleichmäßige Geschwindigkeiten (scharfe Änderungen)
        speed_changes = np.diff(self.current_trajectory, axis=0)
        smoothness_penalty = np.sum(np.linalg.norm(speed_changes, axis=1)**2)

        # 3. Bestrafe hohe Beschleunigungen
        accelerations = np.diff(speed_changes, axis=0)
        acceleration_penalty = np.sum(np.linalg.norm(accelerations, axis=1)**2)

        print(f"Distance penalty: {distance_penalty}")
        print(f"Smoothness penalty: {smoothness_penalty}")
        print(f"Acceleration penalty: {acceleration_penalty}")
        print(f"Velocity RMSE: {velocity_rmse}")

        # Gesamt-Reward
        reward = -0.01 * distance_penalty - 0.01 * smoothness_penalty - 0.05 * acceleration_penalty - 0.5 * velocity_rmse



        # Status: Episode beendet?
        terminated = False  # Setze deine Bedingung hier, z. B. ob das Ziel erreicht wurde
        truncated = False   # Setze dies auf True, wenn eine zeitliche Begrenzung erreicht wurde

        # Plot Geschwindigkeits- und Beschleunigungsprofile
        #self.calculate_and_plot_profiles(self.current_trajectory, reward, step_number=self.step_counter)
        self.step_counter += 1
        self.rewards.append(reward)

        obs = self.current_trajectory[:, 0].astype(np.float32)

        # Rückgabe der Werte
        #return self.current_trajectory.flatten().astype(np.float32), reward, terminated, truncated, {}
        return obs, reward, terminated, truncated, {}


    def render(self, mode="human"):
        """
        Visualisiert die aktuelle Trajektorie (optional).
        """
        print(f"Current trajectory: {self.current_trajectory}")

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
        plt.title(f"Trajektorienänderung nach Schritt {step_number}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()