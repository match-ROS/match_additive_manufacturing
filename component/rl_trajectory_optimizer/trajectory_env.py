#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt



class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, max_displacement=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)
        self.scale_factor = 0.002
        self.max_distance = 2.0
        self.step_counter = 0
        self.rewards = []  # Liste für Rewards
        self.initial_trajectory = self.base_trajectory.copy()

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
            if i > 0 and i < len(self.current_trajectory) - 1:
                
                if i == 0:
                    direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]
                    direction_next /= np.linalg.norm(direction_next) + 1e-8
                    if a > 0:  # Bewegung in Richtung des nächsten Punktes
                        new_trajectory[i] += direction_next * a
                    continue

                # Spezialfall: Letzter Punkt (keine Bewegung zum nächsten)
                if i == len(self.current_trajectory) - 1:
                    direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
                    direction_prev /= np.linalg.norm(direction_prev) + 1e-8
                    if a < 0:  # Bewegung in Richtung des vorherigen Punktes
                        new_trajectory[i] += direction_prev * abs(a)
                    continue

                # if i > 1:
                #     print(f"Vorheriger Punkt {i-1}: {self.current_trajectory[i - 1]}")
                #     print(f"Aktueller Punkt {i}: {self.current_trajectory[i]}")
                #     print(f"Nächster Punkt {i+1}: {self.current_trajectory[i + 1]}")
                #     print(f"Aktion: {a}")
                #     print(f"Neue Position des Punktes {i}: {new_trajectory[i]}")


                # Berechne Richtung zum vorherigen Punkt und zum nächsten Punkt
                direction_prev = self.current_trajectory[i - 1] - self.current_trajectory[i]
                direction_next = self.current_trajectory[i + 1] - self.current_trajectory[i]

                # Normiere die Richtungen
                direction_prev /= np.linalg.norm(direction_prev) + 1e-8
                direction_next /= np.linalg.norm(direction_next) + 1e-8

                # Bewegung proportional zur Aktion
                if a > 0:  # Bewegung in Richtung des nächsten Punktes
                    new_trajectory[i] += direction_next * a * self.scale_factor
                elif a < 0:  # Bewegung in Richtung des vorherigen Punktes
                    new_trajectory[i] += direction_prev * abs(a) * self.scale_factor

        # Aktualisiere die Trajektorie
        self.current_trajectory = new_trajectory

        # Plot der Trajektorienänderung
        if self.step_counter % 10000 == 0:
            self.plot_trajectory_change(self.initial_trajectory, self.current_trajectory, self.step_counter)
        # Abstand zur TCP berechnen
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)

        # Punkte, die zu weit entfernt sind, korrigieren
        exceeds = distances > self.max_distance
        self.current_trajectory[exceeds] = (
            self.tcp_trajectory[exceeds]
            + (self.current_trajectory[exceeds] - self.tcp_trajectory[exceeds])
            * (self.max_distance / distances[exceeds])[:, np.newaxis]
        )


        # Calculate velocities and accelerations
        velocities, accelerations = self.calculate_velocities_and_accelerations(self.current_trajectory)


        # Belohnung berechnen (Beispiel)
        velocity_penalty = np.sum(np.linalg.norm(velocities, axis=1))
        acceleration_penalty = np.sum(np.linalg.norm(accelerations, axis=1))
        distance_penalty = np.sum(
            np.maximum(
                np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1) - self.max_distance,
                0,
            )
        )

        # Penalize exceeding velocity thresholds
        max_velocity = 0.3  # Example threshold
        velocity_threshold_penalty = np.sum(velocities[np.linalg.norm(velocities, axis=1) > max_velocity])

        reward = -0.1 * velocity_penalty - 0.5 * acceleration_penalty - 0.01 * distance_penalty

        # Penalize exceeding acceleration thresholds
        max_acceleration = 0.2  # Example threshold
        acceleration_threshold_penalty = np.sum(accelerations[np.linalg.norm(accelerations, axis=1) > max_acceleration])
        reward -= velocity_threshold_penalty + acceleration_threshold_penalty

        deviation = np.linalg.norm(self.current_trajectory - self.base_trajectory, axis=1)

        speed_changes = np.diff(self.current_trajectory, axis=0)
        smoothness_penalty = np.sum(np.linalg.norm(speed_changes, axis=1)**2)


        reward = -np.sum(deviation) - 0.1 * smoothness_penalty # Bestrafe Abweichungen von der Ausgangstrajektorie

        # Log or plot velocities and accelerations
        # print(f"Max Velocity: {np.max(np.linalg.norm(velocities, axis=1))}")
        # print(f"Max Acceleration: {np.max(np.linalg.norm(accelerations, axis=1))}")
        # print(f"Belohnung: {reward}")


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

    def calculate_and_plot_profiles(self,trajectory, reward, step_number):
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

        print(f"Schritt {step_number}: Reward = {reward}")

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