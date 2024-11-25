#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree



class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, time_step=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)
        self.scale_factor = 0.01
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
    
    def calculate_path_deviation(self,trajectory, reference_path):
        """
        Berechnet die Abweichung der Trajektorie vom ursprünglichen Pfad
        mithilfe eines KD-Baums für effiziente Entfernungssuche.
        """
        tree = cKDTree(reference_path)
        distances, _ = tree.query(trajectory)
        total_deviation = np.sum(distances)
        return total_deviation



 
    def step(self, action):

        # Plot der Trajektorienänderung
        if self.step_counter % 1000 == 0:
            self.plot_trajectory_change(self.initial_trajectory, self.current_trajectory, self.step_counter)
            self.calculate_and_plot_profiles(self.current_trajectory, step_number=self.step_counter)

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

        # Aktualisiere die Trajektorie
        self.current_trajectory = new_trajectory

        # Abstand zur TCP berechnen
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)

        velocities = np.linalg.norm(np.diff(self.current_trajectory, axis=0), axis=1) / self.time_step
        velocity_rmse = np.sqrt(np.mean((velocities) ** 2))

        # Abweichung zur initialen Trajektorie berechnen
        spatial_deviation = self.calculate_path_deviation(self.current_trajectory, self.initial_trajectory)
        print("spatial_deviation: ", spatial_deviation)

        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
        distance_penalty = np.sum(distances[distances > 0.95 * self.max_distance])

        reward = velocity_rmse
        reward -= distance_penalty 
        reward -= 0.1 * spatial_deviation 

        # if np.any(distances > self.max_distance):
        #     terminated = True
        #     reward -= 10  # Optional: eine hohe Strafe für die Verletzung der Bedingung
        # else:
        #     terminated = False

        # # Setze dies auf True, wenn eine zeitliche Begrenzung erreicht wurde
        # if self.step_counter >= 1000:
        #     truncated = True
        # else:
        #     truncated = False

        terminated = False
        truncated = False


        # Plot Geschwindigkeits- und Beschleunigungsprofile
        #self.calculate_and_plot_profiles(self.current_trajectory, reward, step_number=self.step_counter)
        self.step_counter += 1
        self.rewards.append(reward)

        obs = self.current_trajectory[:, 0].astype(np.float32)

        # Rückgabe der Werte
        #return self.current_trajectory.flatten().astype(np.float32), reward, terminated, truncated, {}
        return obs, reward, terminated, truncated, {}

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