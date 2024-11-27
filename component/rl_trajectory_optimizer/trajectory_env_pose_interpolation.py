#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline



class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, time_step=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)
        self.scale_factor = 0.000001
        self.max_distance = 2.0
        self.time_step = time_step
        self.step_counter = 0

        self.base_trajectory[:, 0] = gaussian_filter1d(self.base_trajectory[:, 0], sigma=1)
        self.base_trajectory[:, 1] = gaussian_filter1d(self.base_trajectory[:, 1], sigma=1)

        t_original = np.linspace(0, 1, len(self.base_trajectory))
        self.t_new = t_original
        self.cs_x = CubicSpline(t_original, self.base_trajectory[:, 0])
        self.cs_y = CubicSpline(t_original, self.base_trajectory[:, 1])
        self.splined_trajectory = np.vstack((self.cs_x(t_original), self.cs_y(t_original))).T
        self.current_trajectory = self.splined_trajectory.copy()


        # cs_x, cs_y = self.cubic_spline_interpolation(self.base_trajectory)
        # self.t_new = np.linspace(0, 1, len(self.base_trajectory))
        # self.splined_trajectory = np.vstack((cs_x(self.t_new), cs_y(self.t_new))).T

        # plot the new trajectory
        # plt.plot(new_trajectory[:, 0], new_trajectory[:, 1], 'r-', label='Modified Path')
        # plt.scatter(new_trajectory[:, 0], new_trajectory[:, 1], color='red')
        # plt.plot(self.base_trajectory[:, 0], self.base_trajectory[:, 1], 'b--', label='Initial Path')
        # plt.scatter(self.base_trajectory[:, 0], self.base_trajectory[:, 1], color='blue')
        # plt.legend()
        # plt.title('Trajektorienvergleich')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid()
        # plt.show()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len(self.base_trajectory),),  # Eine Aktion pro Punkt
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.base_trajectory), 2),  # 2D-Beobachtungen (x, y)
            dtype=np.float32
        )

    def cubic_spline_interpolation(self,base_trajectory):
        t = np.linspace(0, 1, len(base_trajectory))
        cs_x = CubicSpline(t, base_trajectory[:, 0])
        cs_y = CubicSpline(t, base_trajectory[:, 1])
        return cs_x, cs_y


    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Setze die aktuelle Trajektorie auf die gesplinte Trajektorie
        self.current_trajectory = self.splined_trajectory.copy()

        # Zähler zurücksetzen
        self.step_counter = 0
        obs = self.current_trajectory.astype(np.float32)
        return obs, {}

    
    def step(self, action):

        if self.step_counter % 50000 == 0:
            self.plot_current_trajectory(self.step_counter)
            self.calculate_and_plot_profiles(self.current_trajectory, self.step_counter)

        # Sampling-Punkte anpassen (0 bis 1 für die Länge des Pfades)
        t_original = np.linspace(0, 1, len(self.splined_trajectory))  # Originale Sampling-Punkte
        self.t_new = self.t_new + action * self.scale_factor  # Agent-Aktion beeinflusst Sampling

        # skaliere die Sampling-Punkte auf den Bereich [0, 1]
        self.t_new -= np.min(self.t_new)  # Verschiebe alle Werte so, dass der kleinste Wert 0 ist
        self.t_new /= np.max(self.t_new)  # Skaliere alle Werte so, dass der größte Wert 1 ist
        #print(t_original)

        new_trajectory = np.vstack((self.cs_x(self.t_new), self.cs_y(self.t_new))).T

        # Berechnung von Abständen, Geschwindigkeiten usw.
        distances = np.linalg.norm(new_trajectory - self.tcp_trajectory, axis=1)
        velocities = np.linalg.norm(np.diff(new_trajectory, axis=0), axis=1) / self.time_step
        velocity_rmse = np.sqrt(np.mean(velocities ** 2))

        # Reward berechnen
        reward = -velocity_rmse  # Belohnung für gleichmäßige Geschwindigkeiten
        if np.any(distances > self.max_distance):
            reward -= 10  # Strafe für Verletzung der Reichweite

        # `terminated` und `truncated` als echte Booleans sicherstellen
        terminated = False#bool(np.any(distances > self.max_distance))
        truncated = False#bool(self.step_counter >= 1000)

        # Trajektorie aktualisieren
        self.current_trajectory = new_trajectory
        self.step_counter += 1

        obs = self.current_trajectory.astype(np.float32)
        return obs, reward, terminated, truncated, {}
    
    def calculate_path_deviation(self, trajectory):
        """
        Berechnet die Abweichung der aktuellen Trajektorie von der gesplinten Trajektorie.
        """
        deviations = np.linalg.norm(trajectory - self.splined_trajectory, axis=1)
        total_deviation = np.sum(deviations)
        return total_deviation


    def plot_current_trajectory(self, step_counter):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.splined_trajectory[:, 0], self.splined_trajectory[:, 1], 'b--', label='Splined Trajectory')
        plt.plot(self.current_trajectory[:, 0], self.current_trajectory[:, 1], 'r-', label='Modified Trajectory')
        plt.scatter(self.current_trajectory[:, 0], self.current_trajectory[:, 1], color='red', s=10)
        plt.title('Trajectory Comparison at Step {}'.format(step_counter))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.legend()
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
    