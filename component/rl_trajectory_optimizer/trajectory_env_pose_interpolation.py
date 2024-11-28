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
        self.scale_factor = 0.029031316514772237 * 0.05
        self.max_distance = 2.0
        self.time_step = time_step
        self.step_counter = 0
        self.previous_velocity_rmse = 0
        self.total_step_counter = 0
        self.reward_history = []
        self.cumulative_reward = 0.0


        self.base_trajectory[:, 0] = gaussian_filter1d(self.base_trajectory[:, 0], sigma=1)
        self.base_trajectory[:, 1] = gaussian_filter1d(self.base_trajectory[:, 1], sigma=1)

        t_original = np.linspace(0, 1, len(self.base_trajectory))
        self.t_new = t_original
        self.cs_x = CubicSpline(t_original, self.base_trajectory[:, 0])
        self.cs_y = CubicSpline(t_original, self.base_trajectory[:, 1])
        self.tcp_cs_x = CubicSpline(t_original, self.tcp_trajectory[:, 0])
        self.tcp_cs_y = CubicSpline(t_original, self.tcp_trajectory[:, 1])
        self.splined_trajectory = np.vstack((self.cs_x(t_original), self.cs_y(t_original))).T
        self.splined_tcp_trajectory = np.vstack((self.tcp_cs_x(t_original), self.tcp_cs_y(t_original))).T
        self.current_trajectory = self.splined_trajectory.copy()

        avg_distance = np.mean(np.linalg.norm(np.diff(base_trajectory, axis=0), axis=1))
        print(f"Average distance between trajectory points: {avg_distance}")



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
            low=0.01,
            high=1.0,
            shape=(len(self.base_trajectory)-1,),  # Eine Aktion pro Punkt
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_trajectory.size,),  # 2D-Beobachtungen (x, y)
            dtype=np.float32
        )

    def cubic_spline_interpolation(self,base_trajectory):
        t = np.linspace(0, 1, len(base_trajectory))
        cs_x = CubicSpline(t, base_trajectory[:, 0])
        cs_y = CubicSpline(t, base_trajectory[:, 1])
        return cs_x, cs_y

    def get_time_step(self):
        return self.time_step

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Setze die aktuelle Trajektorie auf die gesplinte Trajektorie
        self.current_trajectory = self.splined_trajectory.copy()
        self.reward_history = []

        # Zähler zurücksetzen
        self.step_counter = 0
        self.cumulative_reward = 0.0
        self.t_new = np.linspace(0, 1, len(self.base_trajectory))
        obs = self.current_trajectory.astype(np.float32)
        obs = obs.flatten()  # Flache 1D-Struktur erzeugen
        return obs, {}

    def monitor_reward(self, reward):
        # Vergleiche mit dem vorherigen Reward
        avg_reward = np.mean(self.reward_history[-10:])  # Durchschnitt der letzten 10 Rewards
        if len(self.reward_history) > 9 and reward < avg_reward:
            self.worsening_count += 1
        else:
            self.worsening_count = 0  # Zurücksetzen bei Verbesserung

        # Speichere den aktuellen Reward
        self.reward_history.append(reward)

        # Beende die Episode, wenn die Verschlechterung mehrfach hintereinander auftritt
        if self.worsening_count >= 5:  # Beispiel: nach 5 Verschlechterungen
            terminated = True
            #print(f"Episode beendet: Wiederholte Verschlechterungen (Count: {self.worsening_count})")
        else:
            terminated = False
        return terminated


    def step(self, action):
        
        time_intervals = action * self.scale_factor  # Aktion gibt relative Zeitintervalle
        self.t_new = np.concatenate(([0], np.cumsum(time_intervals)))
        self.t_new -= np.min(self.t_new)
        self.t_new /= np.max(self.t_new)

        # Interpolieren mit der synchronisierten Punktanzahl
        new_trajectory = np.vstack((self.cs_x(self.t_new), self.cs_y(self.t_new))).T

        # Berechnung der Distanzen zwischen Punkten
        distances = np.linalg.norm(np.diff(new_trajectory, axis=0), axis=1)

        # Berechnung der Geschwindigkeiten zwischen Punkten
        t_original = np.linspace(0, 1, len(self.base_trajectory))
        time_intervals = np.diff(t_original)
        velocities = distances / time_intervals

        # Berechnung der Differenzen der Geschwindigkeiten
        velocity_differences = np.diff(velocities)

        # Berechnung der Beschleunigungen
        accelerations = velocity_differences / time_intervals[1:]

        # Berechnung der Geschwindigkeitsänderungen
        velocity_penalty = np.std(velocities)

        # Berechne maximale Geschwindigkeit
        max_velocity_penalty = np.max(velocities)

        # Berechnung der Beschleunigungsänderungen
        acceleration_penalty = np.std(accelerations)

        # Berechne maximale Beschleunigung
        max_acceleration_penalty = np.max(accelerations)

        # Berechne Acc RMSE 
        acc_rmse_penalty = np.sqrt(np.mean(accelerations ** 2))

        # Berechne Velocity RMSE
        velocity_rmse_penalty = np.sqrt(np.mean(velocities ** 2))

        # Interpolation der TCP-Trajektorie auf die gleichen Zeitstempel
        tcp_interpolated = np.vstack((self.tcp_cs_x(self.t_new), self.tcp_cs_y(self.t_new))).T

        # Berechnung der euklidischen Abstände zwischen den Trajektorien
        tcp_distances = np.linalg.norm(new_trajectory - tcp_interpolated, axis=1)

        # Bestrafung für Punkte, die weiter als die maximale Distanz entfernt sind
        distance_penalty = np.sum(np.maximum(0, tcp_distances - self.max_distance))

        uniformity_penalty = np.std(time_intervals)



        reward = (
            10.0
            - distance_penalty * 10.0
            - uniformity_penalty * 2.0
            - velocity_penalty * 1.1
            - acceleration_penalty * 0.0
            - max_velocity_penalty * 0.1
            - max_acceleration_penalty * 0.001
            - acc_rmse_penalty * 0.001
            - velocity_rmse_penalty * 0.1
        )

        # Kumulative Belohnung
        self.cumulative_reward += reward

        # Prüfung auf Terminierung
        terminated = self.monitor_reward(reward)  # Optional: Bedingung hinzufügen, wenn nötig
        truncated = self.step_counter >= 500  # Episodenlänge begrenzen
        if terminated or truncated:
            info = {"episode": {"r": self.cumulative_reward}}
            self.cumulative_reward = 0.0  # Zurücksetzen für die nächste Episode
        else:
            info = {}

        # Aktualisierung der Trajektorie
        self.current_trajectory = new_trajectory
        self.step_counter += 1
        self.total_step_counter += 1

        # Visualisierung nach bestimmten Schritten (optional)
        if self.total_step_counter % 20000 == 0:
            print(f"Step {self.total_step_counter}: Reward={reward:.2f}")
            print(f"Distance Penalty: {distance_penalty:.2f}")
            print(f"Uniformity Penalty: {uniformity_penalty:.2f}")
            print(f"Velocity Penalty: {velocity_penalty:.2f}")
            print(f"Acceleration Penalty: {acceleration_penalty:.2f}")
            print(f"Max Velocity Penalty: {max_velocity_penalty:.2f}")
            print(f"Max Acceleration Penalty: {max_acceleration_penalty:.2f}")
            print(f"Acc RMSE Penalty: {acc_rmse_penalty:.2f}")
            print(f"Velocity RMSE Penalty: {velocity_rmse_penalty:.2f}")
            #self.plot_current_trajectory(self.step_counter)
            self.calculate_and_plot_profiles(velocities, accelerations, self.step_counter)

        # Beobachtung zurückgeben
        obs = self.current_trajectory.astype(np.float32)
        obs = obs.flatten()  # Flache 1D-Struktur erzeugen
        return obs, reward, terminated, truncated, {}

    def get_current_trajectory(self):
        return self.current_trajectory.copy()
    
    def calculate_path_deviation(self, trajectory):
        """
        Berechnet die Abweichung der aktuellen Trajektorie von der gesplinten Trajektorie.
        """
        deviations = np.linalg.norm(trajectory - self.splined_trajectory, axis=1)
        total_deviation = np.sum(deviations)
        return total_deviation
    
    def calculate_spacing_penalty(self):
        distances_between_points = np.linalg.norm(np.diff(self.current_trajectory, axis=0), axis=1)
        penalty = np.sqrt(np.mean(distances_between_points ** 2))
        return penalty

    def calculate_uniformity_reward(self):
        distances_between_points = np.linalg.norm(np.diff(self.current_trajectory, axis=0), axis=1)
        # Standardabweichung der Abstände
        std_deviation = np.std(distances_between_points)
        max_distance = np.max(distances_between_points)
        return -std_deviation,  max_distance  # Negativer Wert, da größere Abweichungen schlechter sind



    def plot_current_trajectory(self, step_counter):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.splined_trajectory[:, 0], self.splined_trajectory[:, 1], 'b--', label='Splined Trajectory')
        plt.plot(self.current_trajectory[:, 0], self.current_trajectory[:, 1], 'r-', label='Modified Trajectory')
        plt.scatter(self.splined_tcp_trajectory[:, 0], self.splined_tcp_trajectory[:, 1], color='blue', s=10, label='Splined TCP Trajectory')
        plt.scatter(self.tcp_trajectory[:, 0], self.tcp_trajectory[:, 1], color='blue', s=10, label='TCP Trajectory')
        plt.scatter(self.current_trajectory[:, 0], self.current_trajectory[:, 1], color='red', s=10)
        plt.title('Trajectory Comparison at Step {}'.format(step_counter))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.legend()
        plt.show()


    def calculate_and_plot_profiles(self,velocities,accelerations, step_number):
        # Calculate velocities and accelerations


        # Plot profiles
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(velocities, label="Geschwindigkeit")
        plt.title(f"Schritt {step_number}: Geschwindigkeitsprofil")
        plt.xlabel("Trajektorie Punkt")
        plt.ylabel("Geschwindigkeit")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(accelerations, label="Beschleunigung", color="orange")
        plt.title(f"Schritt {step_number}: Beschleunigungsprofil")
        plt.xlabel("Trajektorie Punkt")
        plt.ylabel("Beschleunigung")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    