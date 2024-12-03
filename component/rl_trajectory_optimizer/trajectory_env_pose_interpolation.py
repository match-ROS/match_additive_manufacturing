#! /usr/bin/env python3
import numpy as np
#import gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, UnivariateSpline
import os
from scipy.spatial import KDTree
from scipy.interpolate import Rbf




class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, tcp_trajectory, base_trajectory, time_step=0.1):
        super(TrajectoryOptimizationEnv, self).__init__()
        self.tcp_trajectory = np.array(tcp_trajectory)
        self.base_trajectory = np.array(base_trajectory)
        self.scale_factor = 1.0 # 0.029031316514772237 * 0.01
        self.max_distance = 2.0
        self.time_step = time_step
        self.step_counter = 0
        self.previous_velocity_rmse = 0
        self.total_step_counter = 0
        self.reward_history = []
        self.cumulative_reward = 0.0
        self.episode_length = 0


        smoothed_x = np.zeros_like(self.base_trajectory)
        smoothed_y = np.zeros_like(self.base_trajectory)
        smoothed_x[:, 0] = gaussian_filter1d(self.base_trajectory[:, 0], sigma=2)
        smoothed_y[:, 1] = gaussian_filter1d(self.base_trajectory[:, 1], sigma=2)

         # doppelte Punkte entfernen
        for i in range(1, len(smoothed_x)):
            if np.all(smoothed_x[i] == smoothed_x[i - 1]) or np.all(smoothed_y[i] == smoothed_y[i - 1]):
                # remove the point
                smoothed_x[i] = np.nan
                smoothed_y[i] = np.nan

        smoothed_x = smoothed_x[~np.isnan(smoothed_x).any(axis=1)]
        smoothed_y = smoothed_y[~np.isnan(smoothed_y).any(axis=1)]

        x = smoothed_x[:, 0]
        y = smoothed_y[:, 1]

        x, y = x[::5], y[::5] # Jeden 5. Punkt verwenden

        # 2. Berechnung der Bogenlänge
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_lengths = np.concatenate(([0], np.cumsum(distances)))
        self.trajectory_length = cumulative_lengths[-1]

        smoothing_factor = 0.01  # Anpassbar: größerer Wert = glatter
        self.cs_x = UnivariateSpline(cumulative_lengths, x, s=smoothing_factor)
        self.cs_y = UnivariateSpline(cumulative_lengths, y, s=smoothing_factor)
        
        self.tcp_cs_x = UnivariateSpline(cumulative_lengths, self.tcp_trajectory[:, 0], s=smoothing_factor)
        self.tcp_cs_y = UnivariateSpline(cumulative_lengths, self.tcp_trajectory[:, 1], s=smoothing_factor)

        uniform_lengths = np.linspace(0, cumulative_lengths[-1], len(base_trajectory))
        # Evaluiere den Spline für gleichmäßige Bogenlängen
        x_smooth = self.cs_x(uniform_lengths)
        y_smooth = self.cs_y(uniform_lengths)
        self.splined_trajectory = np.vstack((x_smooth, y_smooth)).T
        # initialisierte initial_lengths als nur nullen
        self.current_lengths = np.zeros_like(uniform_lengths)
        self.current_trajectory = self.splined_trajectory.copy()

        self.action_space = spaces.Box(
            low=0.00001,
            high=0.001,
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
        # self.cumulative_reward = 0.0
        # self.episode_length = 0
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
        if self.worsening_count >= 20:  # Beispiel: nach 5 Verschlechterungen
            terminated = True
            #print(f"Episode beendet: Wiederholte Verschlechterungen (Count: {self.worsening_count})")
        else:
            terminated = False
        return terminated

    def save_trajectory_log_txt(self,trajectory, step_count, log_dir="./logs/"):
        """
        Speichert die x- und y-Werte der Trajektorie in separate .txt-Dateien.
        Args:
            trajectory (np.ndarray): Die aktuelle Trajektorie, ein 2D-Array mit Spalten [x, y].
            step_count (int): Der aktuelle Schritt des Trainings.
            log_dir (str): Das Verzeichnis, in dem die Logs gespeichert werden sollen.
        """
        # Erstelle das Verzeichnis, falls es nicht existiert
        os.makedirs(log_dir, exist_ok=True)

        # Extrahiere x- und y-Werte
        x_values = trajectory[:, 0]
        y_values = trajectory[:, 1]

        # Definiere die Dateinamen
        x_file = os.path.join(log_dir, f"trajectory_x_step_{step_count}.txt")
        y_file = os.path.join(log_dir, f"trajectory_y_step_{step_count}.txt")

        # Speichere die Werte als separate Dateien
        np.savetxt(x_file, [x_values], header="x", comments="", fmt="%.6f", delimiter=",")
        np.savetxt(y_file, [y_values], header="y", comments="", fmt="%.6f", delimiter=",")


    def step(self, action):
        
        length_steps = action
        self.current_lengths += np.concatenate(([0], np.cumsum(length_steps)))

        # normiere current_length auf die Bahnlänge self.trajectory_length
        self.current_lengths = self.current_lengths / self.current_lengths[-1] * self.trajectory_length
        
        # Aktion elementweise quadrieren
        #action = np.square(action)
        # Evaluiere den Spline für gleichmäßige Bogenlängen
        x_smooth = self.cs_x(self.current_lengths)
        y_smooth = self.cs_y(self.current_lengths)
        new_trajectory = np.vstack((x_smooth, y_smooth)).T
        #print(f"Time intervals: {time_intervals}")
        #print(f"New time: {self.t_new}")
        
        # test
        # t_test = np.linspace(0, 0.1, len(self.base_trajectory)-1)
        # t_test = np.concatenate(((t_test), [1]))

        # Interpolieren mit der synchronisierten Punktanzahl
        # new_trajectory = np.vstack((self.cs_x(self.t_new), self.cs_y(self.t_new))).T

        #new_trajectory = np.vstack((self.cs_x(t_test), self.cs_y(t_test))).T

        # Berechnung der Distanzen zwischen Punkten
        distances = np.linalg.norm(np.diff(new_trajectory, axis=0), axis=1)
        distances_base = np.linalg.norm(np.diff(self.base_trajectory, axis=0), axis=1)

        # Berechnung der Geschwindigkeiten zwischen Punkten
        t_original = np.linspace(0, 1, len(self.base_trajectory))
        time_intervals = np.diff(t_original)
        velocities = distances / time_intervals

        # Berechnung der Differenzen der Geschwindigkeiten
        velocity_differences = np.diff(velocities)

        # Berechnung der Beschleunigungen
        accelerations = velocity_differences / time_intervals[1:]

        # Berechnung der Geschwindigkeitsänderungen
        velocity_penalty = np.std(velocities) * 1.1

        # Berechne maximale Geschwindigkeit
        max_velocity_penalty = np.max(velocities) * 0.2

        # Berechnung der Beschleunigungsänderungen
        acceleration_penalty = np.std(accelerations) * 0.002 

        # Berechne maximale Beschleunigung
        max_acceleration_penalty = np.max(accelerations) * 0.0002

        # Berechne Acc RMSE  
        acc_rmse_penalty = np.sqrt(np.mean(accelerations ** 2)) * 0.001

        # Berechne Velocity RMSE
        velocity_rmse_penalty = np.sqrt(np.mean(velocities ** 2))

        # Interpolation der TCP-Trajektorie auf die gleichen Zeitstempel
        tcp_interpolated = np.vstack((self.tcp_cs_x(self.t_new), self.tcp_cs_y(self.t_new))).T

        # Berechnung der euklidischen Abstände zwischen den Trajektorien
        tcp_distances = np.linalg.norm(new_trajectory - tcp_interpolated, axis=1)

        # Bestrafung für Punkte, die weiter als die maximale Distanz entfernt sind
        tcp_distance_penalty = np.sum(np.maximum(0, tcp_distances - self.max_distance))

        # Bestrafung für 
        uniformity_penalty = np.std(distances) * 1000.0 * 10.0

        # Maximale Abweichung der Abstände
        max_distance_penalty = np.max(distances) * 200.0


        reward = (
            10.0
            - uniformity_penalty 
            - max_distance_penalty 
            - tcp_distance_penalty * 10.0
            - velocity_penalty 
            - acceleration_penalty 
            - max_velocity_penalty * 0.1
            - max_acceleration_penalty 
            - acc_rmse_penalty 
            - velocity_rmse_penalty 
        )

        # Kumulative Belohnung
        self.cumulative_reward += reward

        # Aktualisierung der Trajektorie
        self.current_trajectory = new_trajectory
        self.step_counter += 1
        self.total_step_counter += 1
        self.episode_length += 1

        # Prüfung auf Terminierung
        terminated = self.monitor_reward(reward)  # Optional: Bedingung hinzufügen, wenn nötig
        truncated = self.step_counter >= 1000  # Episodenlänge begrenzen
        #truncated = True  # Episodenlänge nicht begrenzen
        if terminated or truncated:
            info = {"episode": {"r": self.cumulative_reward, "l": self.episode_length}}
            self.cumulative_reward = 0.0  # Zurücksetzen für nächste Episode
            self.episode_length = 0  # Zurücksetzen für nächste Episode
        else:
            info = {}



        # Visualisierung nach bestimmten Schritten (optional)
        if self.total_step_counter % 1 == 0:
            self.save_trajectory_log_txt(self.current_trajectory, self.total_step_counter, log_dir="./logs/")
            print(f"Step {self.total_step_counter}: Reward={reward:.2f}")
            print(f"tcp_distance_penalty Penalty: {tcp_distance_penalty:.2f}")
            print(f"Uniformity Penalty: {uniformity_penalty:.2f}")
            print(f"Max Distance Penalty: {max_distance_penalty:.2f}")
            print(f"Velocity Penalty: {velocity_penalty:.2f}")
            print(f"Acceleration Penalty: {acceleration_penalty:.2f}")
            print(f"Max Velocity Penalty: {max_velocity_penalty:.2f}")
            print(f"Max Acceleration Penalty: {max_acceleration_penalty:.2f}")
            print(f"Acc RMSE Penalty: {acc_rmse_penalty:.2f}")
            print(f"Velocity RMSE Penalty: {velocity_rmse_penalty:.2f}")
            self.plot_current_trajectory(self.step_counter)
            self.calculate_and_plot_profiles(velocities, accelerations, self.step_counter)
            self.plot_distance_profile(distances)
            self.plot_distance_profile(distances_base)

        # Beobachtung zurückgeben
        obs = self.current_trajectory.astype(np.float32)
        obs = obs.flatten()  # Flache 1D-Struktur erzeugen
        return obs, reward, terminated, truncated, info

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


    def plot_distance_profile(self, distances):
        plt.figure(figsize=(8, 6))
        plt.plot(distances, label="Abstand Basistrajektorie-TCP")
        plt.title("Abstand zwischen Punkten der optimierten Trajektorie")
        plt.xlabel("Trajektorie-Punkt")
        plt.ylabel("Abstand")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_current_trajectory(self, step_counter):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.splined_trajectory[:, 0], self.splined_trajectory[:, 1], 'b--', label='Splined Trajectory')
        plt.plot(self.current_trajectory[:, 0], self.current_trajectory[:, 1], 'r-', label='Modified Trajectory')
        plt.scatter(self.base_trajectory[:, 0], self.base_trajectory[:, 1], color='blue', s=10, label='Splined TCP Trajectory')
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
    