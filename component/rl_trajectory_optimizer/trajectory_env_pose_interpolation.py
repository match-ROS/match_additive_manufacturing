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
        self.max_distance = 1.2
        self.time_step = time_step
        self.step_counter = 0
        self.previous_velocity_rmse = 0
        self.total_step_counter = 0
        self.reward_history = []
        self.cumulative_reward = 0.0
        self.episode_length = 0
        self.last_step_reward = 0.0


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
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(distances)))
        self.trajectory_length = self.cumulative_lengths[-1]

        smoothing_factor = 0.01  # Anpassbar: größerer Wert = glatter
        self.cs_x = UnivariateSpline(self.cumulative_lengths, x, s=smoothing_factor)
        self.cs_y = UnivariateSpline(self.cumulative_lengths, y, s=smoothing_factor)

        self.uniform_lengths = np.linspace(0, self.cumulative_lengths[-1], len(base_trajectory))
        # Evaluiere den Spline für gleichmäßige Bogenlängen
        x_smooth = self.cs_x(self.uniform_lengths)
        y_smooth = self.cs_y(self.uniform_lengths)
        self.splined_trajectory = np.vstack((x_smooth, y_smooth)).T
        # initialisierte initial_lengths als nur nullen
        self.current_lengths = np.zeros_like(self.uniform_lengths)
        self.current_trajectory = self.splined_trajectory.copy()

        self.action_space = spaces.Box(
            low=0.0001,
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
        cs_x = UnivariateSpline(t, base_trajectory[:, 0], s=0.01)
        cs_y = UnivariateSpline(t, base_trajectory[:, 1], s=0.01)
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
        self.current_lengths = np.zeros_like(self.uniform_lengths)
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
        if self.worsening_count >= 10:  # Beispiel: nach 5 Verschlechterungen
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
        # action gleich aufsteigende Schritte
    
        length_steps = action * self.scale_factor
        self.current_lengths += np.concatenate(([0], np.cumsum(length_steps)))

        # normiere current_length auf die Bahnlänge self.trajectory_length
        if self.current_lengths[-1] > self.trajectory_length:
            self.current_lengths = self.current_lengths / self.current_lengths[-1] * self.trajectory_length
        else:
            self.current_lengths = np.concatenate((self.current_lengths[0:-1], [self.trajectory_length]))
            
        
        # self.uniform_lengths = np.linspace(0, self.cumulative_lengths[-1], len(self.base_trajectory))
        # self.current_lengths = self.uniform_lengths

        # Evaluiere den Spline für gleichmäßige Bogenlängen
        x = self.cs_x(self.current_lengths)
        y = self.cs_y(self.current_lengths)
        x_smooth = gaussian_filter1d(x, sigma=2)
        y_smooth = gaussian_filter1d(y, sigma=2)
        new_trajectory = np.vstack((x_smooth, y_smooth)).T
        
        # Berechne trajektorie der manipulatorbasis
        self.manipulator_positions = self.calculate_manipulator_base_positions(new_trajectory)

        # Berechnung der Abstände zwischen Manipulatorbasis und TCP-Trajektorie
        tcp_distances = np.linalg.norm(self.manipulator_positions - self.tcp_trajectory, axis=1)

        # Überschreitung der maximalen Distanz berechnen
        exceedances = tcp_distances - self.max_distance

        # Nur positive Überschreitungen berücksichtigen
        positive_exceedances = np.maximum(0, exceedances)

        # Proportionale Bestrafung
        penalty_weight = 10.0  # Gewichtung der Bestrafung
        tcp_distance_penalty = np.sum(positive_exceedances * penalty_weight)

        # Bestrafe zu kurze Trajektorien
        length_penalty = (self.trajectory_length - self.current_lengths[-2]) * 100.0
 
        #new_trajectory[-1] = new_trajectory[-2]  # TODO: irgendwas stimmt mit dem letzten Punkt nicht 
        #        
        # Berechnung der Distanzen zwischen Punkten
        distances = np.linalg.norm(np.diff(new_trajectory, axis=0), axis=1)
        distances_base = np.linalg.norm(np.diff(self.base_trajectory, axis=0), axis=1)

        # Berechnung der Geschwindigkeiten zwischen Punkten
        velocities = distances / self.time_step

        # Berechnung der Differenzen der Geschwindigkeiten
        velocity_differences = np.diff(velocities)

        # Berechnung der Beschleunigungen
        accelerations = velocity_differences / self.time_step

        # Berechnung der Geschwindigkeitsänderungen
        velocity_penalty = np.std(velocities) * 1.1

        # Berechne maximale Geschwindigkeit
        max_velocity_penalty = np.max(velocities) * 0.2

        # Berechnung der Beschleunigungsänderungen
        acceleration_penalty = np.sum(abs(accelerations)) * 0.1

        # Berechne maximale Beschleunigung
        max_acceleration_penalty = np.max(accelerations) * 0.0002

        # Berechne Acc RMSE  
        acc_rmse_penalty = np.sqrt(np.mean(accelerations ** 2)) * 0.001

        # Berechne Velocity RMSE
        velocity_rmse_penalty = np.sqrt(np.mean(velocities ** 2))

        # Bestrafung für 
        uniformity_penalty = np.std(distances) * 1000.0

        # Maximale Abweichung der Abstände
        max_distance_penalty = np.max(distances) * 2000.0


        reward = (
            1000.0
            - uniformity_penalty 
            - max_distance_penalty 
            - tcp_distance_penalty 
            - velocity_penalty 
            - acceleration_penalty 
            - max_velocity_penalty
            - max_acceleration_penalty 
            - acc_rmse_penalty 
            - velocity_rmse_penalty 
            - length_penalty
        )

        

        # Aktualisierung der Trajektorie
        self.current_trajectory = new_trajectory
        self.step_counter += 1
        self.total_step_counter += 1
        self.episode_length += 1
        self.last_step_reward = reward  # Speichere den aktuellen Reward

        # Prüfung auf Terminierung
        terminated = self.monitor_reward(reward)  # Optional: Bedingung hinzufügen, wenn nötig
        truncated = self.step_counter >= 0  # Episodenlänge begrenzen
        #truncated = True  # Episodenlänge nicht begrenzen
        if terminated or truncated:
            #info = {"episode": {"r": self.cumulative_reward, "l": self.episode_length}}
            info = {"episode": {"r": reward, "l": self.episode_length}}
            self.episode_length = 0  # Zurücksetzen für nächste Episode
            self.step_counter = 0
            self.reset()#
            reward_output = reward
        else:
            info = {}
            reward_output = 0.0



        # Visualisierung nach bestimmten Schritten (optional)
        if self.total_step_counter % 10000 == 0:
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
            print(f"Length Penalty: {length_penalty:.2f}")
            # self.plot_current_trajectory(self.step_counter)
            # self.calculate_and_plot_profiles(velocities, accelerations, self.step_counter)
            # self.plot_distance_profile(distances)
            #self.plot_distance_profile(distances_base)

        # Beobachtung zurückgeben
        obs = self.normalize_observations(self.current_trajectory)
        obs = obs.flatten()  # Flache 1D-Struktur erzeugen
        return obs, reward_output, terminated, truncated, info




    def get_current_trajectory(self):
        return self.current_trajectory.copy()
    
    def calculate_manipulator_base_positions(self,new_trajectory, delta_x=0.55, delta_y=-0.32):
        """
        Berechnet die wahren Positionen der Manipulatorbasis entlang der Plattformtrajektorie.

        Args:
        - new_trajectory: Die optimierte Basistrajektorie (N x 2).
        - delta_x: Verschiebung des Manipulators in x-Richtung relativ zur Plattform.
        - delta_y: Verschiebung des Manipulators in y-Richtung relativ zur Plattform.

        Returns:
        - manipulator_positions: Array mit den wahren Positionen der Manipulatorbasis (N x 2).
        """
        # Berechne die Richtungsvektoren
        directions = np.diff(new_trajectory, axis=0)
        directions = []
        for i in range(1, len(new_trajectory)):
            directions.append(new_trajectory[i] - new_trajectory[i - 1])
        directions = np.array(directions)
        
        # Berechne die Orientierung (Theta) entlang der Trajektorie
        thetas = np.arctan2(directions[:, 1], directions[:, 0])

        # plot new_trajectory xy
        # plt.figure()
        # plt.plot(new_trajectory[:, 0], label="x")
        # plt.plot(new_trajectory[:, 1], label="y")
        # plt.title("Trajectory")
        # plt.xlabel("Trajectory Point")
        # plt.ylabel("Position")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # # plot new_trajectory
        # plt.figure()
        # plt.plot(new_trajectory[:, 0], new_trajectory[:, 1])
        # plt.title("Trajectory values")
        # plt.xlabel("Trajectory Point")
        # plt.ylabel("Position")
        # plt.grid()
        # plt.show()

        # # plot thetas
        # plt.figure()
        # plt.plot(thetas)
        # plt.title("Theta values")
        # plt.xlabel("Trajectory Point")
        # plt.ylabel("Theta")
        # plt.grid()
        # plt.show()

        # Füge den letzten Winkel als Kopie hinzu (damit die Länge passt)
        thetas = np.append(thetas, thetas[-1])

        # Rotationsmatrix anwenden (vektorisiert)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        rotated_offsets_x = cos_thetas * delta_x - sin_thetas * delta_y
        rotated_offsets_y = sin_thetas * delta_x + cos_thetas * delta_y
        rotated_offsets = np.vstack((rotated_offsets_x, rotated_offsets_y)).T

        # Addiere die rotierte Verschiebung zur Plattformposition
        manipulator_positions = new_trajectory + rotated_offsets

        return manipulator_positions

    
    def calculate_path_deviation(self, trajectory):
        """
        Berechnet die Abweichung der aktuellen Trajektorie von der gesplinten Trajektorie.
        """
        deviations = np.linalg.norm(trajectory - self.splined_trajectory, axis=1)
        total_deviation = np.sum(deviations)
        return total_deviation
    

    def plot_trajectories_over_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.base_trajectory[:, 0], 'b--', label='Base Trajectory')
        plt.plot(self.tcp_trajectory[:, 0], 'r-', label='Modified Trajectory')
        plt.title('Trajectory Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('X')
        plt.grid()
        plt.legend()
        plt.show()

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
        plt.plot(self.manipulator_positions[:, 0], self.manipulator_positions[:, 1], 'g-', label='Manipulator Basis')
        plt.plot(self.current_trajectory[:, 0], self.current_trajectory[:, 1], 'r-', label='current Trajectory')
        plt.scatter(self.base_trajectory[:, 0], self.base_trajectory[:, 1], color='blue', s=10, label='base Trajectory')
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
    

    def normalize_observations(self, trajectory):
        x_min = np.min(trajectory[:, 0])
        x_max = np.max(trajectory[:, 0])
        y_min = np.min(trajectory[:, 1])
        y_max = np.max(trajectory[:, 1])
        x_norm = (trajectory[:, 0] - x_min) / (x_max - x_min) * 2 - 1
        y_norm = (trajectory[:, 1] - y_min) / (y_max - y_min) * 2 - 1
        return np.vstack((x_norm, y_norm)).T