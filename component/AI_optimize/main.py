#! /usr/bin/env python3
from print_path import xMIR, yMIR, xTCP, yTCP
import numpy as np
import gym
from gym import spaces

class TrajectoryOptimizationEnv(gym.Env):
    def __init__(self, max_distance=2.0):
        super(TrajectoryOptimizationEnv, self).__init__()
        
        # Trajektorien einlesen
        self.tcp_trajectory = np.array([xTCP.xTCP(), yTCP.yTCP()]).T  # Manipulator (TCP)
        self.base_trajectory = np.array([xMIR.xMIR(), yMIR.yMIR()]).T  # Basis

        self.max_distance = max_distance

        # Observation: Positionsdifferenzen zwischen Punkten
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.base_trajectory.shape, dtype=np.float32)
        # Action: Verschiebung eines Punktes entlang des Pfades (-1, 0, 1)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_trajectory = self.base_trajectory.copy()
        self.current_index = 0
        return self.current_trajectory

    def step(self, action):
        if action == 0:
            pass  # Keine Veränderung
        elif action == 1 and self.current_index < len(self.current_trajectory) - 1:
            self.current_trajectory[self.current_index] += (self.current_trajectory[self.current_index + 1] - self.current_trajectory[self.current_index]) * 0.1
        elif action == -1 and self.current_index > 0:
            self.current_trajectory[self.current_index] += (self.current_trajectory[self.current_index - 1] - self.current_trajectory[self.current_index]) * 0.1

        velocities = np.diff(self.current_trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)

        acceleration_penalty = np.sum(np.abs(accelerations))
        distances = np.linalg.norm(self.current_trajectory - self.tcp_trajectory, axis=1)
        distance_penalty = np.sum(distances[distances > self.max_distance])

        reward = -acceleration_penalty - distance_penalty

        done = self.current_index == len(self.current_trajectory) - 1

        self.current_index += 1

        return self.current_trajectory, reward, done, {}

    def render(self, mode="human"):
        print(f"Current trajectory: {self.current_trajectory}")


Hi, ich habe eine Aufgabe. Die Aufgabe ist recht komplex, deswegen frag bitte nach wenn du nicht sicher bist ob du die Aufgabe korrekt verstanden hast. Ich möchte mittels reinforcement learning die Trajektorie eines mobilen Roboters optimieren. Der mobile Roboter hat einen Basis und einen Manipulator. Ich habe eine Trajektorie für den Manipulator (TCP), welche fest vorgegeben ist und nicht verändert werden darf (es handelt sich um eine 3D Druck Bahn). Für jeden Punkt auf der TCP Trajektorie ist ebenfalls ein Zielpunkt für die mobile Roboterplattform vorgegeben. Die Trajektorie für die Plattform/Basis wurde mittels Konturaufweitung erzeugt ist ist daher nicht optimal. Die Trajektorie für die Basis darf also verändert werden. Allerdings darf der Pfad den die Plattform fährt nicht verändert werden. Die Punkte der Trajektorie dürfen also nur näher zusammnen geschoben oder auseinandergezogen werden. Es kann in Abhängigkeit der Kontur des Bauteils also vorkommen, dass mehrere Basis Trajektorienpunkte zusammenfallen – also auf einen Punkt. Die Basis muss dann an diesem Punkt stoppen. Gleichzeitig können die Trajektorienpunkte der Basis sehr weit auseinanderliegen, wodurch die Basis dann sehr schnell fahren muss und einen hohen Regelfehler bekommt. Mein Ziel ist es also die Punkte auf der Basis trajektorie besser zu verteilen, damit die Basis gleichmäßiger fährt – also weniger stehen und dafür langsamer in den schnellsten Passagen. Allerdings muss dabei immer beachtet werden, dass der Manipulator nur eine begrenzte Reichweite (2 meter) hat. Die Basistrajektorie kann also nur unter der Nebenbedingungen verändert werden, dass die Basis näher als 2 m an der TCP zielposition liegt (welche nicht verändert werden kann und darf). Ich möchte die Bahn der Basis für diesen Fall optimieren, um eine möglichst gleichmäßige geschwindigkeit und möglichst geringe Beschleunigung der Basis zu haben. Hast du diese Aufgabe und die Nebenbedingungen verstanden?
