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
