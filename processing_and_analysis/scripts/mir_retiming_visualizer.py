#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
import tf.transformations as tf
import argparse

def parse_layer_selection(sel, max_layer):

    result = set()

    parts = sel.split(",")

    for p in parts:
        if "-" in p:
            a, b = map(int, p.split("-"))
            result.update(range(a, b+1))
        else:
            result.add(int(p))

    return [i for i in sorted(result) if 1 <= i <= max_layer]


class RetimingVisualizer:

    def __init__(self):
        self.path = None
        rospy.Subscriber("/mur620c/mir_path_original", Path, self.cb)

    def cb(self, msg):
        self.path = msg

    def wait_for_path(self):
        rospy.loginfo("Waiting for MiR path...")
        while not rospy.is_shutdown() and self.path is None:
            rospy.sleep(0.1)

    def extract_arrays(self, poses):
        xs, ys, yaw = [], [], []

        for p in poses:
            xs.append(p.pose.position.x)
            ys.append(p.pose.position.y)

            q = [
                p.pose.orientation.x,
                p.pose.orientation.y,
                p.pose.orientation.z,
                p.pose.orientation.w
            ]
            _, _, psi = tf.euler_from_quaternion(q)
            yaw.append(psi)

        xs = np.array(xs)
        ys = np.array(ys)
        yaw = np.unwrap(np.array(yaw))

        t = np.linspace(0, len(xs)-1, len(xs))

        return xs, ys, yaw, t

    def equivalent_arc_length(self, x, y, yaw):
        dx = np.diff(x)
        dy = np.diff(y)
        dyaw = np.diff(yaw)

        v_lin = np.sqrt(dx**2 + dy**2)
        v_rot = np.abs(dyaw) * 0.45

        v = np.maximum(v_lin, v_rot)

        s = np.zeros(len(x))
        s[1:] = np.cumsum(v)

        return s

    def split_layers(self, threshold=1e-3):

        layers = []
        current = []

        last_z = None

        for pose in self.path.poses:

            z = pose.pose.position.z

            if last_z is None:
                last_z = z

            if abs(z - last_z) > threshold:
                layers.append(current)
                current = []

            current.append(pose)
            last_z = z

        if current:
            layers.append(current)

        return layers


    def visualize(self):

        layers = self.split_layers()

        selected = parse_layer_selection(layer_selection, len(layers))

        poses = []
        for i in selected:
            poses.extend(layers[i-1])


        x, y, yaw, t = self.extract_arrays(poses)

        s = self.equivalent_arc_length(x, y, yaw)

        s_target = np.linspace(0, s[-1], len(s))
        i_target = np.interp(s_target, s, np.arange(len(s)))
        d0 = i_target - np.arange(len(s))

        alpha = 0.6
        d = alpha * d0

        fig, axs = plt.subplots(2,2, figsize=(12,8))

        axs[0,0].plot(x,y)
        axs[0,0].set_title("MiR Path Geometry")
        axs[0,0].axis("equal")

        axs[0,1].plot(s, label="s(k)")
        axs[0,1].plot(s_target, label="target")
        axs[0,1].legend()
        axs[0,1].set_title("Equivalent Arc-Length")

        axs[1,0].plot(d0, label="d^(0)")
        axs[1,0].plot(d, label="scaled")
        axs[1,0].legend()
        axs[1,0].set_title("Index Offset")

        v = np.diff(s)
        axs[1,1].plot(v)
        axs[1,1].set_title("Velocity Proxy")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="1-2",
                        help="Layer selection: e.g. 1, 1-3, 2,4")
    args, unknown = parser.parse_known_args()

    layer_selection = args.layers


    rospy.init_node("mir_retiming_visualizer")
    vis = RetimingVisualizer()
    vis.wait_for_path()
    vis.visualize()
