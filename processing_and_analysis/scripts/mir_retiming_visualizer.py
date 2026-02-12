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
        self.ur_path = None
        rospy.Subscriber("/mur620c/ur_path_original", Path, self.cb_ur)

    def cb_ur(self, msg):
        self.ur_path = msg

    def cb(self, msg):
        self.path = msg

    def wait_for_path(self):
        rospy.loginfo("Waiting for MiR path...")
        while not rospy.is_shutdown() and self.path is None:
            rospy.sleep(0.1)

    def extract_ur_xy(self):

        if self.ur_path is None:
            rospy.logwarn("UR path not received!")
            return None, None

        xs, ys = [], []

        for p in self.ur_path.poses:
            xs.append(p.pose.position.x)
            ys.append(p.pose.position.y)

        return np.array(xs), np.array(ys)

    def compute_ur_base(self, x, y, yaw):

        mount_x = 0.549
        mount_y = -0.318

        base_x = x + np.cos(yaw)*mount_x - np.sin(yaw)*mount_y
        base_y = y + np.sin(yaw)*mount_x + np.cos(yaw)*mount_y

        return base_x, base_y


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

        ur_x, ur_y = self.extract_ur_xy()

        base_x, base_y = self.compute_ur_base(x, y, yaw)

        # simple index alignment
        min_len = min(len(base_x), len(ur_x))
        dist = np.sqrt(
            (base_x[:min_len] - ur_x[:min_len])**2 +
            (base_y[:min_len] - ur_y[:min_len])**2
        )


        s_target = np.linspace(0, s[-1], len(s))
        i_target = np.interp(s_target, s, np.arange(len(s)))
        d0 = i_target - np.arange(len(s))

        alpha = 0.6
        d = alpha * d0

        fig, axs = plt.subplots(2,2, figsize=(12,8))

        plt.figure()
        plt.plot(x,y)
        plt.axis("equal")
        plt.title("MiR Geometry")
        plt.savefig("geometry.pdf")
        plt.close()


        plt.figure()
        plt.plot(s, label="s(k)")
        plt.plot(s_target, label="target")
        plt.legend()
        plt.title("Arc Length Mapping")
        plt.savefig("arc_length.pdf")
        plt.close()


        plt.figure()
        plt.plot(d0, label="d0")
        plt.plot(d, label="scaled")
        plt.legend()
        plt.title("Index Offset")
        plt.savefig("offset.pdf")
        plt.close()

        plt.figure()
        plt.plot(np.diff(s))
        plt.title("Velocity Proxy")
        plt.savefig("velocity.pdf")
        plt.close()

        plt.figure()
        plt.plot(dist)
        plt.axhline(1.1, linestyle="--", label="reach limit")
        plt.legend()
        plt.title("UR Reach Utilization")
        plt.savefig("reachability.pdf")
        plt.close()

        plt.figure()
        plt.plot(base_x, base_y, label="UR base")
        plt.plot(ur_x, ur_y, label="UR TCP")
        plt.axis("equal")
        plt.legend()
        plt.title("Base vs TCP")
        plt.savefig("base_tcp_xy.pdf")
        plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="2",
                        help="Layer selection: e.g. 1, 1-3, 2,4")
    args, unknown = parser.parse_known_args()

    layer_selection = args.layers


    rospy.init_node("mir_retiming_visualizer")
    vis = RetimingVisualizer()
    vis.wait_for_path()
    vis.visualize()
