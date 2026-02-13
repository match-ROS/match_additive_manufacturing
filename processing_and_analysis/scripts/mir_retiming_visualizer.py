#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
import tf.transformations as tf
import argparse
from std_msgs.msg import Float32MultiArray

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
        self.index_offset = None
        rospy.Subscriber(
            "/mur620c/mir_index_offset",
            Float32MultiArray,
            self.cb_offset
        )

    def cb_ur(self, msg):
        self.ur_path = msg

    def cb(self, msg):
        self.path = msg

    def cb_offset(self, msg):
        self.index_offset = np.array(msg.data)


    def wait_for_path(self):
        rospy.loginfo("Waiting for MiR and UR path...")
        while not rospy.is_shutdown() and self.path is None or self.ur_path is None:
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

    def apply_index_offset(self, x, y, yaw, offset):

        idx = np.arange(len(x)) + offset
        idx = np.clip(idx, 0, len(x)-1)

        x_i = np.interp(idx, np.arange(len(x)), x)
        y_i = np.interp(idx, np.arange(len(y)), y)
        yaw_i = np.interp(idx, np.arange(len(yaw)), yaw)

        return x_i, y_i, yaw_i


    def compute_reach(self, base_x, base_y, ur_x, ur_y):

        n = min(len(base_x), len(ur_x))

        return np.sqrt(
            (base_x[:n] - ur_x[:n])**2 +
            (base_y[:n] - ur_y[:n])**2
        )

    def velocity_proxy(self, x, y):

        dx = np.diff(x)
        dy = np.diff(y)

        return np.sqrt(dx**2 + dy**2)


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

        for i, pose in enumerate(self.path.poses):

            z = pose.pose.position.z

            if last_z is None:
                last_z = z

            if abs(z - last_z) > threshold:
                layers.append(current)
                current = []

            current.append((i, pose))  # ← index speichern!
            last_z = z

        if current:
            layers.append(current)

        return layers

    def plot_velocity_colormap(self,
                            x0, y0,
                            x1, y1,
                            ur_x, ur_y,
                            filename="velocity_colormap.pdf"):

        # --- velocity proxies ---
        v0 = self.velocity_proxy(x0, y0)
        v1 = self.velocity_proxy(x1, y1)

        n = min(len(v0), len(v1), len(ur_x)-1)

        eps = 1.0
        ratio = (v1[:n]+eps) / (v0[:n] + eps)

        delta = (ratio - 1.0) * 3.0  

        vmax = np.max(np.abs(delta))
        if vmax < 1e-6:
            vmax = 1.0

        colors = delta / vmax

        cmap = plt.get_cmap("coolwarm")

        plt.figure(figsize=(7,7))

        # --- MiR colored ---
        plt.scatter(
            x1[:n],
            y1[:n],
            c=colors,
            cmap=cmap,
            s=10,
            label="MiR (retimed)"
        )

        # --- UR colored ---
        plt.scatter(
            ur_x[:n],
            ur_y[:n],
            c=colors,
            cmap=cmap,
            s=10,
            marker="x",
            label="UR TCP"
        )

        # original MiR reference
        plt.plot(x0, y0, "--", linewidth=1,
                label="MiR original")

        plt.axis("equal")

        cbar = plt.colorbar()
        cbar.set_label(
            "velocity change\n(orange = faster, blue = slower)"
        )

        plt.title("Retiming Effect on MiR and UR Trajectories")
        plt.legend()

        plt.savefig(filename)
        plt.close()




    def visualize(self):

        layers = self.split_layers()
        selected = parse_layer_selection(layer_selection, len(layers))

        poses = []
        indices = []

        for i in selected:
            for idx, pose in layers[i-1]:
                poses.append(pose)
                indices.append(idx)

        indices = np.array(indices)
        offset_layer = self.index_offset[indices]



        x, y, yaw, t = self.extract_arrays(poses)
        x0, y0, yaw0 = x, y, yaw
        x1, y1, yaw1 = self.apply_index_offset(
            x0, y0, yaw0, offset_layer
        )

        s = self.equivalent_arc_length(x, y, yaw)

        ur_x, ur_y = self.extract_ur_xy()

        base_x, base_y = self.compute_ur_base(x, y, yaw)
        bx0, by0 = self.compute_ur_base(x0, y0, yaw0)
        bx1, by1 = self.compute_ur_base(x1, y1, yaw1)
        # d0 = self.compute_reach(bx0, by0, ur_x, ur_y)
        # d1 = self.compute_reach(bx1, by1, ur_x, ur_y)
        v0 = self.velocity_proxy(x0, y0)
        v1 = self.velocity_proxy(x1, y1)
        d_reach_no = self.compute_reach(bx0, by0, ur_x, ur_y)
        d_reach_off = self.compute_reach(bx1, by1, ur_x, ur_y)

        # simple index alignment
        min_len = min(len(base_x), len(ur_x))
        dist = np.sqrt(
            (base_x[:min_len] - ur_x[:min_len])**2 +
            (base_y[:min_len] - ur_y[:min_len])**2
        )


        s_target = np.linspace(0, s[-1], len(s))
        i_target = np.interp(s_target, s, np.arange(len(s)))
        d_arc = i_target - np.arange(len(s))

        alpha = 0.6
        d = alpha * d_arc

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
        plt.plot(d_arc, label="arc-length")
        plt.plot(d, label="scaled")
        plt.plot(offset_layer, label="final optimizer")
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
        plt.plot(d_reach_no, label="no offset")
        plt.plot(d_reach_off, label="with offset")
        plt.axhline(1.2, linestyle="--", label="limit")
        plt.legend()
        plt.title("Reach Utilization Comparison")
        plt.savefig("reach_comparison.pdf")
        plt.close()

        plt.figure()
        plt.plot(v0, label="no offset")
        plt.plot(v1, label="with offset")
        plt.legend()
        plt.title("MiR Velocity Comparison")
        plt.savefig("velocity_comparison.pdf")
        plt.close()


        plt.figure()
        plt.plot(base_x, base_y, label="UR base")
        plt.plot(ur_x, ur_y, label="UR TCP")
        plt.plot(x, y, "--", label="MiR path")

        plt.axis("equal")
        plt.legend()
        plt.title("Base vs TCP")
        plt.savefig("base_tcp_xy.pdf")
        plt.close()

        self.plot_velocity_colormap(
            x0, y0,
            x1, y1,
            ur_x, ur_y
        )




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
