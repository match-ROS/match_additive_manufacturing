#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt


BAG_PATH = "record_20260203_161353_GUI-PC.bag"

TOPIC_MOCAP = "/mur620c/UR10_r/global_tcp_pose_mocap"
TOPIC_PATH = "/mur620c/ur_path_transformed"

Z_THRESHOLD = 0.9
X_SHIFT = -0.24
Y_SHIFT = -0.22

SMOOTH_WINDOW = 15   # odd number recommended


def moving_average(data, window_size=15):
    """
    Simple centered moving average with edge padding.
    """
    if len(data) == 0:
        return data
    if window_size <= 1:
        return data.copy()

    window_size = min(window_size, len(data))
    if window_size % 2 == 0:
        window_size += 1
        if window_size > len(data):
            window_size -= 1

    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


def extract_first_contiguous_layer_segment(bag_path, topic_mocap, z_threshold=0.87):
    """
    Reads mocap PoseStamped messages and returns only the first contiguous
    segment where z < z_threshold.
    """
    x_vals = []
    y_vals = []
    z_vals = []
    t_vals = []

    in_segment = False
    segment_finished = False

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic_mocap]):
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z

            if z < z_threshold:
                if not segment_finished:
                    in_segment = True
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                    t_vals.append(t.to_sec())
            else:
                if in_segment:
                    # first contiguous z<threshold block is over
                    segment_finished = True
                    in_segment = False

    return np.array(x_vals), np.array(y_vals), np.array(z_vals), np.array(t_vals)


def read_path_xy(bag_path, topic_path):
    """
    Reads the last nav_msgs/Path message from the bag and returns XY points.
    """
    last_path_msg = None

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic_path]):
            last_path_msg = msg

    if last_path_msg is None:
        return np.array([]), np.array([])

    x_path = [p.pose.position.x for p in last_path_msg.poses]
    y_path = [p.pose.position.y for p in last_path_msg.poses]

    return np.array(x_path), np.array(y_path)


def plot_xy_comparison(x_path, y_path,
                       x_mocap_raw, y_mocap_raw,
                       x_mocap_filtered, y_mocap_filtered):
    plt.figure(figsize=(10, 8))

    if len(x_path) > 0:
        plt.plot(x_path, y_path, '-', linewidth=1, label='Sollpfad')

    if len(x_mocap_raw) > 0:
        plt.plot(x_mocap_raw, y_mocap_raw, '.', markersize=1, alpha=0.4,
                 label='Istpfad roh (1. Layer, verschoben)')

    if len(x_mocap_filtered) > 0:
        plt.plot(x_mocap_filtered, y_mocap_filtered, '-', linewidth=2,
                 label='Istpfad geglättet (1. Layer, verschoben)')

        # start marker
        plt.plot(x_mocap_filtered[0], y_mocap_filtered[0], 'o', markersize=8,
                 label='Start Istpfad')

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Sollpfad vs. Istpfad der ersten zusammenhängenden Schicht")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Read first contiguous first-layer segment
    x_mocap, y_mocap, z_mocap, t_mocap = extract_first_contiguous_layer_segment(
        BAG_PATH, TOPIC_MOCAP, Z_THRESHOLD
    )

    # Shift raw mocap path
    x_mocap_shifted = x_mocap + X_SHIFT
    y_mocap_shifted = y_mocap + Y_SHIFT

    # Smooth shifted mocap path
    x_mocap_smooth = moving_average(x_mocap_shifted, SMOOTH_WINDOW)
    y_mocap_smooth = moving_average(y_mocap_shifted, SMOOTH_WINDOW)

    # Read target path
    x_path, y_path = read_path_xy(BAG_PATH, TOPIC_PATH)

    print(f"Anzahl Mocap-Punkte im ersten zusammenhängenden Layer-Segment: {len(x_mocap)}")
    print(f"Anzahl Sollpfad-Punkte: {len(x_path)}")

    if len(x_mocap) == 0:
        print("Kein zusammenhängender erster Layer-Abschnitt mit z < 0.87 gefunden.")

    if len(x_path) == 0:
        print("Kein Sollpfad auf /mur620c/ur_path_transformed gefunden.")

    plot_xy_comparison(
        x_path, y_path,
        x_mocap_shifted, y_mocap_shifted,
        x_mocap_smooth, y_mocap_smooth
    )


if __name__ == "__main__":
    main()