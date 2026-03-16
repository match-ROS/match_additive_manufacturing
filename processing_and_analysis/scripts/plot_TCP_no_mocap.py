#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs import point_cloud2


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BAG_MOCAP = "record_20260203_161353_GUI-PC.bag"
BAG_PROFILES = "record_20260203_161353_MuR.bag"

TOPIC_MOCAP = "/mur620c/UR10_r/global_tcp_pose_mocap"
TOPIC_PROFILES = "/profiles"

Z_THRESHOLD_MOCAP = 0.87

SMOOTH_WINDOW_MOCAP = 15      # smoothing for measured path
SMOOTH_WINDOW_REF = 9         # light smoothing for reconstructed reference

# Outlier removal for profile point clouds
NEIGHBOR_RADIUS = 0.005       # 5 mm
MIN_NEIGHBORS = 8
MIN_POINTS_AFTER_FILTER = 5

# Layer end detection from reconstructed maxima
START_RETURN_THRESHOLD = 0.05   # 5 cm
MUST_BE_AWAY_THRESHOLD = 0.10   # 10 cm


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def moving_average(data, window_size=15):
    """Centered moving average with edge padding."""
    if len(data) == 0:
        return data
    if window_size <= 1:
        return data.copy()

    window_size = min(window_size, len(data))
    if window_size < 3:
        return data.copy()

    if window_size % 2 == 0:
        window_size += 1
        if window_size > len(data):
            window_size -= 1

    if window_size < 3:
        return data.copy()

    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode='valid')


def pairwise_xy_neighbor_filter(points_xyz, radius=0.005, min_neighbors=8):
    """
    Removes isolated outliers by checking how many neighbors each point has
    within a given XY radius.
    """
    if len(points_xyz) == 0:
        return points_xyz

    xy = points_xyz[:, :2]
    keep_mask = np.zeros(len(points_xyz), dtype=bool)

    for i in range(len(points_xyz)):
        d2 = np.sum((xy - xy[i]) ** 2, axis=1)
        neighbor_count = np.count_nonzero(d2 <= radius * radius) - 1
        if neighbor_count >= min_neighbors:
            keep_mask[i] = True

    return points_xyz[keep_mask]


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
                    segment_finished = True
                    in_segment = False

    return np.array(x_vals), np.array(y_vals), np.array(z_vals), np.array(t_vals)


def read_xyz_from_pointcloud2(msg):
    """Convert sensor_msgs/PointCloud2 to Nx3 numpy array."""
    pts = []
    for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    arr = np.asarray(pts, dtype=np.float64)
    finite_mask = np.isfinite(arr).all(axis=1)
    return arr[finite_mask]


def get_profile_maximum(points_xyz,
                        neighbor_radius=0.005,
                        min_neighbors=8,
                        min_points_after_filter=5):
    """
    Remove outliers and return highest point in filtered profile.
    """
    if len(points_xyz) == 0:
        return None, points_xyz

    filtered = pairwise_xy_neighbor_filter(
        points_xyz,
        radius=neighbor_radius,
        min_neighbors=min_neighbors
    )

    if len(filtered) < min_points_after_filter:
        return None, filtered

    idx_max = np.argmax(filtered[:, 2])
    max_point = filtered[idx_max]
    return max_point, filtered


def reconstruct_reference_from_profiles(bag_path,
                                        topic_profiles,
                                        start_return_threshold=0.05,
                                        must_be_away_threshold=0.10,
                                        neighbor_radius=0.005,
                                        min_neighbors=8,
                                        min_points_after_filter=5):
    """
    Reconstruct reference trajectory from highest point of each filtered profile.
    """
    ref_x = []
    ref_y = []
    ref_z = []
    ref_t = []

    first_max_xy = None
    was_far_away = False
    layer_finished = False

    num_profiles_total = 0
    num_profiles_valid = 0
    num_profiles_invalid = 0

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic_profiles]):
            num_profiles_total += 1

            pts = read_xyz_from_pointcloud2(msg)
            max_point, _ = get_profile_maximum(
                pts,
                neighbor_radius=neighbor_radius,
                min_neighbors=min_neighbors,
                min_points_after_filter=min_points_after_filter
            )

            if max_point is None:
                num_profiles_invalid += 1
                continue

            num_profiles_valid += 1

            x_max, y_max, z_max = max_point

            if first_max_xy is None:
                first_max_xy = np.array([x_max, y_max], dtype=np.float64)

            current_xy = np.array([x_max, y_max], dtype=np.float64)
            dist_to_start = np.linalg.norm(current_xy - first_max_xy)

            if dist_to_start > must_be_away_threshold:
                was_far_away = True

            if was_far_away and dist_to_start < start_return_threshold and len(ref_x) > 10:
                layer_finished = True
                break

            ref_x.append(x_max)
            ref_y.append(y_max)
            ref_z.append(z_max)
            ref_t.append(t.to_sec())

    return {
        "x": np.array(ref_x),
        "y": np.array(ref_y),
        "z": np.array(ref_z),
        "t": np.array(ref_t),
        "num_profiles_total": num_profiles_total,
        "num_profiles_valid": num_profiles_valid,
        "num_profiles_invalid": num_profiles_invalid,
        "layer_finished": layer_finished,
    }


def align_path_start(x_path, y_path, x_ref_start, y_ref_start):
    """
    Shift path so that its first point coincides with reference start.
    """
    if len(x_path) == 0:
        return x_path, y_path, 0.0, 0.0

    dx = x_ref_start - x_path[0]
    dy = y_ref_start - y_path[0]

    return x_path + dx, y_path + dy, dx, dy


def plot_xy_comparison(x_ref_raw, y_ref_raw,
                       x_ref_smooth, y_ref_smooth,
                       x_mocap_raw, y_mocap_raw,
                       x_mocap_smooth, y_mocap_smooth):
    plt.figure(figsize=(10, 8))
    PLOT_STRIDE = 10
    # if len(x_ref_raw) > 0:
    #     plt.plot(x_ref_raw, y_ref_raw, '.', markersize=2, alpha=0.25,
    #              label='Referenz roh')

    if len(x_ref_smooth) > 0:
        plt.plot(x_ref_smooth[::PLOT_STRIDE], y_ref_smooth[::PLOT_STRIDE], '-', linewidth=2.2,
                 label='Referenz geglättet')
        plt.plot(x_ref_smooth[0], y_ref_smooth[0], 'o', markersize=8,
                 label='Start Referenz')

    # if len(x_mocap_raw) > 0:
    #     plt.plot(x_mocap_raw, y_mocap_raw, '.', markersize=2, alpha=0.35,
    #              label='Istpfad roh')

    if len(x_mocap_smooth) > 0:
        plt.plot(x_mocap_smooth[::PLOT_STRIDE], y_mocap_smooth[::PLOT_STRIDE], '-', linewidth=2,
                 label='Istpfad geglättet')
        plt.plot(x_mocap_smooth[0], y_mocap_smooth[0], 'o', markersize=8,
                 label='Start Istpfad')

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Istpfad der ersten Lage vs. Referenz aus vorheriger Schicht")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # ------------------------------------------------------------
    # 1) Ist-Pfad aus Mocap
    # ------------------------------------------------------------
    x_mocap, y_mocap, z_mocap, t_mocap = extract_first_contiguous_layer_segment(
        BAG_MOCAP, TOPIC_MOCAP, Z_THRESHOLD_MOCAP
    )

    x_mocap_smooth = moving_average(x_mocap, SMOOTH_WINDOW_MOCAP)
    y_mocap_smooth = moving_average(y_mocap, SMOOTH_WINDOW_MOCAP)

    # ------------------------------------------------------------
    # 2) Referenz aus Profil-Maxima
    # ------------------------------------------------------------
    ref = reconstruct_reference_from_profiles(
        BAG_PROFILES,
        TOPIC_PROFILES,
        start_return_threshold=START_RETURN_THRESHOLD,
        must_be_away_threshold=MUST_BE_AWAY_THRESHOLD,
        neighbor_radius=NEIGHBOR_RADIUS,
        min_neighbors=MIN_NEIGHBORS,
        min_points_after_filter=MIN_POINTS_AFTER_FILTER
    )

    x_ref_raw = ref["x"]
    y_ref_raw = ref["y"]

    # leichte XY-Glättung der Referenz
    x_ref_smooth = moving_average(x_ref_raw, SMOOTH_WINDOW_REF)
    y_ref_smooth = moving_average(y_ref_raw, SMOOTH_WINDOW_REF)

    # ------------------------------------------------------------
    # 3) Ist-Pfad auf Referenz-Start ausrichten
    # ------------------------------------------------------------
    if len(x_ref_smooth) > 0 and len(x_mocap) > 0:
        x_mocap_aligned, y_mocap_aligned, dx_align, dy_align = align_path_start(
            x_mocap, y_mocap, x_ref_smooth[0], y_ref_smooth[0]
        )
        x_mocap_smooth_aligned, y_mocap_smooth_aligned, _, _ = align_path_start(
            x_mocap_smooth, y_mocap_smooth, x_ref_smooth[0], y_ref_smooth[0]
        )
    else:
        x_mocap_aligned, y_mocap_aligned = x_mocap, y_mocap
        x_mocap_smooth_aligned, y_mocap_smooth_aligned = x_mocap_smooth, y_mocap_smooth
        dx_align, dy_align = 0.0, 0.0

    # ------------------------------------------------------------
    # 4) Output
    # ------------------------------------------------------------
    print("----- Mocap -----")
    print("Number of mocap points in first contiguous layer segment: {}".format(len(x_mocap)))

    print("\n----- Profiles -----")
    print("Total profiles read: {}".format(ref["num_profiles_total"]))
    print("Valid profiles used: {}".format(ref["num_profiles_valid"]))
    print("Invalid/skipped profiles: {}".format(ref["num_profiles_invalid"]))
    print("Reference points reconstructed: {}".format(len(x_ref_raw)))
    print("Layer closure detected: {}".format(ref["layer_finished"]))

    print("\n----- Alignment -----")
    print("Applied start alignment shift:")
    print("dx = {:.6f} m".format(dx_align))
    print("dy = {:.6f} m".format(dy_align))

    # ------------------------------------------------------------
    # 5) Plot
    # ------------------------------------------------------------
    plot_xy_comparison(
        x_ref_raw, y_ref_raw,
        x_ref_smooth, y_ref_smooth,
        x_mocap_aligned, y_mocap_aligned,
        x_mocap_smooth_aligned, y_mocap_smooth_aligned
    )


if __name__ == "__main__":
    main()