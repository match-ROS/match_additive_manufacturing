#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import math
import warnings

import numpy as np

import rosbag

# Optional: nicer smoothing if scipy is available
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Optional: faster nearest-neighbor search if scipy is available
try:
    from scipy.spatial import cKDTree
    CKDTREE_AVAILABLE = True
except Exception:
    CKDTREE_AVAILABLE = False


TCP_TOPIC = "/mur620c/UR10_r/global_tcp_pose_mocap"
CMD_VEL_TOPIC = "/mur620c/cmd_vel"
MIR_POSE_TOPIC = "/qualisys_map/mur620c/pose"
PATH_TOPIC = "/mur620c/ur_path_transformed"


def safe_to_sec(t):
    try:
        return t.to_sec()
    except Exception:
        return float(t)


def moving_average(data, window_size=11):
    if len(data) < 3:
        return data.copy()
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if len(data) < window_size:
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
    if window_size < 3:
        return data.copy()

    kernel = np.ones(window_size) / float(window_size)
    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_signal(data, preferred_window=21, polyorder=3):
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 3:
        return data.copy()

    window = min(preferred_window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return data.copy()
    if window <= polyorder:
        polyorder = max(1, window - 1)

    if SCIPY_AVAILABLE and window >= 5 and polyorder >= 1:
        try:
            return savgol_filter(data, window_length=window, polyorder=polyorder, mode="interp")
        except Exception:
            pass

    return moving_average(data, window_size=window)


def remove_duplicate_times(t, *arrays):
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return (t,) + tuple(np.asarray(a) for a in arrays)

    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9

    result = [t[keep]]
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[keep])
    return tuple(result)


def compute_xy_speed_from_pose_timeseries(times, x, y, smoothing_window=21):
    times = np.asarray(times, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(times) < 2:
        return {
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "speed_series": np.array([]),
            "time_series": np.array([]),
        }

    times, x, y = remove_duplicate_times(times, x, y)
    if len(times) < 2:
        return {
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "speed_series": np.array([]),
            "time_series": np.array([]),
        }

    x_s = smooth_signal(x, preferred_window=smoothing_window)
    y_s = smooth_signal(y, preferred_window=smoothing_window)

    dx_dt = np.gradient(x_s, times)
    dy_dt = np.gradient(y_s, times)
    speed = np.sqrt(dx_dt**2 + dy_dt**2)

    return {
        "speed_mean": float(np.mean(speed)),
        "speed_max": float(np.max(speed)),
        "speed_series": speed,
        "time_series": times,
    }


def compute_mir_cmd_vel_stats(times, vx, vy=None):
    times = np.asarray(times, dtype=float)
    vx = np.asarray(vx, dtype=float)

    if vy is None:
        vy = np.zeros_like(vx)
    else:
        vy = np.asarray(vy, dtype=float)

    if len(vx) == 0:
        return {
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "speed_series": np.array([]),
        }

    speed = np.sqrt(vx**2 + vy**2)

    return {
        "speed_mean": float(np.mean(speed)),
        "speed_max": float(np.max(speed)),
        "speed_series": speed,
    }


def compute_distance_from_pose(times, x, y):
    times = np.asarray(times, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(times) < 2:
        return np.nan

    times, x, y = remove_duplicate_times(times, x, y)
    if len(times) < 2:
        return np.nan

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    return float(np.sum(ds))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))

    cx = ax + t * abx
    cy = ay + t * aby

    return math.hypot(px - cx, py - cy)


def distances_to_polyline(points_xy, polyline_xy):
    """
    Computes distance from each point to the polyline defined by polyline_xy.
    If scipy KDTree is available, a quick preselection of nearby vertices is used.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    polyline_xy = np.asarray(polyline_xy, dtype=float)

    if len(points_xy) == 0 or len(polyline_xy) == 0:
        return np.array([])

    if len(polyline_xy) == 1:
        return np.sqrt(np.sum((points_xy - polyline_xy[0]) ** 2, axis=1))

    distances = []

    if CKDTREE_AVAILABLE and len(polyline_xy) > 10:
        tree = cKDTree(polyline_xy)
        _, idxs = tree.query(points_xy, k=1)
        for (px, py), idx in zip(points_xy, idxs):
            i0 = max(0, idx - 2)
            i1 = min(len(polyline_xy) - 2, idx + 1)

            best_d = float("inf")
            for i in range(i0, i1 + 1):
                ax, ay = polyline_xy[i]
                bx, by = polyline_xy[i + 1]
                d = point_to_segment_distance(px, py, ax, ay, bx, by)
                if d < best_d:
                    best_d = d
            distances.append(best_d)
    else:
        for px, py in points_xy:
            best_d = float("inf")
            for i in range(len(polyline_xy) - 1):
                ax, ay = polyline_xy[i]
                bx, by = polyline_xy[i + 1]
                d = point_to_segment_distance(px, py, ax, ay, bx, by)
                if d < best_d:
                    best_d = d
            distances.append(best_d)

    return np.asarray(distances, dtype=float)


def compute_path_error(actual_xy, target_xy):
    if len(actual_xy) == 0 or len(target_xy) == 0:
        return {
            "error_mean": np.nan,
            "error_std": np.nan,
            "error_max": np.nan,
            "errors": np.array([]),
        }

    errors = distances_to_polyline(actual_xy, target_xy)

    if len(errors) == 0:
        return {
            "error_mean": np.nan,
            "error_std": np.nan,
            "error_max": np.nan,
            "errors": np.array([]),
        }

    return {
        "error_mean": float(np.mean(abs((errors)))),
        "error_std": float(np.std(errors)),
        "error_max": float(np.max(errors)),
        "errors": errors,
    }


def read_bag_data(bag_path):
    tcp_t, tcp_x, tcp_y = [], [], []
    mir_pose_t, mir_pose_x, mir_pose_y = [], [], []
    cmd_t, cmd_vx, cmd_vy = [], [], []

    target_xy = []

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[TCP_TOPIC, CMD_VEL_TOPIC, MIR_POSE_TOPIC, PATH_TOPIC]):
            if topic == TCP_TOPIC:
                ts = safe_to_sec(msg.header.stamp) if msg.header.stamp.to_sec() > 0 else safe_to_sec(t)
                tcp_t.append(ts)
                tcp_x.append(msg.pose.position.x)
                tcp_y.append(msg.pose.position.y)

            elif topic == MIR_POSE_TOPIC:
                ts = safe_to_sec(msg.header.stamp) if msg.header.stamp.to_sec() > 0 else safe_to_sec(t)
                mir_pose_t.append(ts)
                mir_pose_x.append(msg.pose.position.x)
                mir_pose_y.append(msg.pose.position.y)

            elif topic == CMD_VEL_TOPIC:
                ts = safe_to_sec(t)
                cmd_t.append(ts)
                # translational speed in XY
                vx = getattr(msg.linear, "x", 0.0)
                vy = getattr(msg.linear, "y", 0.0)
                cmd_vx.append(vx)
                cmd_vy.append(vy)

            elif topic == PATH_TOPIC:
                msg_type = msg._type

                if msg_type == "nav_msgs/Path":
                    current_xy = []
                    for p in msg.poses:
                        current_xy.append([p.pose.position.x, p.pose.position.y])
                    # keep latest non-empty path
                    if len(current_xy) > 0:
                        target_xy = current_xy

                elif msg_type == "geometry_msgs/PoseStamped":
                    target_xy.append([msg.pose.position.x, msg.pose.position.y])

                else:
                    warnings.warn(
                        f"Unsupported message type on {PATH_TOPIC} in {bag_path}: {msg_type}"
                    )

    return {
        "tcp_t": np.asarray(tcp_t, dtype=float),
        "tcp_x": np.asarray(tcp_x, dtype=float),
        "tcp_y": np.asarray(tcp_y, dtype=float),
        "mir_pose_t": np.asarray(mir_pose_t, dtype=float),
        "mir_pose_x": np.asarray(mir_pose_x, dtype=float),
        "mir_pose_y": np.asarray(mir_pose_y, dtype=float),
        "cmd_t": np.asarray(cmd_t, dtype=float),
        "cmd_vx": np.asarray(cmd_vx, dtype=float),
        "cmd_vy": np.asarray(cmd_vy, dtype=float),
        "target_xy": np.asarray(target_xy, dtype=float),
    }


def analyze_bag(bag_path, smoothing_window=21):
    data = read_bag_data(bag_path)

    tcp_stats = compute_xy_speed_from_pose_timeseries(
        data["tcp_t"], data["tcp_x"], data["tcp_y"], smoothing_window=smoothing_window
    )

    mir_cmd_stats = compute_mir_cmd_vel_stats(
        data["cmd_t"], data["cmd_vx"], data["cmd_vy"]
    )

    mir_distance = compute_distance_from_pose(
        data["mir_pose_t"], data["mir_pose_x"], data["mir_pose_y"]
    )

    actual_xy = np.column_stack((data["tcp_x"], data["tcp_y"])) if len(data["tcp_x"]) > 0 else np.empty((0, 2))
    path_error = compute_path_error(actual_xy, data["target_xy"])

    return {
        "bag_name": os.path.basename(bag_path),
        "tcp_speed_mean_mps": tcp_stats["speed_mean"],
        "tcp_speed_max_mps": tcp_stats["speed_max"],
        "mir_speed_mean_mps": mir_cmd_stats["speed_mean"],
        "mir_speed_max_mps": mir_cmd_stats["speed_max"],
        "mir_distance_m": mir_distance,
        "path_error_mean_m": path_error["error_mean"],
        "path_error_std_m": path_error["error_std"],
        "path_error_max_m": path_error["error_max"],
        "num_tcp_samples": len(data["tcp_t"]),
        "num_cmd_vel_samples": len(data["cmd_t"]),
        "num_mir_pose_samples": len(data["mir_pose_t"]),
        "num_target_points": len(data["target_xy"]),
    }


def format_value(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.6f}"
    return str(v)


def main():
    bag_files = sorted(glob.glob("*.bag"))

    if not bag_files:
        print("Keine .bag-Dateien im aktuellen Ordner gefunden.")
        return

    results = []

    print(f"Gefundene Bags: {len(bag_files)}")
    print("-" * 100)

    for bag_file in bag_files:
        try:
            res = analyze_bag(bag_file, smoothing_window=21)
            results.append(res)

            print(f"Bag: {res['bag_name']}")
            print(f"  TCP mean speed [m/s]:   {format_value(res['tcp_speed_mean_mps'])}")
            print(f"  TCP max speed [m/s]:    {format_value(res['tcp_speed_max_mps'])}")
            print(f"  MiR mean speed [m/s]:   {format_value(res['mir_speed_mean_mps'])}")
            print(f"  MiR max speed [m/s]:    {format_value(res['mir_speed_max_mps'])}")
            print(f"  MiR distance [m]:       {format_value(res['mir_distance_m'])}")
            print(f"  Path error mean [m]:    {format_value(res['path_error_mean_m'])}")
            print(f"  Path error std [m]:     {format_value(res['path_error_std_m'])}")
            print(f"  Path error max [m]:     {format_value(res['path_error_max_m'])}")
            print(f"  TCP samples:            {res['num_tcp_samples']}")
            print(f"  cmd_vel samples:        {res['num_cmd_vel_samples']}")
            print(f"  MiR pose samples:       {res['num_mir_pose_samples']}")
            print(f"  Target points:          {res['num_target_points']}")
            print("-" * 100)

        except Exception as e:
            print(f"Fehler bei {bag_file}: {e}")
            print("-" * 100)

    if results:
        csv_path = "rosbag_summary_metrics.csv"
        fieldnames = list(results[0].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"CSV geschrieben: {csv_path}")


if __name__ == "__main__":
    main()