#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap


def extract_points_from_message(msg):
    """
    Returns Nx3 numpy array of points from a profile message.

    Supported:
    - sensor_msgs/PointCloud2
    - custom message with field 'points', where each point has x, y, z
    """
    if hasattr(msg, "_type") and msg._type == "sensor_msgs/PointCloud2":
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    if hasattr(msg, "points"):
        pts = []
        for p in msg.points:
            if hasattr(p, "x") and hasattr(p, "y"):
                z = p.z if hasattr(p, "z") else 0.0
                pts.append([p.x, p.y, z])
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    raise TypeError(
        "Unsupported /profiles message type: {} (ROS type: {}).".format(
            type(msg), getattr(msg, "_type", "unknown")
        )
    )



def compute_profile_center(points_xyz, mode="median"):
    """
    Representative profile center in XY.
    """
    if points_xyz.shape[0] == 0:
        return None

    if mode == "mean":
        cx = np.mean(points_xyz[:, 0])
        cy = np.mean(points_xyz[:, 1])
    else:
        cx = np.median(points_xyz[:, 0])
        cy = np.median(points_xyz[:, 1])

    return np.array([cx, cy], dtype=np.float64)


def compute_profile_max_z(points_xyz):
    if points_xyz.shape[0] == 0:
        return None
    return float(np.max(points_xyz[:, 2]))


def build_layers_from_profiles(profile_entries, jump_threshold=0.10):
    """
    Builds layers from chronologically ordered profiles.

    Input:
        profile_entries: list of dicts with keys
            - t
            - center_xy
            - points_xyz
            - profile_max_z
            - n_points

    Output:
        layers: list of dicts, each with
            - layer_idx
            - profiles
            - points_xyz
            - highest_point
            - profile_maxima_z
            - median_profile_max_z
            - layer_max_z
            - deviation
    """
    if not profile_entries:
        return []

    raw_layers = []
    current_profiles = [profile_entries[0]]

    for i in range(1, len(profile_entries)):
        prev_c = profile_entries[i - 1]["center_xy"]
        curr_c = profile_entries[i]["center_xy"]

        dist = np.linalg.norm(curr_c - prev_c)

        if dist > jump_threshold:
            raw_layers.append(current_profiles)
            current_profiles = [profile_entries[i]]
        else:
            current_profiles.append(profile_entries[i])

    if current_profiles:
        raw_layers.append(current_profiles)

    layers = []

    for layer_idx, profiles in enumerate(raw_layers, start=1):
        pts_list = [p["points_xyz"] for p in profiles if p["points_xyz"].shape[0] > 0]

        if len(pts_list) == 0:
            all_pts = np.empty((0, 3), dtype=np.float64)
            highest_point = None
            profile_maxima_z = np.array([], dtype=np.float64)
            median_profile_max_z = np.nan
            layer_max_z = np.nan
            deviation = np.nan
        else:
            all_pts = np.vstack(pts_list)

            max_idx = np.argmax(all_pts[:, 2])
            highest_point = all_pts[max_idx]

            profile_maxima_z = np.array(
                [p["profile_max_z"] for p in profiles if p["profile_max_z"] is not None],
                dtype=np.float64
            )

            if len(profile_maxima_z) > 0:
                median_profile_max_z = float(np.median(profile_maxima_z))
                layer_max_z = float(np.max(profile_maxima_z))
                deviation = layer_max_z - median_profile_max_z
            else:
                median_profile_max_z = np.nan
                layer_max_z = np.nan
                deviation = np.nan

        layers.append({
            "layer_idx": layer_idx,
            "profiles": profiles,
            "points_xyz": all_pts,
            "highest_point": highest_point,
            "profile_maxima_z": profile_maxima_z,
            "median_profile_max_z": median_profile_max_z,
            "layer_max_z": layer_max_z,
            "deviation": deviation,
        })

    # remove layers with fewer than 20 points
    #layers = [layer for layer in layers if len(layer["profiles"]) >= 20]


    return layers


def save_layer_deviation_plot(layers, output_pdf):
    layer_ids = [layer["layer_idx"] for layer in layers if not np.isnan(layer["deviation"])]
    deviations = [layer["deviation"] for layer in layers if not np.isnan(layer["deviation"])]

    plt.figure(figsize=(10, 5))
    plt.plot(layer_ids, deviations, marker='o')
    plt.xlabel("Layer index")
    plt.ylabel("Layer max - layer median [m]")
    plt.title("Deviation of highest profile maximum from median per layer")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    plt.close()


def save_layer_csv(layers, output_csv):
    with open(output_csv, "w") as f:
        f.write("layer_idx,n_profiles,n_points,median_profile_max_z,layer_max_z,deviation,highest_x,highest_y,highest_z\n")
        for layer in layers:
            hp = layer["highest_point"]
            if hp is None:
                hx, hy, hz = np.nan, np.nan, np.nan
            else:
                hx, hy, hz = hp[0], hp[1], hp[2]

            f.write("{},{},{},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f}\n".format(
                layer["layer_idx"],
                len(layer["profiles"]),
                layer["points_xyz"].shape[0],
                layer["median_profile_max_z"],
                layer["layer_max_z"],
                layer["deviation"],
                hx, hy, hz
            ))

def filter_profile_outliers_z(points_xyz, mad_scale=3.5):
    """
    Remove vertical outliers in one profile using a robust MAD-based filter.

    Parameters:
        points_xyz: Nx3 numpy array
        mad_scale: threshold factor, typical values 2.5 ... 5.0

    Returns:
        filtered_points_xyz: Mx3 numpy array
    """
    if points_xyz.shape[0] == 0:
        return points_xyz

    z = points_xyz[:, 2]
    z_med = np.median(z)

    abs_dev = np.abs(z - z_med)
    mad = np.median(abs_dev)

    # fallback if profile is almost perfectly flat
    if mad < 1e-12:
        return points_xyz.copy()

    robust_z_score = abs_dev / (1.4826 * mad)
    mask = robust_z_score <= mad_scale

    filtered = points_xyz[mask]

    # avoid returning empty profile
    if filtered.shape[0] == 0:
        return points_xyz.copy()

    return filtered

from matplotlib.colors import LinearSegmentedColormap


def plot_layers_3d(layers,
                   output_png="layers_3d.png",
                   output_pdf="layers_3d.pdf",
                   point_stride=50,
                   elev=25,
                   azim=-60):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Custom color map (dark gray -> gray -> green)
    colors = ["#696969", "#d1d1d1" , "#b1e629"]
    cmap = LinearSegmentedColormap.from_list("layer_cmap", colors)

    n_layers = 17

    color_index = 1
    print("number of layers: {}".format(len(layers)))
    for i, layer in enumerate(layers):

        pts = layer["points_xyz"]
        if pts.shape[0] == 0:
            continue

        # plot every Nth point
        pts_plot = pts[::point_stride]

        color = cmap(color_index / max(1, n_layers - 1))
        
        color_index += 1
        ax.scatter(
            pts_plot[:, 0],
            pts_plot[:, 1],
            pts_plot[:, 2],
            s=2,
            color=color,
            alpha=0.9
        )

        # mark highest point
        hp = layer["highest_point"]
        if hp is not None:
            pass
            # ax.scatter(
            #     hp[0], hp[1], hp[2],
            #     s=120,
            #     color="red",
            #     marker="x",
            #     linewidths=2
            # )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D view of detected layers")

    ax.view_init(elev=elev, azim=azim)

    # Equal scaling
    all_nonempty = [layer["points_xyz"] for layer in layers if layer["points_xyz"].shape[0] > 0]

    if all_nonempty:
        all_pts = np.vstack(all_nonempty)

        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)

        max_range = np.max(maxs - mins)
        mids = 0.5 * (mins + maxs)

        ax.set_xlim(mids[0] - max_range/2, mids[0] + max_range/2)
        ax.set_ylim(mids[1] - max_range/2, mids[1] + max_range/2)
        ax.set_zlim(mids[2] - max_range/2, mids[2] + max_range/2)

    plt.tight_layout()
    # plt.savefig(output_png, dpi=300)
    # plt.savefig(output_pdf, format="pdf")
    # plt.close()
    plt.show()

def plot_deviation_vs_layer(layers, output_pdf="deviation_vs_layer.pdf"):
    """
    Plot deviation vs layer index.
    """

    layer_ids = []
    deviations = []

    for layer in layers:
        if not np.isnan(layer["deviation"]):
            if layer["layer_idx"] < 11 or layer["layer_idx"] > 13:
                layer_ids.append(layer["layer_idx"])
                deviations.append(layer["deviation"])

    plt.figure(figsize=(10,5))

    plt.plot(
        layer_ids,
        deviations,
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Layer number")
    plt.ylabel("Deviation [m]")
    plt.title("Layer height control performance")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Split /profiles into layers based on center jumps and evaluate layer maxima."
    )
    parser.add_argument("--bag", type=str, default="profiles_roi_merged.bag")
    parser.add_argument("--topic", type=str, default="/profiles")
    parser.add_argument("--jump-threshold", type=float, default=0.20,
                        help="Distance threshold in meters for detecting a new layer")
    parser.add_argument("--center-mode", type=str, default="median", choices=["median", "mean"],
                        help="How to compute representative profile center")
    parser.add_argument("--deviation-plot", type=str, default="layer_max_deviation.pdf")
    parser.add_argument("--csv", type=str, default="layer_max_deviation.csv")
    parser.add_argument("--layers3d-png", type=str, default="layers_3d.png")
    parser.add_argument("--layers3d-pdf", type=str, default="layers_3d.pdf")
    parser.add_argument("--max-points-per-layer", type=int, default=15000)
    parser.add_argument("--outlier-mad-scale", type=float, default=3.5,
                    help="MAD threshold for removing z-outliers in each profile")

    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        raise FileNotFoundError("Bag not found: {}".format(args.bag))

    profile_entries = []

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[args.topic]):
            pts = extract_points_from_message(msg)
            if pts.shape[0] == 0:
                continue

            pts_filtered = filter_profile_outliers_z(pts, mad_scale=args.outlier_mad_scale)
            if pts_filtered.shape[0] == 0:
                continue

            center_xy = compute_profile_center(pts_filtered, mode=args.center_mode)
            max_z = compute_profile_max_z(pts_filtered)

            profile_entries.append({
                "t": t.to_sec(),
                "center_xy": center_xy,
                "points_xyz": pts_filtered,      # bereinigte Punkte speichern
                "profile_max_z": max_z,
                "n_points": pts_filtered.shape[0],
            })

    if len(profile_entries) == 0:
        raise RuntimeError("No valid profiles found in topic {}".format(args.topic))

    layers = build_layers_from_profiles(
        profile_entries,
        jump_threshold=args.jump_threshold
    )

    if len(layers) == 0:
        raise RuntimeError("No layers found.")

    save_layer_deviation_plot(layers, args.deviation_plot)
    save_layer_csv(layers, args.csv)
    plot_layers_3d(
        layers,
        output_png=args.layers3d_png,
        output_pdf=args.layers3d_pdf,
    )

    plot_deviation_vs_layer(
        layers,
        output_pdf="layer_height_deviation.pdf"
    )

    print("Done.")
    print("Profiles processed: {}".format(len(profile_entries)))
    print("Layers detected:    {}".format(len(layers)))
    print("Deviation plot:     {}".format(args.deviation_plot))
    print("CSV saved to:       {}".format(args.csv))
    print("3D PNG saved to:    {}".format(args.layers3d_png))
    print("3D PDF saved to:    {}".format(args.layers3d_pdf))
    print("")

    for layer in layers:
        print(
            "Layer {:3d}: profiles={}, points={}, median_profile_max_z={:.4f}, layer_max_z={:.4f}, deviation={:.4f}".format(
                layer["layer_idx"],
                len(layer["profiles"]),
                layer["points_xyz"].shape[0],
                layer["median_profile_max_z"],
                layer["layer_max_z"],
                layer["deviation"]
            )
        )


if __name__ == "__main__":
    main()