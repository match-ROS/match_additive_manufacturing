#!/usr/bin/env python3
import os
import glob
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import csv
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
scan_topic_pc2 = "/profiles"

# Keep every x-th profile message (1 = keep all, 2 = keep every 2nd, ...)
profile_density = 1

# Keep every x-th point within a message (1 = keep all, 2 = keep every 2nd, ...)
point_density = 5
min_cluster_points = 10      # how many points must be near the max point (incl. itself)
cluster_radius_m = 0.02      # [m] radius around max point

# Optional: only consider bags matching pattern (default: all .bag files)
bag_glob = "*.bag"

# --- Per-bag Z shift (by filename prefix) ---
special_prefix = "record_20260205"
special_z_shift = 0.80  # [m] shift in +z

# --- ROI trigger box (if ANY point of profile is inside -> take profile max) ---
x_min = 48.8
x_max = 49.3
y_min = 43.5
y_max = 44.0

# Outputs
output_csv = "profile_maxima.csv"
output_plot = "profile_maxima_xz.pdf"
# -------------------------------------------


def validate_density(name: str, x: int):
    if not isinstance(x, int) or x < 1:
        raise ValueError(f"{name} must be an integer >= 1 (got {x}).")


def main():
    validate_density("profile_density", profile_density)
    validate_density("point_density", point_density)

    cwd = os.getcwd()
    bagfiles = sorted(glob.glob(os.path.join(cwd, bag_glob)))
    if not bagfiles:
        raise FileNotFoundError(f"No bag files found in {cwd} matching '{bag_glob}'")

    maxima = []  # rows: [bag, msg_idx_used, stamp, x, y, z]

    total_msgs = 0
    used_msgs = 0

    print(f"[INFO] Found {len(bagfiles)} bag(s) in {cwd}")
    print(f"[INFO] Topic: {scan_topic_pc2}")
    print(f"[INFO] profile_density={profile_density}, point_density={point_density}")
    print(f"[INFO] ROI: x[{x_min},{x_max}], y[{y_min},{y_max}]")

    for bagfile in bagfiles:
        base = os.path.basename(bagfile)
        bag_z_shift = special_z_shift if base.startswith(special_prefix) else 0.0
        if bag_z_shift != 0.0:
            print(f"[INFO] Applying +{bag_z_shift} m z-shift to: {base}")

        msg_idx = 0
        used_idx_in_bag = 0

        with rosbag.Bag(bagfile, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[scan_topic_pc2]):
                total_msgs += 1
                msg_idx += 1

                # keep only every profile_density-th message
                if (msg_idx - 1) % profile_density != 0:
                    continue

                used_msgs += 1
                used_idx_in_bag += 1

                # Collect current profile points (downsampled)
                profile_pts = []
                roi_hit = False

                p_idx = 0
                for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    if p_idx % point_density != 0:
                        p_idx += 1
                        continue

                    x, y, z = float(p[0]), float(p[1]), float(p[2]) + bag_z_shift
                    profile_pts.append((x, y, z))

                    if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                        roi_hit = True

                    p_idx += 1

                if not profile_pts:
                    continue

                if roi_hit:
                    # "Maximum des Profils" = Punkt mit größtem z
                    xM, yM, zM = max(profile_pts, key=lambda q: q[2])

                    # --- validity check: enough nearby points around (xM,yM,zM) ---
                    pts_np = np.asarray(profile_pts, dtype=np.float64)  # shape (N,3)
                    d2 = np.sum((pts_np - np.array([xM, yM, zM]))**2, axis=1)
                    neighbor_count = int(np.count_nonzero(d2 <= (cluster_radius_m**2)))

                    if neighbor_count >= min_cluster_points:
                        stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else float(t.to_sec())
                        maxima.append([base, used_idx_in_bag, stamp, xM, yM, zM])

    if not maxima:
        raise RuntimeError("No maxima collected. Either ROI never hit, or no points read.")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bag", "profile_idx_used", "stamp_s", "x", "y", "z"])
        w.writerows(maxima)

    print(f"[OK] Wrote CSV: {output_csv} ({len(maxima)} maxima)")

    # Plot x vs z
    xs = [r[3] for r in maxima]
    zs = [r[5] for r in maxima]

    plt.figure()
    plt.scatter(xs, zs, s=8)   # no lines
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Profile maxima trajectory (x-z)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    print(f"[OK] Saved plot: {output_plot}")


if __name__ == "__main__":
    main()