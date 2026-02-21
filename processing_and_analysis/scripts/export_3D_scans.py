#!/usr/bin/env python3
import os
import glob
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2

# ----------------- CONFIG -----------------
scan_topic_pc2 = "/profiles"
output_ply = "scans_export_merged.ply"

# Keep every x-th profile message (1 = keep all, 2 = keep every 2nd, ...)
profile_density = 1

# Keep every x-th point within a message (1 = keep all, 2 = keep every 2nd, ...)
point_density = 50

# Optional: only consider bags matching pattern (default: all .bag files)
bag_glob = "*.bag"

# --- Z-REPAIR (duplicate a z-slab and shift down) ---
enable_z_repair = True
z_copy_min = 0.20   # [m]
z_copy_max = 0.40   # [m]
z_offset = -0.2    # [m] (30 cm down)

# --- Per-bag Z shift (by filename prefix) ---
special_prefix = "record_20260205"
special_z_shift = 0.80  # [m] shift in +z

# -------------------------------------------


def save_ply(filename, points: np.ndarray):
    """Save Nx3 points as ASCII PLY file."""
    if points.size == 0:
        raise RuntimeError("No points to save (points array is empty).")

    N = points.shape[0]
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

    print(f"[OK] Saved PLY file: {filename} ({N} points)")


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

    print(f"[INFO] Found {len(bagfiles)} bag(s) in {cwd}")
    print(f"[INFO] Topic: {scan_topic_pc2}")
    print(f"[INFO] profile_density={profile_density} (keep every {profile_density}th message)")
    print(f"[INFO] point_density={point_density} (keep every {point_density}th point)")

    points_list = []
    total_msgs = 0
    used_msgs = 0

    for bagfile in bagfiles:
        print(f"[INFO] Reading: {os.path.basename(bagfile)}")
        msg_idx = 0

        base = os.path.basename(bagfile)
        bag_z_shift = special_z_shift if base.startswith(special_prefix) else 0.0
        if bag_z_shift != 0.0:
            print(f"[INFO] Applying +{bag_z_shift} m z-shift to: {base}")

        with rosbag.Bag(bagfile, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[scan_topic_pc2]):
                total_msgs += 1
                msg_idx += 1

                # keep only every profile_density-th message
                if (msg_idx - 1) % profile_density != 0:
                    continue

                used_msgs += 1

                # iterate points in PointCloud2
                p_idx = 0
                for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    # keep only every point_density-th point
                    if p_idx % point_density == 0:
                        points_list.append([float(p[0]), float(p[1]), float(p[2]) + bag_z_shift])
                    p_idx += 1

    if not points_list:
        raise RuntimeError(
            "No points collected. Check topic name, densities, and whether bags contain data."
        )

    points = np.asarray(points_list, dtype=np.float32)
    print(f"[INFO] Messages processed: {total_msgs}, messages used: {used_msgs}")
    print(f"[INFO] Total points: {points.shape[0]}")

    # ---- Z-REPAIR: copy z-slab and insert shifted copy ----
    if enable_z_repair:
        z = points[:, 2]
        mask = (z >= z_copy_min) & (z <= z_copy_max)
        slab = points[mask].copy()

        if slab.size == 0:
            print("[WARN] Z-repair enabled, but no points found in the given z-range.")
        else:
            slab[:, 2] += z_offset  # shift in z
            points = np.vstack([points, slab])
            print(f"[INFO] Z-repair: duplicated {slab.shape[0]} points from "
                  f"[{z_copy_min}, {z_copy_max}] m and shifted by {z_offset} m.")
            print(f"[INFO] Total points after Z-repair: {points.shape[0]}")

    print(f"[INFO] Messages processed: {total_msgs}, messages used: {used_msgs}")
    print(f"[INFO] Total points: {points.shape[0]}")

    save_ply(output_ply, points)


if __name__ == "__main__":
    main()