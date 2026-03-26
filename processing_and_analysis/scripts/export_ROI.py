#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import traceback

import rosbag
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


# ============================================================
# Configuration
# ============================================================

PROFILE_TOPIC = "/profiles"
OUTPUT_BAG = "profiles_roi_merged.bag"
OUTPUT_PLY = "profiles_roi_points.ply"

# --- ROI trigger box (used exactly as provided) ---
x_min = 50.595 - 1.37 
x_max = 50.595 + 1.37
y_min = 43.686 - 1.506 
y_max = 43.686 + 1.506 

# ============================================================
# Helper functions
# ============================================================

def point_in_roi(x, y):
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def extract_points_from_message(msg):
    """
    Returns Nx3 numpy array of points from a profile message.

    Supported:
    - sensor_msgs/PointCloud2
    - custom message with field 'points', where each point has x, y, z
    """
    # Case 1: PointCloud2
    if hasattr(msg, "_type") and msg._type == "sensor_msgs/PointCloud2":
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    # Case 2: Custom msg with msg.points
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
        "Unsupported /profiles message type: {} (ROS type: {}). "
        "Currently supported: sensor_msgs/PointCloud2 or custom msg with msg.points[].x/y/z".format(
            type(msg), getattr(msg, "_type", "unknown")
        )
    )

def write_ascii_ply(points_xyz, output_path):
    """
    Writes Nx3 numpy array to ASCII PLY.
    """
    n = len(points_xyz)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(n))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write("{:.9f} {:.9f} {:.9f}\n".format(p[0], p[1], p[2]))


# ============================================================
# Main
# ============================================================

def main():
    bag_files = sorted(glob.glob("*.bag"))

    # Avoid re-reading output bag if script is run multiple times
    bag_files = [b for b in bag_files if os.path.basename(b) != OUTPUT_BAG]

    if not bag_files:
        print("No .bag files found in current directory.")
        sys.exit(1)

    print("Found {} bag(s):".format(len(bag_files)))
    for b in bag_files:
        print("  - {}".format(b))

    total_profiles_read = 0
    total_profiles_kept = 0
    all_roi_points = []

    with rosbag.Bag(OUTPUT_BAG, "w") as outbag:
        for bag_path in bag_files:
            print("\nProcessing bag: {}".format(bag_path))
            bag_read = 0
            bag_kept = 0

            try:
                with rosbag.Bag(bag_path, "r") as inbag:
                    for topic, msg, t in inbag.read_messages(topics=[PROFILE_TOPIC]):
                        total_profiles_read += 1
                        bag_read += 1

                        try:
                            pts = extract_points_from_message(msg)
                        except Exception as e:
                            print("  [WARN] Could not parse message at time {} in {}: {}".format(
                                t.to_sec(), bag_path, e
                            ))
                            continue

                        if pts.shape[0] == 0:
                            continue

                        mask = (
                            (pts[:, 0] >= x_min) &
                            (pts[:, 0] <= x_max) &
                            (pts[:, 1] >= y_min) &
                            (pts[:, 1] <= y_max)
                        )

                        if np.any(mask):
                            outbag.write(PROFILE_TOPIC, msg, t)
                            total_profiles_kept += 1
                            bag_kept += 1

                            # Export only ROI points
                            roi_pts = pts[mask]
                            if roi_pts.shape[0] > 0:
                                all_roi_points.append(roi_pts)

                print("  Read profiles: {}".format(bag_read))
                print("  Kept profiles: {}".format(bag_kept))

            except Exception as e:
                print("  [ERROR] Failed to process {}: {}".format(bag_path, e))
                traceback.print_exc()

    if len(all_roi_points) > 0:
        merged_points = np.vstack(all_roi_points)
    else:
        merged_points = np.empty((0, 3), dtype=np.float64)

    write_ascii_ply(merged_points, OUTPUT_PLY)

    print("\n============================================================")
    print("Done.")
    print("Profiles read:  {}".format(total_profiles_read))
    print("Profiles kept:  {}".format(total_profiles_kept))
    print("ROI points:     {}".format(len(merged_points)))
    print("Output bag:     {}".format(OUTPUT_BAG))
    print("Output PLY:     {}".format(OUTPUT_PLY))
    print("============================================================")


if __name__ == "__main__":
    main()