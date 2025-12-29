#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

# ---- helpers ----

def quat_to_yaw(q):
    """Quaternion -> yaw (rad)."""
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def path_to_arrays(path_msg):
    """Return Nx2 positions and N yaws from nav_msgs/Path."""
    poses = path_msg.poses
    n = len(poses)
    xy = np.zeros((n, 2), dtype=np.float64)
    yaw = np.zeros((n,), dtype=np.float64)

    for i, ps in enumerate(poses):
        xy[i, 0] = ps.pose.position.x
        xy[i, 1] = ps.pose.position.y
        yaw[i] = quat_to_yaw(ps.pose.orientation)

    return xy, yaw

def speeds_from_xy_t(xy, t):
    """Compute speed magnitude from xy positions and timestamps."""
    # forward differences
    dxy = np.diff(xy, axis=0)
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-9)  # avoid div-by-zero
    v = np.linalg.norm(dxy, axis=1) / dt
    # align to length N by padding last value
    v_full = np.concatenate([v, [v[-1] if len(v) else 0.0]])
    return v_full

def rotate2d(vx, vy, yaw):
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rx = cy * vx - sy * vy
    ry = sy * vx + cy * vy
    return rx, ry

# ---- main ----

def main():
    rospy.init_node("mir_ur_path_live_analyzer", anonymous=True)

    mir_path_topic = rospy.get_param("~mir_path_topic", "/mur620c/mir_path_original")
    ur_path_topic  = rospy.get_param("~ur_path_topic",  "/mur620c/ur_path_original")
    mir_ts_topic   = rospy.get_param("~mir_ts_topic",   "/mur620c/mir_path_timestamps")

    # UR mounting offset on MiR base frame (meters)
    # given: xyz="0.549 -0.318 0.49" (we use x,y for planar distance)
    ur_mount_x = rospy.get_param("~ur_mount_x", 0.549)
    ur_mount_y = rospy.get_param("~ur_mount_y", -0.318)

    rospy.loginfo("Waiting for live messages...")
    mir_path_msg = rospy.wait_for_message(mir_path_topic, Path, timeout=20.0)
    ur_path_msg  = rospy.wait_for_message(ur_path_topic,  Path, timeout=20.0)
    mir_ts_msg   = rospy.wait_for_message(mir_ts_topic,   Float32MultiArray, timeout=20.0)

    mir_xy, mir_yaw = path_to_arrays(mir_path_msg)
    ur_xy, _ = path_to_arrays(ur_path_msg)

    t_mir = np.array(mir_ts_msg.data, dtype=np.float64)

    n_mir = mir_xy.shape[0]
    n_ur  = ur_xy.shape[0]

    if n_mir < 2 or n_ur < 2:
        raise RuntimeError("Paths must have at least 2 points.")

    if len(t_mir) != n_mir:
        #raise RuntimeError(f"Timestamp length ({len(t_mir)}) != MiR path length ({n_mir}).")
        t_mir = t_mir[:n_mir]

    # Build UR timestamps: same start/end as MiR, constant dt, same number of steps as UR path.
    t0 = float(t_mir[0])
    t1 = float(t_mir[-1])
    dt_ur = (t1 - t0) / float(max(n_ur - 1, 1))
    t_ur = t0 + dt_ur * np.arange(n_ur, dtype=np.float64)

    # Speeds
    v_mir = speeds_from_xy_t(mir_xy, t_mir)
    v_ur  = speeds_from_xy_t(ur_xy,  t_ur)

    # Distance UR-basis <-> UR-TCP per index (planar)
    # UR basis position = MiR position + R(yaw_mir) * mount_offset_xy
    # Then dist = || ur_tcp_xy - ur_basis_xy ||
    if n_ur != n_mir:
        rospy.logwarn(f"UR path length ({n_ur}) != MiR path length ({n_mir}). "
                      f"Using min length for paired-index analysis.")
    n = min(n_mir, n_ur)
    dist = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        ox, oy = rotate2d(ur_mount_x, ur_mount_y, float(mir_yaw[i]))
        ur_base_x = float(mir_xy[i, 0] + ox)
        ur_base_y = float(mir_xy[i, 1] + oy)
        dx = float(ur_xy[i, 0] - ur_base_x)
        dy = float(ur_xy[i, 1] - ur_base_y)
        dist[i] = math.hypot(dx, dy)

    # ---- plots ----
    fig1 = plt.figure()
    plt.plot(t_mir[:n], v_mir[:n], label="MiR speed")
    plt.plot(t_ur[:n],  v_ur[:n],  label="UR TCP speed (const dt)")
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.grid(True)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(t_mir[:n], dist, label="||UR_TCP - UR_base|| (planar)")
    plt.xlabel("time [s]")
    plt.ylabel("distance [m]")
    plt.grid(True)
    plt.legend()

    fig3 = plt.figure()
    plt.plot(mir_xy[:n, 0], mir_xy[:n, 1], label="MiR path")
    plt.plot(ur_xy[:n, 0],  ur_xy[:n, 1],  label="UR TCP path")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()

    plt.show()
    rospy.loginfo("Done.")

if __name__ == "__main__":
    main()
