#!/usr/bin/env python3
import rosbag
import numpy as np
import tf.transformations as tf
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2

# ----------------- CONFIG -----------------
bagfile = "record_20251203_155544_MuR.bag"
scan_topic_float = "/profiles_float"
scan_topic_pc2   = "/profiles"
pose_topic = "/mur620c/UR10_r/global_tcp_pose"
nozzle_angle_deg = 100.0
# -------------------------------------------

scans = []
poses = []
use_pc2 = True
scanner_rot = tf.rotation_matrix(np.deg2rad(nozzle_angle_deg), (0, 0, 1))[0:3, 0:3]

# ----------------- LOAD BAG -----------------
with rosbag.Bag(bagfile, "r") as bag:

    topics = bag.get_type_and_topic_info().topics.keys()
    float_available = scan_topic_float in topics
    pc2_available   = scan_topic_pc2 in topics

    if not float_available and not pc2_available:
        raise RuntimeError("Neither /profiles_float nor /profiles found in bag.")

    # Prefer float-mode if available
    use_pc2 = not float_available

    for topic, msg, t in bag.read_messages():
        # FLOAT MODE
        if not use_pc2 and topic == scan_topic_float:
            scans.append({
                "t": None,
                "data": np.array(msg.data)  # 1D profile
            })

        # POINTCLOUD2 MODE
        elif use_pc2 and topic == scan_topic_pc2:
            pts = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append([p[0], p[1], p[2]])
            scans.append({
                "t": t.to_sec(),
                "data": np.array(pts)  # Nx3 points in scanner frame
            })

        # TCP POSE
        elif topic == pose_topic:
            p = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          msg.pose.position.z])
            q = np.array([msg.pose.orientation.x,
                          msg.pose.orientation.y,
                          msg.pose.orientation.z,
                          msg.pose.orientation.w])
            poses.append((t.to_sec(), p, q))

# ----------------- PREPARE POSE DATA -----------------
poses = sorted(poses, key=lambda x: x[0])

T_pose = np.array([p[0] for p in poses])
P_pose = np.array([p[1] for p in poses])
Q_pose = np.array([p[2] for p in poses])

t_start = T_pose[0]
t_end   = T_pose[-1]

# ----------------- RECONSTRUCT TIMESTAMPS -----------------
if not use_pc2:
    N = len(scans)
    T_scan = np.linspace(t_start, t_end, N)
    for i in range(N):
        scans[i]["t"] = T_scan[i]
else:
    T_scan = np.array([scan["t"] for scan in scans])

# ----------------- SLERP INTERPOLATION -----------------
def slerp(q1, q2, t):
    return tf.quaternion_slerp(q1, q2, t)

def interp_pose(t):
    idx = np.searchsorted(T_pose, t)
    idx = np.clip(idx, 1, len(T_pose)-1)

    t0, t1 = T_pose[idx-1], T_pose[idx]
    u = (t - t0) / (t1 - t0)

    p = (1-u)*P_pose[idx-1] + u*P_pose[idx]
    q = slerp(Q_pose[idx-1], Q_pose[idx], u)
    return p, q

# ----------------- TRANSFORM SCANS -----------------
points = []

for scan in scans:
    t = scan["t"]
    p_tcp, q_tcp = interp_pose(t)
    R = tf.quaternion_matrix(q_tcp)[0:3, 0:3]

    data = scan["data"]

    # FLOAT MODE: 1D values along Y of TCP
    if data.ndim == 1:
        for val in data:
            local = np.array([0.0, 0.0, val])
            local = scanner_rot.dot(local)
            world = p_tcp + R.dot(local)
            points.append(world)

    # PC2 MODE: Nx3 points
    else:
        for pt in data:
            local = scanner_rot.dot(pt)
            world = p_tcp + R.dot(local)
            points.append(world)

points = np.array(points)
# invert z axis
points[:,2] = -points[:,2]

# ----------------- PLOT -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[::100, 0], points[::100, 1], points[::100, 2], s=1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# --- Equal axis scaling ---
max_range = np.array([
    points[:,0].max() - points[:,0].min(),
    points[:,1].max() - points[:,1].min(),
    points[:,2].max() - points[:,2].min()
]).max() / 2.0

mid_x = (points[:,0].max() + points[:,0].min()) * 0.5
mid_y = (points[:,1].max() + points[:,1].min()) * 0.5
mid_z = (points[:,2].max() + points[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
