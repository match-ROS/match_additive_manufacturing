#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path



# ----------------- CONFIG -----------------
bagfile = "record_20251205_135424_MuR.bag"
scan_topic_pc2 = "/profiles"
output_ply = "scans_export.ply"
path_topic = "/ur_path_original"
use_live_path_if_missing = True
z_offset = 0.62
# -------------------------------------------

def extract_lowest_layer(path_points, tol=1e-4):
    path_points = np.array(path_points)
    if len(path_points) == 0:
        raise RuntimeError("No path points available.")

    z0 = path_points[0, 2]
    idx_same = np.where(np.abs(path_points[:,2] - z0) < tol)[0]
    end_idx = idx_same[-1]
    return path_points[:end_idx+1]

# ----------------- LOAD BAG -----------------
points = []

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[scan_topic_pc2]):

        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

points = np.array(points)

# ----------------- TRY LOADING PATH FROM BAG -----------------
path_points = []

with rosbag.Bag(bagfile, "r") as bag:
    topics = bag.get_type_and_topic_info().topics.keys()

    if path_topic in topics:
        # Path exists in bag → load it
        for topic, msg, t in bag.read_messages(topics=[path_topic]):
            for pose in msg.poses:
                p = pose.pose.position
                path_points.append([p.x, p.y, p.z])
            break  # take first full Path message

# ----------------- FALLBACK: SUBSCRIBE LIVE IF NOT FOUND -----------------
if len(path_points) == 0 and use_live_path_if_missing:
    import rospy
    from nav_msgs.msg import Path

    rospy.init_node("path_reader_temp", anonymous=True)
    print("[INFO] No path in bag. Waiting for live path message on /ur_path_original...")

    received = []

    def cb(msg):
        for pose in msg.poses:
            p = pose.pose.position
            received.append([p.x, p.y, p.z])
        rospy.signal_shutdown("Path received.")

    sub = rospy.Subscriber(path_topic, Path, cb)

    rospy.spin()
    path_points = received
    print(f"[INFO] Received live path with {len(path_points)} points.")


# ----------------- EXTRACT LOWEST LAYER -----------------
lowest_layer = extract_lowest_layer(path_points)


# ----------------- PLOT -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot scanned points
ax.scatter(points[::100, 0], points[::100, 1], points[::100, 2], s=1, label="Scans")

# Plot lowest layer of path
ax.plot(lowest_layer[:,0], lowest_layer[:,1], lowest_layer[:,2] + z_offset,
        color='red', linewidth=2, label="UR Path (lowest layer)")

ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Equal axis scaling
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



