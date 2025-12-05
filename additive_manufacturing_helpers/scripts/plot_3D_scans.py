#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2

# ----------------- CONFIG -----------------
bagfile = "record_20251205_112954_MuR.bag"
scan_topic_pc2 = "/profiles"
# -------------------------------------------

points = []

# ----------------- LOAD BAG -----------------
with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[scan_topic_pc2]):

        # Each /profiles message is already a PointCloud2 in map frame
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

points = np.array(points)

# ----------------- PLOT -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[::100, 0], points[::100, 1], points[::100, 2], s=1)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

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
