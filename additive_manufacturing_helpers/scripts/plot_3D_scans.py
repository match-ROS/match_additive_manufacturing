#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2

# ----------------- CONFIG -----------------
bagfile = "record_20251205_130609_MuR.bag"
scan_topic_pc2 = "/profiles"
output_ply = "scans_export.ply"
# -------------------------------------------


# ----------------- PLY EXPORT -----------------
def save_ply(filename, points):
    """Save Nx3 points as ASCII PLY file."""
    N = points.shape[0]
    with open(filename, "w") as f:
        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # points
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

    print(f"[OK] Saved PLY file: {filename}")


# ----------------- LOAD BAG -----------------
points = []

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[scan_topic_pc2]):

        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

points = np.array(points)


# ----------------- EXPORT -----------------
save_ply(output_ply, points)


# ----------------- PLOT -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[::100, 0], points[::100, 1], points[::100, 2], s=1)
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