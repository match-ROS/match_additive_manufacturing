#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class ProfileCollectorPlotter:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/profiles")
        self.num_msgs = int(rospy.get_param("~num_msgs", 151))      # total messages to read
        self.every_n = int(rospy.get_param("~every_n", 5))        # store every n-th message
        self.max_points_per_msg = int(rospy.get_param("~max_points_per_msg", 0))

        self.points = []
        self.total_count = 0
        self.stored_count = 0

        rospy.Subscriber(self.topic, PointCloud2, self.cb, queue_size=1)
        rospy.loginfo(f"Reading {self.num_msgs} messages, storing every {self.every_n}-th")

    def cb(self, msg: PointCloud2):
        if self.total_count >= self.num_msgs:
            return

        self.total_count += 1

        # Only keep every n-th message
        if (self.total_count - 1) % self.every_n != 0:
            return

        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if not pts:
            return

        pts = np.asarray(pts, dtype=np.float32)

        if self.max_points_per_msg > 0 and pts.shape[0] > self.max_points_per_msg:
            idx = np.random.choice(pts.shape[0], self.max_points_per_msg, replace=False)
            pts = pts[idx]

        self.points.append(pts)
        self.stored_count += 1

        rospy.loginfo(f"Msg {self.total_count}/{self.num_msgs} | stored {self.stored_count}")

        if self.total_count == self.num_msgs:
            self.plot_and_exit()

    def plot_and_exit(self):
        if not self.points:
            rospy.logwarn("No points collected.")
            rospy.signal_shutdown("Done")
            return

        all_pts = np.vstack(self.points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], s=1)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"{self.stored_count} profiles from {self.num_msgs} messages")

        plt.show()
        rospy.signal_shutdown("Finished")

if __name__ == "__main__":
    rospy.init_node("profile_collector_plotter", anonymous=True)
    ProfileCollectorPlotter()
    rospy.spin()