#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
import tf2_ros
import tf2_geometry_msgs
from scipy.signal import medfilt, find_peaks
from collections import deque


class ProfileCenterFromPointCloud:
    def __init__(self):
        rospy.init_node("profile_center_pointcloud_node")

        self.cloud_topic = rospy.get_param("~scan_topic", "/line_laser/pointcloud")
        self.center_point_topic = rospy.get_param("~center_point_topic", "/profile/center_point")
        self.lateral_offset_topic = rospy.get_param("~lateral_offset_topic", "/layer_center/offset_mm")
        self.tf_target_frame = rospy.get_param("~tf_target_frame", "mur620a/base_link")
        self.peak_prominence = 0.002  # adapt as needed
        self.median_filter_kernel = 5

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.center_point_topic, PointStamped, queue_size=1)
        self.offset_pub = rospy.Publisher(self.lateral_offset_topic, Float32, queue_size=1)
        self.offset_buffer = deque(maxlen=10)

        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback)

        rospy.loginfo("Profile Center from PointCloud2 started.")
        rospy.spin()

    def cloud_callback(self, msg: PointCloud2):
        rospy.loginfo_throttle(2.0, "Received PointCloud2 message.")

        # Read points
        points = np.array(list(point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)))
        if len(points) < 5:
            rospy.logwarn_throttle(5, "PointCloud has too few valid points.")
            return

        point_index = points[:, 0]  # index (0 to N)
        depth = points[:, 2]        # height

        # Filter out NaN and Inf
        valid_mask = np.isfinite(point_index) & np.isfinite(depth)
        point_index = point_index[valid_mask]
        depth = depth[valid_mask]

        if len(depth) == 0:
            rospy.logwarn_throttle(5, "No valid depth values.")
            return

        # Find highest point
        max_idx = np.argmin(depth)
        max_depth = depth[max_idx]
        max_index = point_index[max_idx]

        # Calculate offset from center
        center_index = (np.max(point_index) + np.min(point_index)) / 2
        offset_from_center = max_index - center_index

        rospy.loginfo_throttle(2.0, f"Max height: {max_depth:.3f} at index {max_index:.1f}, offset from center: {offset_from_center:.1f}")



if __name__ == "__main__":
    try:
        ProfileCenterFromPointCloud()
    except rospy.ROSInterruptException:
        pass
