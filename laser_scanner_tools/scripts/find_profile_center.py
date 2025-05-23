#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from scipy.signal import medfilt, find_peaks
import tf2_ros
import tf2_geometry_msgs
from collections import deque
from std_msgs.msg import Float32


class ProfileCenterDetector:
    def __init__(self):
        rospy.init_node("profile_center_detector_node")

        # Parameters
        self.scan_topic = "/mur620a/UR10_r/line_laser/scan"
        self.output_topic = "/profile/center_point"
        self.tf_target_frame = "mur620a/base_link"
        self.median_filter_kernel = 5
        self.peak_prominence = 0.005
        self.gradient_threshold = 0.02
        self.fallback_ratio = 0.6
        self.min_profile_width = 20  # minimum number of points to consider a valid profile

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=1)
        self.offset_pub = rospy.Publisher("/layer_center/offset_mm", Float32, queue_size=1)
        self.offset_buffer = deque(maxlen=100)  # gleitende Mittelung über 100 Scans

        # Subscriber
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback)

        rospy.loginfo("Profile Center detector started.")
        rospy.spin()

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges[ranges == 0.0] = np.nan  # remove invalid zero values

        # Apply median filter
        filtered = medfilt(ranges, kernel_size=self.median_filter_kernel)

        # Crop the scan to remove edges
        filtered_cropped, offset = self.crop_scan_adaptive(filtered)

        # Invert and find peaks
        inverted = -filtered_cropped
        peaks, properties = find_peaks(inverted, prominence=self.peak_prominence)

        

        if len(peaks) == 0:
            rospy.logwarn("No profile peak found.")
            return

        # Get most prominent peak
        best_peak = peaks[np.argmax(properties['prominences'])]
        absolute_index = best_peak + offset
        range_at_peak = filtered[absolute_index]
        angle_at_peak = msg.angle_min + absolute_index * msg.angle_increment


        center_idx = len(filtered) // 2
        offset_idx = absolute_index - center_idx
        offset_mm = offset_idx * msg.angle_increment * range_at_peak * 1000  # in mm
        self.offset_buffer.append(offset_mm)

        mean_offset = np.mean(self.offset_buffer)
        self.offset_pub.publish(Float32(data=mean_offset))

        # Convert to Cartesian in laser frame
        x = range_at_peak * np.cos(angle_at_peak)
        y = range_at_peak * np.sin(angle_at_peak)
        z = 0.0

        point_laser = PointStamped()
        point_laser.header = msg.header
        point_laser.point.x = x
        point_laser.point.y = y
        point_laser.point.z = z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.tf_target_frame,
                point_laser.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            point_robot = tf2_geometry_msgs.do_transform_point(point_laser, transform)
            self.pub.publish(point_robot)

            rospy.loginfo_throttle(1.0, f"Max: x={point_robot.point.x:.3f}, y={point_robot.point.y:.3f}, z={point_robot.point.z:.3f}")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF transform failed: {e}")

    def crop_scan_adaptive(self, filtered, grad_threshold=None, fallback_ratio=None):
        if grad_threshold is None:
            grad_threshold = self.gradient_threshold
        if fallback_ratio is None:
            fallback_ratio = self.fallback_ratio

        gradient = np.abs(np.gradient(filtered))

        edge_indices = np.where(gradient > grad_threshold)[0]

        if len(edge_indices) >= 2 and max(edge_indices) - min(edge_indices) >= self.min_profile_width:
            # Use the first and last edge indices to define the crop
            left = min(edge_indices)
            right = max(edge_indices)
        elif len(edge_indices) == 1:
            center = len(filtered) // 2
            width = int(len(filtered) * fallback_ratio)
            if edge_indices[0] < center:
                left = edge_indices[0]
                right = min(len(filtered) - 1, left + width)
            else:
                right = edge_indices[0]
                left = max(0, right - width)
        else:
            # Fallback: use center region
            margin = int((1 - fallback_ratio) / 2 * len(filtered))
            left = margin
            right = len(filtered) - margin

        if right <= left:
            left = 0
            right = len(filtered) - 1

        return filtered[left:right + 1], left


if __name__ == '__main__':
    try:
        ProfileCenterDetector()
    except rospy.ROSInterruptException:
        pass
