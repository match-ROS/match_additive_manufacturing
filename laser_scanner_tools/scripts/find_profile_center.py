#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from scipy.signal import medfilt, find_peaks
import tf2_ros
import tf2_geometry_msgs


class WeldSeamDetector:
    def __init__(self):
        rospy.init_node("weld_seam_detector_node")

        # Parameters
        self.scan_topic = "/mur620a/UR10_r/line_laser/scan"
        self.filtered_pub_topic = "/weld_seam/center_point"
        self.median_filter_kernel = 5
        self.peak_prominence = 0.005  # adjust based on noise level
        self.tf_target_frame = "mur620a/base_link"

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.pub = rospy.Publisher(self.filtered_pub_topic, PointStamped, queue_size=1)

        # Subscriber
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback)

        rospy.loginfo("Weld seam detector started.")
        rospy.spin()

    def scan_callback(self, msg: LaserScan):
        # Convert scan ranges to numpy array
        ranges = np.array(msg.ranges)
        ranges[ranges == 0.0] = np.nan  # remove invalid points

        # Apply median filter
        filtered = medfilt(ranges, kernel_size=self.median_filter_kernel)

        # Invert and find peaks (highest becomes lowest)
        inverted = -filtered
        peaks, properties = find_peaks(inverted, prominence=self.peak_prominence)

        if len(peaks) == 0:
            rospy.logwarn("No weld seam peak found.")
            return

        # Take the most prominent peak
        best_peak = peaks[np.argmax(properties['prominences'])]
        range_at_peak = filtered[best_peak]
        angle_at_peak = msg.angle_min + best_peak * msg.angle_increment

        # Convert polar to cartesian in laser frame
        x = range_at_peak * np.cos(angle_at_peak)
        y = range_at_peak * np.sin(angle_at_peak)
        z = 0.0  # Laser scans in 2D plane; z=0 in its frame

        point_laser = PointStamped()
        point_laser.header = msg.header
        point_laser.point.x = x
        point_laser.point.y = y
        point_laser.point.z = z

        # Transform to robot frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.tf_target_frame,
                point_laser.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            point_robot = tf2_geometry_msgs.do_transform_point(point_laser, transform)
            self.pub.publish(point_robot)

            rospy.loginfo_throttle(1.0, f"Detected weld seam at x={point_robot.point.x:.3f}, y={point_robot.point.y:.3f}, z={point_robot.point.z:.3f}")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF transform failed: {e}")


if __name__ == '__main__':
    try:
        WeldSeamDetector()
    except rospy.ROSInterruptException:
        pass
