#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
import tf2_ros
from geometry_msgs.msg import PointStamped


class WeldSeamOffsetController:
    def __init__(self):
        rospy.init_node("weld_seam_offset_controller")

        # Params
        self.offset_topic = rospy.get_param("~offset_topic", "/layer_center/offset_mm")
        self.path_topic = rospy.get_param("~path_topic", "/ur_path_original")
        self.index_topic = rospy.get_param("~index_topic", "/path_index")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/mur620a/UR10_r/twist_fb_command")
        self.sensor_frame = rospy.get_param("~sensor_frame", "mur620a/UR10_r/line_laser_frame")
        self.target_frame = rospy.get_param("~target_frame", "mur620a/base_link")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.max_correction = 0.05  # m/s
        self.gain = 0.2  # proportional gain (m/s per mm)

        # Internal state
        self.current_index = 0
        self.path = None
        self.latest_offset = 0.0

        # Subscribers
        rospy.Subscriber(self.offset_topic, Float32, self.offset_callback)
        rospy.Subscriber(self.index_topic, Int32, self.index_callback)
        rospy.Subscriber(self.path_topic, Path, self.path_callback)

        # Publisher
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz
        rospy.loginfo("Weld seam offset controller started.")
        rospy.spin()

    def offset_callback(self, msg: Float32):
        self.latest_offset = msg.data

    def index_callback(self, msg: Int32):
        self.current_index = msg.data

    def path_callback(self, msg: Path):
        self.path = msg.poses

    def control_loop(self, event):
        if self.path is None or not (0 <= self.current_index < len(self.path)):
            rospy.logwarn_throttle(1.0, "Path not available or index out of bounds.")
            return

        try:
            # Offset in Sensor-X-Richtung
            offset_m = self.latest_offset / 1000.0

            # Erzeuge Punkt auf X-Achse des Sensors
            offset_point = PointStamped()
            offset_point.header.frame_id = self.sensor_frame
            offset_point.header.stamp = rospy.Time(0)
            offset_point.point.x = -offset_m  # gegen +X des Sensors
            offset_point.point.y = 0.0
            offset_point.point.z = 0.0

            # Transformiere in Weltkoordinaten
            transformed = self.tf_buffer.transform(offset_point, self.target_frame, rospy.Duration(0.5))

            # Verwende x/y für Geschwindigkeit
            cmd = Twist()
            cmd.linear.x = np.clip(transformed.point.x * self.gain, -self.max_correction, self.max_correction)
            cmd.linear.y = np.clip(transformed.point.y * self.gain, -self.max_correction, self.max_correction)
            self.cmd_pub.publish(cmd)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF lookup failed: {e}")


if __name__ == '__main__':
    try:
        WeldSeamOffsetController()
    except rospy.ROSInterruptException:
        pass
