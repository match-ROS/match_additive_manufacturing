#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion


class WeldSeamOffsetController:
    def __init__(self):
        rospy.init_node("weld_seam_offset_controller")

        # Params
        self.offset_topic = "/layer_center/offset_mm"
        self.path_topic = "/ur_path_original"
        self.index_topic = "/path_index"
        self.cmd_topic = "/mur620a/UR10_r/twist_fb_command"

        self.max_correction = 0.05  # m/s
        self.gain = 2.0  # proportional gain (m/s per mm)

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

        pose = self.path[self.current_index].pose
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Transform offset from scanner frame (right = +, left = -) to world
        offset_m = self.latest_offset / 1000.0
        correction_x = -offset_m * np.sin(yaw)
        correction_y =  offset_m * np.cos(yaw)

        # Apply gain and saturation
        cmd = Twist()
        cmd.linear.x = np.clip(correction_x * self.gain, -self.max_correction, self.max_correction)
        cmd.linear.y = np.clip(correction_y * self.gain, -self.max_correction, self.max_correction)
        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(1.0, f"Correction (x={cmd.linear.x:.3f}, y={cmd.linear.y:.3f})")

if __name__ == '__main__':
    try:
        WeldSeamOffsetController()
    except rospy.ROSInterruptException:
        pass
