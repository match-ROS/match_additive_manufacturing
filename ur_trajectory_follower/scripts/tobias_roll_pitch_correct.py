#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion
from math import pi

class KeepUprightController:
    def __init__(self):
        self.kp = rospy.get_param("~kp", 0.1)      # P-gain
        self.max_w = rospy.get_param("~max_w", 0.1)  # rad/s limit

        self.pub = rospy.Publisher(
            "/ur_roll_pitch_twist_world",
            Twist,
            queue_size=1
        )

        rospy.Subscriber(
            "/mur620c/UR10_r/ur_calibrated_pose",
            PoseStamped,
            self.pose_cb,
            queue_size=1
        )

    def pose_cb(self, msg):
        q = msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Desired: roll = 0, pitch = 0
        err_roll  = pi - roll
        
        err_pitch = - pitch
        print("Pitch error:", err_pitch)

        twist = Twist()
        twist.angular.x = np.clip(self.kp * err_roll,  -self.max_w, self.max_w)
        twist.angular.y = np.clip(self.kp * err_pitch, -self.max_w, self.max_w)
        twist.angular.z = 0.0  # yaw free

        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        self.pub.publish(twist)

if __name__ == "__main__":
    rospy.init_node("ur_keep_upright_controller")
    KeepUprightController()
    rospy.spin()
