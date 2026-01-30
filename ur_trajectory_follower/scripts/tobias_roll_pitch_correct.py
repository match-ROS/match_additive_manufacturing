#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion
import math
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_multiply
)

class KeepUprightController:
    def __init__(self):
        self.kp = rospy.get_param("~kp", 0.1)      # P-gain
        self.max_w = rospy.get_param("~max_w", 0.1)  # rad/s limit

        # self.pub = rospy.Publisher(
        #     "/ur_roll_pitch_twist",
        #     Twist,
        #     queue_size=1
        # )

        self.pub = rospy.Publisher(
            "/mur620c/UR10_r/twist_controller/command_collision_free",
            Twist,
            queue_size=1
        )

        rospy.Subscriber(
            "/mur620c/UR10_r/ur_calibrated_pose",
            PoseStamped,
            self.pose_cb,
            queue_size=1
        )


    def quat_to_rotvec(self, q):
        # q = [x, y, z, w]
        v = np.array(q[:3])
        w = q[3]

        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return np.zeros(3)

        theta = 2.0 * np.arctan2(norm_v, w)
        return theta * v / norm_v


    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def pose_cb(self, msg):
            q = msg.pose.orientation
            q_cur = np.array([q.x, q.y, q.z, q.w])

            # current yaw
            roll, pitch, yaw = euler_from_quaternion(q_cur)

            # desired orientation: upright, keep yaw
            q_des = quaternion_from_euler(0.0, np.pi, yaw)

            # quaternion error
            q_err = quaternion_multiply(q_des, quaternion_inverse(q_cur))

            # small-angle rotation error
            rx, ry, rz = euler_from_quaternion(q_err)
            print(f"Roll error: {rx:.4f}, Pitch error: {ry:.4f}, Yaw error: {rz:.4f}")
            print(f"Current roll: {roll:.4f}, Current pitch: {pitch:.4f}, Current yaw: {yaw:.4f}")

            twist = Twist()
            twist.angular.x = np.clip(self.kp * rx, -self.max_w, self.max_w)
            twist.angular.y = np.clip(self.kp * ry, -self.max_w, self.max_w)
            twist.angular.z = 0.0  # yaw not controlled

            self.pub.publish(twist)

if __name__ == "__main__":
    rospy.init_node("ur_keep_upright_controller")
    KeepUprightController()
    rospy.spin()
