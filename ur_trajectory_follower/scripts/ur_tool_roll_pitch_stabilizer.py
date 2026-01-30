#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Twist


class ToolRollPitchStabilizer:
    """Stabilize tool orientation about base-frame X/Y (roll/pitch)."""
    def __init__(self):
        # Roll control gains
        self.kp_roll = rospy.get_param("~kp_roll", 0.1)
        self.ki_roll = rospy.get_param("~ki_roll", 0.0)
        self.kd_roll = rospy.get_param("~kd_roll", 0.0)

        # Pitch control gains
        self.kp_pitch = rospy.get_param("~kp_pitch", 0.1)
        self.ki_pitch = rospy.get_param("~ki_pitch", 0.0)
        self.kd_pitch = rospy.get_param("~kd_pitch", 0.0)

        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.9)
        self.use_current_orientation_as_reference = rospy.get_param("~use_current_orientation_as_reference", True)
        self.reference_roll = rospy.get_param("~reference_roll", 0.0)
        self.reference_pitch = rospy.get_param("~reference_pitch", 0.0)
        self.reference_orientation = None

        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.prev_error_roll = 0.0
        self.prev_error_pitch = 0.0

        self.command_old_twist = Twist()
        self.current_pose: Optional[PoseStamped] = None
        self.reference_set = not self.use_current_orientation_as_reference

        pose_topic = rospy.get_param("~current_pose_topic", "/mur620c/UR10_r/ur_calibrated_pose")
        twist_topic = rospy.get_param("~twist_topic", "/ur_roll_pitch_twist")

        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=1)

        self.pub_ur_velocity_world = rospy.Publisher(twist_topic, Twist, queue_size=10)

    def pose_callback(self, pose_msg: PoseStamped):
        self.current_pose = pose_msg
        if not self.reference_set:
            self.reference_roll, self.reference_pitch = self.quaternion_to_roll_pitch(pose_msg.pose.orientation)
            self.reference_orientation = pose_msg.pose.orientation
            self.reference_set = True
            rospy.loginfo(
                "Roll/pitch reference set from current pose: roll=%.3f, pitch=%.3f",
                self.reference_roll,
                self.reference_pitch,
            )
        self.calculate_twist()

    @staticmethod
    def quaternion_to_roll_pitch(orientation) -> Tuple[float, float]:
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (orientation.w * orientation.x + orientation.y * orientation.z)
        cosr_cosp = 1.0 - 2.0 * (orientation.x ** 2 + orientation.y ** 2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (orientation.w * orientation.y - orientation.z * orientation.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        return roll, pitch

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def smooth_output(self, control_command: Twist) -> Twist:
        smoothed_command = Twist()
        smoothed_command.angular.x = (
            self.output_smoothing_coeff * self.command_old_twist.angular.x
            + (1 - self.output_smoothing_coeff) * control_command.angular.x
        )
        smoothed_command.angular.y = (
            self.output_smoothing_coeff * self.command_old_twist.angular.y
            + (1 - self.output_smoothing_coeff) * control_command.angular.y
        )
        self.command_old_twist = smoothed_command
        return smoothed_command

    def calculate_twist(self):
        if self.current_pose is None or not self.reference_set:
            return

        if self.reference_orientation is None:
            return

        q_current = self.current_pose.pose.orientation
        q_ref = self.reference_orientation
        q_ref_tf = (q_ref.x, q_ref.y, q_ref.z, q_ref.w)
        q_cur_tf = (q_current.x, q_current.y, q_current.z, q_current.w)
        q_err = tft.quaternion_multiply(q_ref_tf, tft.quaternion_inverse(q_cur_tf))
        roll_error, pitch_error = self.quaternion_to_roll_pitch(
            type("Q", (), {"x": q_err[0], "y": q_err[1], "z": q_err[2], "w": q_err[3]})()
        )
        roll_error = self.wrap_to_pi(roll_error)
        pitch_error = self.wrap_to_pi(pitch_error)

        # velocity_scale = self.velocity_override

        omega_x = (roll_error * self.kp_roll)
        omega_y = (pitch_error * self.kp_pitch)

        control_command = Twist()
        control_command.angular.x = omega_x
        control_command.angular.y = omega_y
        control_command.angular.z = 0.0

        control_command_smoothed = self.smooth_output(control_command)
        self.pub_ur_velocity_world.publish(control_command_smoothed)


def main():
    rospy.init_node("ur_tool_roll_pitch_stabilizer")
    ToolRollPitchStabilizer()
    rospy.spin()


if __name__ == "__main__":
    main()
