#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32


class OrientationController:
    def __init__(self):
        self.kp_orientation = rospy.get_param("~kp_orientation", 1.0)
        self.ki_orientation = rospy.get_param("~ki_orientation", 0.0)
        self.kd_orientation = rospy.get_param("~kd_orientation", 0.0)
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.9)

        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

        self.path: Optional[Path] = None
        self.current_pose: Optional[PoseStamped] = None
        self.command_old_twist = Twist()
        self.current_index = 1
        self.velocity_override = 1.0

        self.initial_path_index = self._parse_initial_path_index(rospy.get_param("~initial_path_index", -1))
        if self.initial_path_index is not None:
            self.current_index = self.initial_path_index
            rospy.loginfo("Orientation controller starting from path index %s.", self.current_index)

        path_topic = rospy.get_param("~path_topic", "/path")
        pose_topic = rospy.get_param("~current_pose_topic", "/current_pose")
        index_topic = rospy.get_param("~path_index_topic", "/path_index")
        velocity_topic = rospy.get_param("~velocity_override_topic", "/velocity_override")
        twist_topic = rospy.get_param("~twist_topic", "/ur_orientation_twist")

        rospy.Subscriber(path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(index_topic, Int32, self.path_index_callback, queue_size=1)
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber(velocity_topic, Float32, self.velocity_override_callback, queue_size=1)

        self.pub_twist = rospy.Publisher(twist_topic, Twist, queue_size=10)

    def path_callback(self, path_msg: Path):
        if len(path_msg.poses) < 2:
            rospy.logwarn("Received path with less than two waypoints; cannot compute orientation.")
            return
        self.path = path_msg
        self.current_index = min(max(self.current_index, 1), len(self.path.poses) - 1)

    def path_index_callback(self, index_msg: Int32):
        if self.path is None or len(self.path.poses) < 2:
            return
        new_index = max(1, min(index_msg.data, len(self.path.poses) - 1))
        if new_index == self.current_index:
            return
        self.current_index = new_index
        if self.current_pose is not None:
            self.calculate_twist()

    def pose_callback(self, pose_msg: PoseStamped):
        self.current_pose = pose_msg
        if self.path is None or len(self.path.poses) < 2:
            rospy.logwarn_throttle(5.0, "No valid path received yet.")
            return
        self.calculate_twist()

    def velocity_override_callback(self, velocity_msg: Float32):
        self.velocity_override = max(0.0, min(velocity_msg.data, 1.0))

    @staticmethod
    def _axis_angle_from_quaternion(quat: np.ndarray):
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:
            return np.zeros(3), 0.0
        quat = quat / quat_norm

        w = max(-1.0, min(1.0, quat[3]))
        angle = 2.0 * math.acos(w)
        if angle > math.pi:
            angle -= 2.0 * math.pi

        sin_half = math.sqrt(max(1.0 - w * w, 0.0))
        if sin_half < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = quat[0:3] / sin_half
        return axis, angle

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
        smoothed_command.angular.z = (
            self.output_smoothing_coeff * self.command_old_twist.angular.z
            + (1 - self.output_smoothing_coeff) * control_command.angular.z
        )
        self.command_old_twist = smoothed_command
        return smoothed_command

    def calculate_twist(self):
        if self.current_pose is None or self.path is None:
            return

        goal_pose = self.path.poses[self.current_index]
        q_des = np.array(
            [
                goal_pose.pose.orientation.x,
                goal_pose.pose.orientation.y,
                goal_pose.pose.orientation.z,
                goal_pose.pose.orientation.w,
            ],
            dtype=float,
        )
        q_cur = np.array(
            [
                self.current_pose.pose.orientation.x,
                self.current_pose.pose.orientation.y,
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w,
            ],
            dtype=float,
        )

        if np.linalg.norm(q_des) < 1e-6 or np.linalg.norm(q_cur) < 1e-6:
            rospy.logwarn_throttle(5.0, "Invalid quaternion for orientation control.")
            return

        q_des = q_des / np.linalg.norm(q_des)
        q_cur = q_cur / np.linalg.norm(q_cur)
        q_err = tft.quaternion_multiply(q_des, tft.quaternion_inverse(q_cur))
        axis, angle = self._axis_angle_from_quaternion(np.array(q_err, dtype=float))
        error_vec = axis * angle

        omega = (
            error_vec * self.kp_orientation
            + self.integral_error * self.ki_orientation
            + (error_vec - self.prev_error) * self.kd_orientation
        )
        omega *= self.velocity_override

        self.integral_error += error_vec
        self.prev_error = error_vec

        control_command = Twist()
        control_command.angular.x = omega[0]
        control_command.angular.y = omega[1]
        control_command.angular.z = omega[2]

        control_command_smoothed = self.smooth_output(control_command)
        self.pub_twist.publish(control_command_smoothed)

    @staticmethod
    def _parse_initial_path_index(raw_value):
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "Invalid initial_path_index value '%s', ignoring.", raw_value)
            return None
        return idx if idx >= 0 else None


def main():
    rospy.init_node("ur_path_orientation_controller")
    OrientationController()
    rospy.spin()


if __name__ == "__main__":
    main()
