#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Int32


class DirectionYawController:
    def __init__(self):
        # Yaw control gains
        self.kp_yaw = rospy.get_param("~kp_yaw", 0.5)
        self.ki_yaw = rospy.get_param("~ki_yaw", 0.0)
        self.kd_yaw = rospy.get_param("~kd_yaw", 0.0)

        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.9)
        self.ff_only = rospy.get_param("~ff_only", False)
        self.calculate_path_direction = rospy.get_param("~calculate_path_direction", False)

        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0

        self.path: Optional[Path] = None
        self.current_pose: Optional[PoseStamped] = None
        self.command_old_twist = Twist()
        self.current_index = 1
        self.velocity_override = 1.0

        self.initial_path_index = self._parse_initial_path_index(rospy.get_param("~initial_path_index", -1))
        if self.initial_path_index is not None:
            self.current_index = self.initial_path_index
            rospy.loginfo(f"Yaw controller starting from path index {self.current_index} (parameter).")

        # Topics (defaults follow the user's request but remain configurable)
        path_topic = rospy.get_param("~path_topic", "/path")
        pose_topic = rospy.get_param("~current_pose_topic", "/global_nozzle_pose")
        index_topic = rospy.get_param("~path_index_topic", "/path_index")
        velocity_topic = rospy.get_param("~velocity_override_topic", "/velocity_override")
        twist_topic = rospy.get_param("~twist_topic", "/ur_twist_world")

        # Subscribers
        rospy.Subscriber(path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(index_topic, Int32, self.path_index_callback, queue_size=1)
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber(velocity_topic, Float32, self.velocity_override_callback, queue_size=1)

        # Publisher
        self.pub_ur_velocity_world = rospy.Publisher(twist_topic, Twist, queue_size=10)

    # ------------------------- Callbacks -------------------------
    def path_callback(self, path_msg: Path):
        if len(path_msg.poses) < 2:
            rospy.logwarn("Received path with less than two waypoints; cannot compute direction.")
            return
        self.path = path_msg
        self.current_index = min(max(self.current_index, 1), len(self.path.poses) - 1)

    def path_index_callback(self, index_msg: Int32):
        path = self.path
        if path is None or len(path.poses) < 2:
            return
        new_index = max(1, min(index_msg.data, len(path.poses) - 1))
        if new_index == self.current_index:
            return
        self.current_index = new_index
        if self.current_pose is not None:
            self.calculate_twist()

    def pose_callback(self, pose_msg: PoseStamped):
        self.current_pose = pose_msg
        path = self.path
        if path is None or len(path.poses) < 2:
            rospy.logwarn_throttle(5.0, "No valid path received yet.")
            return
        self.calculate_twist()

    def velocity_override_callback(self, velocity_msg: Float32):
        self.velocity_override = max(0.0, min(velocity_msg.data, 1.0))

    # ----------------------- Helper methods ----------------------

    def get_direction(self):
        path = self.path
        if path is None:
            return np.zeros(2)

        goal_pose = path.poses[self.current_index]
        if self.ff_only:
            from_pose = path.poses[max(self.current_index - 1, 0)]
        else:
            from_pose = self.current_pose if self.current_pose is not None else path.poses[max(self.current_index - 1, 0)]

        direction = np.array([
            goal_pose.pose.position.x - from_pose.pose.position.x,
            goal_pose.pose.position.y - from_pose.pose.position.y,
        ])
        norm_xy = np.linalg.norm(direction)
        if norm_xy < 1e-6:
            return np.zeros(2)
        return direction / norm_xy

    @staticmethod
    def quaternion_to_yaw(orientation) -> float:
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def smooth_output(self, control_command: Twist) -> Twist:
        smoothed_command = Twist()
        smoothed_command.linear.x = (
            self.output_smoothing_coeff * self.command_old_twist.linear.x
            + (1 - self.output_smoothing_coeff) * control_command.linear.x
        )
        smoothed_command.linear.y = (
            self.output_smoothing_coeff * self.command_old_twist.linear.y
            + (1 - self.output_smoothing_coeff) * control_command.linear.y
        )
        smoothed_command.linear.z = (
            self.output_smoothing_coeff * self.command_old_twist.linear.z
            + (1 - self.output_smoothing_coeff) * control_command.linear.z
        )
        smoothed_command.angular.z = (
            self.output_smoothing_coeff * self.command_old_twist.angular.z
            + (1 - self.output_smoothing_coeff) * control_command.angular.z
        )
        self.command_old_twist = smoothed_command
        return smoothed_command

    # --------------------------- Control -------------------------
    def calculate_twist(self):
        if self.current_pose is None:
            rospy.logwarn("No current pose received yet.")
            return
        path = self.path
        if path is None or len(path.poses) < 2:
            rospy.logwarn_throttle(5.0, "No valid path to follow.")
            return

        goal_pose = path.poses[self.current_index]
        desired_yaw = self.quaternion_to_yaw(goal_pose.pose.orientation)

        direction_xy_norm = None
        if self.calculate_path_direction:
            rospy.logwarn_once("Calculating yaw from path direction.")
            direction_xy_norm = self.get_direction()
            direction_norm = np.linalg.norm(direction_xy_norm)
            if direction_norm > 1e-6:
                desired_yaw = math.atan2(direction_xy_norm[1], direction_xy_norm[0])

        # Yaw PID
        current_yaw = self.quaternion_to_yaw(self.current_pose.pose.orientation)
        yaw_error = self.wrap_to_pi(desired_yaw - current_yaw)
        if yaw_error > math.pi / 2:
            rospy.logwarn_throttle(5.0, "Large yaw error detected: %.2f radians. desired_yaw=%.2f, current_yaw=%.2f",
                                  yaw_error, desired_yaw, current_yaw)

        velocity_scale = self.velocity_override
        proportional = yaw_error * self.kp_yaw
        derivative = (yaw_error - self.prev_error_yaw) * self.kd_yaw
        integral_term = self.integral_yaw * self.ki_yaw
        omega_z = (proportional + integral_term + derivative) * velocity_scale

        self.integral_yaw += yaw_error * velocity_scale
        self.prev_error_yaw = yaw_error

        control_command = Twist()
        control_command.linear.x = 0.0
        control_command.linear.y = 0.0
        control_command.linear.z = 0.0
        control_command.angular.z = omega_z

        control_command_smoothed = self.smooth_output(control_command)
        self.pub_ur_velocity_world.publish(control_command_smoothed)

    @staticmethod
    def _parse_initial_path_index(raw_value):
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, f"Invalid initial_path_index value '{raw_value}', ignoring.")
            return None
        return idx if idx >= 0 else None


def main():
    rospy.init_node("ur_path_yaw_controller")
    DirectionYawController()
    rospy.spin()


if __name__ == "__main__":
    main()
