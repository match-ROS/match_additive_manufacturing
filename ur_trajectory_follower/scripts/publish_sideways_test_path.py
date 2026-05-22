#!/usr/bin/env python3
from typing import Any

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path
from additive_manufacturing_msgs.msg import Vector3Array
from std_msgs.msg import Bool


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback
    return vec / norm


def _build_orientation(nozzle_axis: np.ndarray, x_axis_hint: np.ndarray):
    z_axis = _normalize(nozzle_axis, np.array([0.0, 0.0, 1.0]))
    ref_axis = _normalize(x_axis_hint, np.array([1.0, 0.0, 0.0]))
    if abs(np.dot(ref_axis, z_axis)) > 0.95:
        ref_axis = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.95 else np.array([0.0, 1.0, 0.0])

    x_axis = _normalize(np.cross(ref_axis, z_axis), np.array([1.0, 0.0, 0.0]))
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0]))
    x_axis = _normalize(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0]))

    rotation = np.eye(4)
    rotation[0:3, 0] = x_axis
    rotation[0:3, 1] = y_axis
    rotation[0:3, 2] = z_axis
    quat = tft.quaternion_from_matrix(rotation)
    return quat, z_axis


class SidewaysTestPathPublisher:
    def __init__(self):
        rospy.init_node("ur_sideways_test_path_publisher")

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.path_topic = rospy.get_param("~path_topic", "/ur_path_transformed")
        self.original_path_topic = rospy.get_param("~original_path_topic", "/ur_path_original")
        self.normals_topic = rospy.get_param("~normals_topic", "/ur_path_normals")

        self.use_current_pose = rospy.get_param("~use_current_pose", True)
        self.current_pose_topic = rospy.get_param("~current_pose_topic", "/mur620a/UR10_r/global_tcp_pose")
        self.wait_for_home_pose = rospy.get_param("~wait_for_home_pose", True)
        self.home_pose_ready_topic = rospy.get_param("~home_pose_ready_topic", "/home_pose_ready")
        self.start_offset = np.array(rospy.get_param("~start_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.start_xyz = np.array(
            [
                rospy.get_param("~start_x", 51.6),
                rospy.get_param("~start_y", 39.7),
                rospy.get_param("~start_z", 0.5),
            ],
            dtype=float,
        )

        self.direction = np.array(rospy.get_param("~direction", [1.0, 0.0, 0.0]), dtype=float)
        self.nozzle_axis = np.array(rospy.get_param("~nozzle_axis", [0.0, 1.0, 0.0]), dtype=float)
        self.x_axis_hint = np.array(rospy.get_param("~x_axis_hint", [1.0, 0.0, 0.0]), dtype=float)

        self.path_length = _as_float(rospy.get_param("~path_length", 0.6), 0.6)
        self.num_points = _as_int(rospy.get_param("~num_points", 50), 50)
        self.time_step = _as_float(rospy.get_param("~time_step", 0.1), 0.1)
        self.publish_rate = _as_float(rospy.get_param("~publish_rate", 1.0), 1.0)

        if self.num_points < 2:
            rospy.logwarn("num_points must be >= 2; clamping to 2.")
            self.num_points = 2

        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.original_pub = rospy.Publisher(self.original_path_topic, Path, queue_size=1, latch=True)
        self.normals_pub = rospy.Publisher(self.normals_topic, Vector3Array, queue_size=1, latch=True)

        if self.wait_for_home_pose:
            rospy.loginfo("Waiting for home pose ready on %s", self.home_pose_ready_topic)
            rospy.wait_for_message(self.home_pose_ready_topic, Bool)
            rospy.loginfo("Home pose ready received; generating sideways test path.")

        start_point = self._resolve_start_point()
        self.path_msg, self.normals_msg = self._build_messages(start_point)
        rospy.loginfo(
            "Publishing sideways UR test path with nozzle axis %s on %s",
            self.nozzle_axis.tolist(),
            self.path_topic,
        )

    def _resolve_start_point(self) -> np.ndarray:
        if self.use_current_pose:
            rospy.loginfo("Waiting for current pose on %s", self.current_pose_topic)
            pose_msg = rospy.wait_for_message(self.current_pose_topic, PoseStamped)
            if pose_msg is None or pose_msg.pose is None:
                raise rospy.ROSException("Failed to read current pose.")
            start = np.array(
                [
                    pose_msg.pose.position.x,
                    pose_msg.pose.position.y,
                    pose_msg.pose.position.z,
                ],
                dtype=float,
            )
        else:
            start = self.start_xyz
        return start + self.start_offset

    def _build_messages(self, start_point: np.ndarray):
        direction = _normalize(self.direction, np.array([1.0, 0.0, 0.0]))
        step = self.path_length / max(self.num_points - 1, 1)
        quat, nozzle_axis = _build_orientation(self.nozzle_axis, self.x_axis_hint)

        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_poses = path_msg.poses
        assert path_poses is not None

        normals_msg = Vector3Array()
        normals_msg.header.frame_id = self.frame_id
        normal_vectors = normals_msg.vectors
        assert normal_vectors is not None

        start_time = rospy.Time.now()
        for i in range(self.num_points):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = start_time + rospy.Duration.from_sec(self.time_step * i)

            position = start_point + direction * (step * i)
            pose.pose.position.x = position[0]
            pose.pose.position.y = position[1]
            pose.pose.position.z = position[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_poses.append(pose)

            normal_vectors.append(
                Vector3(x=float(nozzle_axis[0]), y=float(nozzle_axis[1]), z=float(nozzle_axis[2]))
            )

        return path_msg, normals_msg

    def publish(self):
        now = rospy.Time.now()
        self.path_msg.header.stamp = now
        self.normals_msg.header.stamp = now
        self.path_pub.publish(self.path_msg)
        self.original_pub.publish(self.path_msg)
        self.normals_pub.publish(self.normals_msg)

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


if __name__ == "__main__":
    try:
        SidewaysTestPathPublisher().run()
    except rospy.ROSInterruptException:
        pass
