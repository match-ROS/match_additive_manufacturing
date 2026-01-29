#!/usr/bin/env python3

import math
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Int32

from helper.orthogonal_error_accumulator_core import (
    LayeredNearestNeighborAccumulator,
    build_reference_layer_points,
)


class OrthogonalErrorAccumulator:
    def __init__(self):
        self.layer_delta_z = float(rospy.get_param("~layer_delta_z", 0.01))
        self.reference_z = rospy.get_param("~reference_z", None)
        if self.reference_z is not None:
            self.reference_z = float(self.reference_z)

        self.nn_window = int(rospy.get_param("~nn_window", 50))
        self.bias_gain = float(rospy.get_param("~bias_gain", 1.0))
        self.bias_sign = float(rospy.get_param("~bias_sign", 1.0))
        self.normalize_normal = bool(rospy.get_param("~normalize_normal", True))

        self.path_topic = rospy.get_param("~path_topic", "/ur_path_transformed")
        self.goal_topic = rospy.get_param("~goal_topic", "/next_goal")
        self.normal_topic = rospy.get_param("~normal_topic", "/normal_vector")
        self.error_topic = rospy.get_param("~orthogonal_error_topic", "/orthogonal_error")
        self.path_index_topic = rospy.get_param("~path_index_topic", "/path_index")
        self.reset_topic = rospy.get_param("~reset_topic", "/orthogonal_error_accumulator/reset")

        self._latest_goal: Optional[PoseStamped] = None
        self._latest_normal: Optional[Vector3] = None
        self._latest_error: Optional[Twist] = None
        self._latest_index: Optional[int] = None

        self._reference_points = []
        self._accumulator: Optional[LayeredNearestNeighborAccumulator] = None

        self.bias_pub = rospy.Publisher("/orthogonal_error_bias", Twist, queue_size=10)
        self.corrected_pub = rospy.Publisher("/orthogonal_error_corrected", Twist, queue_size=10)
        self.ref_index_pub = rospy.Publisher("/orthogonal_error_reference_index", Int32, queue_size=10)
        self.mean_pub = rospy.Publisher("/orthogonal_error_bias_mean", Float32, queue_size=10)
        self.count_pub = rospy.Publisher("/orthogonal_error_sample_count", Int32, queue_size=10)

        self._load_reference_path()

        rospy.Subscriber(self.goal_topic, PoseStamped, self._goal_cb)
        rospy.Subscriber(self.normal_topic, Vector3, self._normal_cb)
        rospy.Subscriber(self.error_topic, Twist, self._error_cb)
        rospy.Subscriber(self.path_index_topic, Int32, self._index_cb)
        if self.reset_topic:
            rospy.Subscriber(self.reset_topic, Bool, self._reset_cb)

        update_rate = float(rospy.get_param("~update_rate", 30.0))
        if update_rate <= 0:
            update_rate = 30.0
        self._timer = rospy.Timer(rospy.Duration(1.0 / update_rate), self._tick)

    def _load_reference_path(self):
        path_msg = rospy.wait_for_message(self.path_topic, Path)
        points = [(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) for pose in path_msg.poses]
        self.reference_z, self._reference_points = build_reference_layer_points(
            points,
            delta_z=self.layer_delta_z,
            reference_z=self.reference_z,
            reference_layer_id=0,
        )
        self._accumulator = LayeredNearestNeighborAccumulator(self._reference_points, self.nn_window)
        rospy.loginfo(
            "OrthogonalErrorAccumulator: %d reference points at z=%.4f",
            len(self._reference_points),
            self.reference_z,
        )

    def _goal_cb(self, msg: PoseStamped):
        self._latest_goal = msg

    def _normal_cb(self, msg: Vector3):
        self._latest_normal = msg

    def _error_cb(self, msg: Twist):
        self._latest_error = msg

    def _index_cb(self, msg: Int32):
        self._latest_index = int(msg.data)

    def _reset_cb(self, msg: Bool):
        if msg.data:
            self._reset_accumulator()

    def _reset_accumulator(self):
        if self._accumulator is not None:
            self._accumulator.stats.clear()
            self._accumulator.last_ref_list_index = None
        rospy.loginfo("OrthogonalErrorAccumulator: reset accumulator")

    def _tick(self, _event):
        if self._latest_goal is None or self._latest_error is None or self._latest_normal is None:
            return
        if self._accumulator is None:
            return

        goal_pos = self._latest_goal.pose.position
        ref_index = self._accumulator.find_nearest(goal_pos.x, goal_pos.y)
        if ref_index is None:
            return

        normal = self._latest_normal
        normal_vec = (normal.x, normal.y, normal.z)
        normal_mag = math.sqrt(sum(v * v for v in normal_vec))
        if normal_mag < 1e-9:
            return

        if self.normalize_normal:
            normal_vec = tuple(v / normal_mag for v in normal_vec)

        err = self._latest_error.linear
        error_scalar = err.x * normal_vec[0] + err.y * normal_vec[1] + err.z * normal_vec[2]

        stats = self._accumulator.update(ref_index, error_scalar)
        bias_scalar = self.bias_sign * self.bias_gain * stats.mean

        bias_twist = Twist()
        bias_twist.linear.x = bias_scalar * normal_vec[0]
        bias_twist.linear.y = bias_scalar * normal_vec[1]
        bias_twist.linear.z = bias_scalar * normal_vec[2]

        corrected = Twist()
        corrected.linear.x = err.x + bias_twist.linear.x
        corrected.linear.y = err.y + bias_twist.linear.y
        corrected.linear.z = err.z + bias_twist.linear.z
        corrected.angular = self._latest_error.angular

        self.bias_pub.publish(bias_twist)
        self.corrected_pub.publish(corrected)
        self.ref_index_pub.publish(ref_index)
        self.mean_pub.publish(stats.mean)
        self.count_pub.publish(stats.count)


if __name__ == "__main__":
    rospy.init_node("orthogonal_error_accumulator", anonymous=True)
    node = OrthogonalErrorAccumulator()
    rospy.spin()
