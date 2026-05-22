#! /usr/bin/env python3

from typing import Any

import rospy
from geometry_msgs.msg import PoseStamped, Vector3
from additive_manufacturing_msgs.msg import Vector3Array
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Int32


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


class IncrementPathIndex:
    def __init__(self):
        rospy.init_node('increment_path_index', anonymous=True)

        self.path_index_topic: str = str(rospy.get_param('~path_index_topic', '/path_index'))
        self.next_goal_topic: str = str(rospy.get_param('~next_goal_topic', '/next_goal'))
        self.normal_topic: str = str(rospy.get_param('~normal_topic', '/normal_vector'))
        self.initial_path_index: int = _as_int(rospy.get_param('~initial_path_index', 0), 0)
        self.path_topic: str = str(rospy.get_param('~path_topic', '/ur_path_transformed'))
        self.normals_topic: str = str(rospy.get_param('~normals_topic', '/ur_path_normals'))
        self.publish_rate_hz: float = _as_float(rospy.get_param('~publish_rate', 10.0), 10.0)
        self.start_condition_topic: str = str(rospy.get_param('~start_condition_topic', '/start_condition'))
        self.wait_for_start_condition: bool = bool(rospy.get_param('~wait_for_start_condition', True))

        self.start_enabled: bool = not self.wait_for_start_condition

        self.index_pub = rospy.Publisher(self.path_index_topic, Int32, queue_size=10, latch=True)
        self.goal_pose_pub = rospy.Publisher(self.next_goal_topic, PoseStamped, queue_size=10, latch=True)
        self.normal_pub = rospy.Publisher(self.normal_topic, Vector3, queue_size=10, latch=True)

        path_msg = rospy.wait_for_message(self.path_topic, Path)
        if path_msg is None:
            raise rospy.ROSException('Received empty path message.')
        path_poses = path_msg.poses
        if path_poses is None or len(path_poses) == 0:
            raise rospy.ROSException('Received empty path.')

        normals_msg = rospy.wait_for_message(self.normals_topic, Vector3Array)
        if normals_msg is None:
            raise rospy.ROSException('Received empty normals message.')
        normal_vectors = normals_msg.vectors
        if normal_vectors is None or len(normal_vectors) == 0:
            raise rospy.ROSException('Received empty normals array.')

        self.path_msg = path_msg
        self.normals_msg = normals_msg

        self.path_length: int = len(path_poses)
        if self.initial_path_index < 0:
            rospy.logwarn('Initial path index is less than 0. Setting to 0.')
            self.initial_path_index = 0
        if self.initial_path_index >= self.path_length:
            rospy.logwarn('Initial path index exceeds path length. Clamping to last waypoint.')
            self.initial_path_index = self.path_length - 1

        rospy.Subscriber(self.start_condition_topic, Bool, self._start_condition_callback, queue_size=1)

    def _start_condition_callback(self, msg: Bool):
        if msg.data:
            self.start_enabled = True

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)
        path_index = self.initial_path_index
        path_poses = self.path_msg.poses
        normal_vectors = self.normals_msg.vectors
        assert path_poses is not None
        assert normal_vectors is not None

        while not rospy.is_shutdown():
            if self.start_enabled and path_index < self.path_length - 1:
                path_index += 1

            self.index_pub.publish(Int32(data=path_index))
            self.goal_pose_pub.publish(path_poses[path_index])

            normal_index = max(0, min(path_index - 1, len(normal_vectors) - 1))
            self.normal_pub.publish(normal_vectors[normal_index])

            rate.sleep()


if __name__ == '__main__':
    try:
        IncrementPathIndex().run()
    except rospy.ROSInterruptException:
        pass