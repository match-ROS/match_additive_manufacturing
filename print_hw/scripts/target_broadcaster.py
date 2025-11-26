#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
import tf


class TargetPoseBroadcaster(object):
    def __init__(self):
        rospy.init_node("target_pose_broadcaster", anonymous=True)

        # Parameter
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.mir_child_frame = rospy.get_param("~mir_child_frame", "mir_target")
        self.modified_mir_child_frame = rospy.get_param("~modified_mir_child_frame", "mir_target_modified")
        self.ur_child_frame = rospy.get_param("~ur_child_frame", "ur_target")
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)  # Hz
        self.initial_path_index = rospy.get_param("~initial_path_index", 150)

        # interne Zustände
        self.mir_path = None          # nav_msgs/Path
        self.ur_path = None           # nav_msgs/Path
        self.current_index = self.initial_path_index     # int
        self.path_index_modified = self.initial_path_index  # int

        # Publisher
        self.mir_pub = rospy.Publisher("mir_target_pose", PoseStamped, queue_size=1)
        self.ur_pub = rospy.Publisher("ur_target_pose", PoseStamped, queue_size=1)

        # TF Broadcaster
        self.tf_br = tf.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber("/mir_path_original", Path, self.mir_path_cb, queue_size=1)
        rospy.Subscriber("/ur_path_original", Path, self.ur_path_cb, queue_size=1)
        rospy.Subscriber("/path_index", Int32, self.index_cb, queue_size=1)
        rospy.Subscriber("/path_index_modified", Int32, self.index_modified_cb, queue_size=1)

        # Timer für zyklisches Publizieren
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)

    def mir_path_cb(self, msg):
        self.mir_path = msg
        self.update_and_publish_once()

    def ur_path_cb(self, msg):
        self.ur_path = msg
        self.update_and_publish_once()

    def index_cb(self, msg):
        # bei jeder Änderung direkt aktualisieren
        if self.current_index != msg.data:
            self.current_index = msg.data
            self.update_and_publish_once()
        
    def index_modified_cb(self, msg):
        # bei jeder Änderung direkt aktualisieren
        if self.path_index_modified != msg.data:
            self.path_index_modified = msg.data
            self.update_and_publish_modified_once()

    def timer_cb(self, event):
        # zyklisches Publizieren der aktuellen Zielpose
        self.update_and_publish_once(publish_if_same_index=True)

    def get_pose_from_path(self, path_msg, index):
        if path_msg is None:
            return None
        if not path_msg.poses:
            return None
        if index is None:
            return None

        # Index begrenzen
        idx = max(0, min(index, len(path_msg.poses) - 1))
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose = path_msg.poses[idx].pose
        return pose

    def publish_pose_and_tf(self, pose_msg, child_frame_id):
        if pose_msg is None:
            return

        # Pose publishen
        self.mir_pub.publish(pose_msg) if child_frame_id == self.mir_child_frame else self.ur_pub.publish(pose_msg)

        # TF broadcasten
        t = pose_msg.pose.position
        q = pose_msg.pose.orientation
        self.tf_br.sendTransform(
            (t.x, t.y, t.z),
            (q.x, q.y, q.z, q.w),
            pose_msg.header.stamp,
            child_frame_id,
            self.frame_id
        )

    def update_and_publish_once(self, publish_if_same_index=False):
        # Wenn noch nichts initialisiert ist, nichts tun
        if self.current_index is None or self.mir_path is None or self.ur_path is None:
            return

        # Zielposen bestimmen
        mir_pose = self.get_pose_from_path(self.mir_path, self.current_index)
        ur_pose = self.get_pose_from_path(self.ur_path, self.current_index)

        # publish+TF
        if mir_pose is not None:
            self.publish_pose_and_tf(mir_pose, self.mir_child_frame)
        if ur_pose is not None:
            self.publish_pose_and_tf(ur_pose, self.ur_child_frame)

    def update_and_publish_modified_once(self):
        # Wenn noch nichts initialisiert ist, nichts tun
        if self.path_index_modified is None or self.mir_path is None or self.ur_path is None:
            return

        # Zielposen bestimmen
        mir_pose = self.get_pose_from_path(self.mir_path, self.path_index_modified)
        ur_pose = self.get_pose_from_path(self.ur_path, self.path_index_modified)

        # publish+TF
        if mir_pose is not None:
            self.publish_pose_and_tf(mir_pose, self.modified_mir_child_frame)



if __name__ == "__main__":
    try:
        node = TargetPoseBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
