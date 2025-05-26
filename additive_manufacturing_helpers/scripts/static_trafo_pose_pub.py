#!/usr/bin/env python3
import rospy
import numpy as np
import tf
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped

class PoseTransformer(object):
    def __init__(self):
        rospy.init_node("ur_base_ideal", anonymous=False)

        # Get configurable parameters.
        self.source_frame = rospy.get_param("~source_frame", "UR10_r/base_link")
        self.target_frame = rospy.get_param("~target_frame", "UR10_r/base_ideal")
        self.input_topic = rospy.get_param("~input_topic", "UR10_r/tcp_pose")
        self.output_topic = rospy.get_param("~output_topic", "UR10_r/tcp_pose_base_ideal")

        # Initialize a tf listener.
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("Waiting for transform from {} to {}.".format(self.source_frame, self.target_frame))
        self.tf_listener.waitForTransform(self.target_frame, self.source_frame, rospy.Time(0), rospy.Duration(10.0))
        try:
            # Wait for up to 10 seconds for the transform.
            self.tf_listener.waitForTransform(self.target_frame, self.source_frame, rospy.Time(0), rospy.Duration(10.0))
            (trans, rot) = self.tf_listener.lookupTransform(self.target_frame, self.source_frame, rospy.Time(0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to get transform from {} to {}.".format(self.source_frame, self.target_frame))
            rospy.signal_shutdown("Transform lookup failed")
            return

        # Build the homogeneous transformation matrix for the static transform.
        self.T_static = tft.quaternion_matrix(rot)
        self.T_static[0:3, 3] = trans

        # Publisher for the transformed pose.
        self.pub_pose = rospy.Publisher(self.output_topic, PoseStamped, queue_size=10)

        # Subscriber for the raw pose.
        rospy.Subscriber(self.input_topic, PoseStamped, self.pose_callback)
        rospy.loginfo("PoseTransformer initialized. Transforming poses from {} to {}.".format(
            self.source_frame, self.target_frame))

    def transform_pose(self, pos, quat):
        # Create homogeneous transformation for the input pose.
        T_pose = tft.quaternion_matrix(quat)
        T_pose[0:3, 3] = pos

        # Apply the static transform.
        T_transformed = np.dot(self.T_static, T_pose)

        # Extract transformed position and orientation.
        pos_trans = T_transformed[0:3, 3]
        quat_trans = tft.quaternion_from_matrix(T_transformed)

        return pos_trans, quat_trans

    def pose_callback(self, msg):

        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

        pos_trans, quat_trans = self.transform_pose(pos, quat)

        msg.header.frame_id = self.target_frame
        msg.pose.position.x = pos_trans[0]
        msg.pose.position.y = pos_trans[1]
        msg.pose.position.z = pos_trans[2]
        msg.pose.orientation.x = quat_trans[0]
        msg.pose.orientation.y = quat_trans[1]
        msg.pose.orientation.z = quat_trans[2]
        msg.pose.orientation.w = quat_trans[3]

        self.pub_pose.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    transformer = PoseTransformer()
    transformer.run()
