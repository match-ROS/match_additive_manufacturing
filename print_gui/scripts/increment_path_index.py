#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Vector3
from additive_manufacturing_msgs.msg import Vector3Array
from nav_msgs.msg import Path
from std_msgs.msg import Int32
def increment_path_index():
    # Initialize the ROS node
    rospy.init_node('increment_path_index', anonymous=True)

    path_index_topic = rospy.get_param('~path_index_topic', '/path_index')
    next_goal_topic = rospy.get_param('~next_goal_topic', '/next_goal')
    normal_topic = rospy.get_param('~normal_topic', '/normal_vector')
    initial_path_index = rospy.get_param('~initial_path_index', 0)
    path_topic = rospy.get_param('~path_topic', '/ur_path_transformed')
    normals_topic = rospy.get_param('~normals_topic', '/ur_path_normals')
    publish_rate_hz = rospy.get_param('~publish_rate', 10.0)

    # Create publishers
    index_pub = rospy.Publisher(path_index_topic, Int32, queue_size=10, latch=True)
    goal_pose_pub = rospy.Publisher(next_goal_topic, PoseStamped, queue_size=10, latch=True)
    normal_pub = rospy.Publisher(normal_topic, Vector3, queue_size=10, latch=True)

    # Get the path and normals from the configured topics
    path_msg = rospy.wait_for_message(path_topic, Path)
    path_length = len(path_msg.poses)
    if path_length == 0:
        rospy.logerr("Received empty path. Shutting down.")
        rospy.signal_shutdown("Empty path")
        return

    normals_msg = rospy.wait_for_message(normals_topic, Vector3Array)
    if len(normals_msg.vectors) == 0:
        rospy.logerr("Received empty normals array. Shutting down.")
        rospy.signal_shutdown("Empty normals")
        return

    if initial_path_index < 0:
        rospy.logwarn("Initial path index is less than 0. Setting to 0.")
        initial_path_index = 0
    if initial_path_index >= path_length:
        rospy.logwarn("Initial path index exceeds path length. Clamping to last waypoint.")
        initial_path_index = path_length - 1

    # Set the rate at which to publish messages
    rate = rospy.Rate(publish_rate_hz)

    # Initialize the path index
    path_index = initial_path_index

    while not rospy.is_shutdown():
        # Increment the path index
        if path_index < path_length - 1:
            path_index += 1

        # Create a message with the current path index
        msg = Int32()
        msg.data = path_index

        # Publish the index, next goal pose, and normal vector
        index_pub.publish(msg)
        goal_pose_pub.publish(path_msg.poses[path_index])

        normal_index = max(0, min(path_index - 1, len(normals_msg.vectors) - 1))
        normal_pub.publish(normals_msg.vectors[normal_index])

        # Log the published path index
        #rospy.loginfo(f"Published path index: {path_index}")

        # Sleep for the specified rate
        rate.sleep()


if __name__ == '__main__':
    try:
        increment_path_index()
    except rospy.ROSInterruptException:
        pass