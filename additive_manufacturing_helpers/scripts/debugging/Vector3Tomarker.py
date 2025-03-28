#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Vector3, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class PoseVectorVisualizer:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pose_vector_listener')

        # Get parameters for topic names and frame_id
        self.pose_topic = rospy.get_param('~pose_topic', '/next_goal')
        self.vector_topic = rospy.get_param('~vector3_topic', '/normal_vector')
        self.frame_id = rospy.get_param('~frame_id', 'base_link')

        # Initialize variables for pose and vector
        self.pose = None
        self.vector = None

        # Publisher for Marker
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

        # Subscribers for PoseStamped and Vector3
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        rospy.Subscriber(self.vector_topic, Vector3, self.vector_callback)

        # Spin to keep the subscriber active
        rospy.spin()

    def pose_callback(self, msg):
        """Callback to receive PoseStamped messages."""
        self.pose = msg
        self.publish_marker()

    def vector_callback(self, msg):
        """Callback to receive Vector3 messages."""
        self.vector = msg
        self.publish_marker()

    def publish_marker(self):
        """Publish the marker when both pose and vector are received."""
        if self.pose is not None and self.vector is not None:
            # Create the Marker message
            marker = Marker()
            marker.header.frame_id = self.pose.header.frame_id  # Use the same frame_id as PoseStamped
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vector3_marker"
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Set the position and orientation of the marker
            marker.pose.position = self.pose.pose.position
            marker.pose.orientation.w = 1.0

            # Set the scale of the marker (size of the arrow)
            marker.scale.x = 0.1  # Length of the arrow in the X direction
            marker.scale.y = 0.15  # Width of the arrow (Y axis)
            marker.scale.z = 0.5  # Height of the arrow (Z axis)

            # Set the color of the marker (RGBA format)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green color
            
            marker.points.append(Point(0, 0, 0))
            marker.points.append(Point(self.vector.x, self.vector.y, self.vector.z))

            # Publish the marker
            self.marker_pub.publish(marker)


if __name__ == '__main__':
    try:
        PoseVectorVisualizer()
    except rospy.ROSInterruptException:
        pass
