#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, TwistStamped

class TwistRepublisherNode:
    def __init__(self):
        # Retrieve parameters with defaults
        self.input_topic = rospy.get_param('~input_topic', '/ur_twist_world_in_mir')
        self.output_topic = rospy.get_param('~output_topic', '/output_twist_stamped')
        self.frame_id = rospy.get_param('~frame_id', 'mur620a/base_link')

        # Initialize publisher and subscriber
        self.pub = rospy.Publisher(self.output_topic, TwistStamped, queue_size=10)
        rospy.Subscriber(self.input_topic, Twist, self.twist_callback)

    def twist_callback(self, twist_msg):
        # Create TwistStamped message
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = rospy.Time.now()
        twist_stamped.header.frame_id = self.frame_id
        twist_stamped.twist = twist_msg
        
        # Publish
        self.pub.publish(twist_stamped)

if __name__ == '__main__':
    rospy.init_node('republish_twist_stamped_node')
    node = TwistRepublisherNode()
    rospy.loginfo("Republisher node started")
    rospy.spin()