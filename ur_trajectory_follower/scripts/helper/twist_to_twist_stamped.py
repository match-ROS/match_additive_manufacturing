import rospy
from geometry_msgs.msg import Twist, TwistStamped

#!/usr/bin/env python

def twist_callback(twist_msg):
    twist_stamped = TwistStamped()
    twist_stamped.header.stamp = rospy.Time.now()
    twist_stamped.header.frame_id = frame_id
    twist_stamped.twist = twist_msg
    pub.publish(twist_stamped)

if __name__ == '__main__':
    rospy.init_node('twist_to_twist_stamped')

    # Get the frame from parameter server, default to "base_link" if not set.
    frame_id = rospy.get_param('~frame_id', 'mur620/base_footprint')

    # Get topic names from parameter server or use defaults.
    input_topic = rospy.get_param('~input_topic', '/ur_cmd')
    output_topic = rospy.get_param('~output_topic', '/ur_cmd_stamped')

    pub = rospy.Publisher(output_topic, TwistStamped, queue_size=10)
    rospy.Subscriber(input_topic, Twist, twist_callback)

    rospy.spin()