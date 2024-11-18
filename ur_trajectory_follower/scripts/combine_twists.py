#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

class TwistCombiner:
    def __init__(self):
        rospy.init_node('twist_combiner', anonymous=True)
        
        # Get the list of twist topics from the parameter server
        twist_topics = rospy.get_param('~twist_topics', [])
        rospy.loginfo('Subscribing to Twist topics: %s', twist_topics)
        
        # Initialize twist messages dictionary
        self.twists = {topic: Twist() for topic in twist_topics}
        
        # Create subscribers for each twist topic
        self.subscribers = [rospy.Subscriber(topic, Twist, self.create_callback(topic)) for topic in twist_topics]
        
        # Publisher for the combined twist
        self.combined_twist_pub = rospy.Publisher('/combined_twist', Twist, queue_size=10)
        
        # Set the publish rate
        self.rate = rospy.Rate(10)  # 10 Hz

    def create_callback(self, topic):
        def callback(msg):
            self.twists[topic] = msg
            self.publish_combined_twist()
        return callback

    def publish_combined_twist(self):
        combined_twist = Twist()
        
        for twist in self.twists.values():
            combined_twist.linear.x += twist.linear.x
            combined_twist.linear.y += twist.linear.y
            combined_twist.linear.z += twist.linear.z
            combined_twist.angular.x += twist.angular.x
            combined_twist.angular.y += twist.angular.y
            combined_twist.angular.z += twist.angular.z
        
        self.combined_twist_pub.publish(combined_twist)

if __name__ == '__main__':
    try:
        combiner = TwistCombiner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass