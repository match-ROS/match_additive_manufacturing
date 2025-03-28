#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Twist, Vector3



class OrthogonalErrorCorrection:
    def __init__(self):        
        self.current_pose: PoseStamped = None
        self.goal_pose: PoseStamped = None
        self.normal: Vector3 = None
        
        rospy.Subscriber('~current_endeffector_pose', PoseStamped, self.current_pose_callback)
        rospy.Subscriber('~goal_endeffector_pose', PoseStamped, self.goal_pose_callback)
        rospy.Subscriber('~normal_vector', Vector3, self.normal_callback)
        
        self.twist_pub = rospy.Publisher('~orthogonal_twist', Twist, queue_size=10)
    
    def publish_twist(self):
        twist = self.calculate_twist()
        self.twist_pub.publish(twist)
    
    def current_pose_callback(self, msg: PoseStamped):
        self.current_pose = msg.pose
        self.publish_twist()

    def goal_pose_callback(self, msg: PoseStamped):
        self.goal_pose = msg.pose
        self.publish_twist()

    def normal_callback(self, msg):
        self.normal = msg
        self.publish_twist()

    def calculate_twist(self):
        twist = Twist()
        
        if self.current_pose and self.goal_pose and self.normal: # maybe also check max dt between the 3 messages
        # Calculate the error between current and target pose
            error_x = self.goal_pose.position.x - self.current_pose.position.x
            error_y = self.goal_pose.position.y - self.current_pose.position.y
            error_z = self.goal_pose.position.z - self.current_pose.position.z

            # Get the error along the normal vector
            error_normal = error_x * self.normal.x + error_y * self.normal.y + error_z * self.normal.z

            # Get the twist along the normal vector
            twist.linear.x = error_normal * self.normal.x
            twist.linear.y = error_normal * self.normal.y
            twist.linear.z = error_normal * self.normal.z

        return twist

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('orthogonal_error_correction', anonymous=True)
    
    node = OrthogonalErrorCorrection()
    node.run()
   
   