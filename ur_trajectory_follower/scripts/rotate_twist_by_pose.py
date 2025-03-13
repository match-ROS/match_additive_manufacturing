#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import Pose, PoseStamped, Twist

from helper.ur_helper import rotate_twist_by_pose

class PubRotatedTwist():
    def __init__(self):
        self.world_T_base = Pose()
        self.twist = Twist()
        self.world_T_base_sub = rospy.Subscriber("world_T_base", PoseStamped, self.world_T_base_callback)
        self.twist_sub = rospy.Subscriber("twist_in", Twist, self.twist_callback)
        self.base_twist_pub = rospy.Publisher("base_twist", Twist, queue_size=1)

    def world_T_base_callback(self, msg: PoseStamped):
        self.world_T_base = msg.pose
        self.update_world_T_twist()

    def twist_callback(self, msg: Twist):
        self.twist = msg
        self.update_world_T_twist()

    def update_world_T_twist(self):
        base_twist = rotate_twist_by_pose(self.world_T_base, self.twist, True)
        self.base_twist_pub.publish(base_twist)

if __name__ == "__main__":
    rospy.init_node("pub_rotated_twist")
    pub_rotated_twist = PubRotatedTwist()
    rospy.spin()