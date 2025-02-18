import rospy
import rostopic

from geometry_msgs.msg import Pose, PoseStamped
from helper.ur_helper import transform_pose_by_pose

# Listens to Pose messages and calculates the kinematic chain to publish the total Pose
# in: world_T_base, base_T_ee; out: world_T_ee

class TotalPose:
    def __init__(self):
        # msg_class_base,_,_=rostopic.get_topic_class("world_T_base", blocking=True)
        self.world_T_base = PoseStamped()
        self.base_T_ee = PoseStamped()
        self.world_T_ee = PoseStamped()
        self.world_T_ee.header.frame_id = rospy.get_param("~world_frame", "map")

        self.world_T_base_sub = rospy.Subscriber("world_T_base", PoseStamped, self.world_T_base_callback)
        self.base_T_ee_sub = rospy.Subscriber("base_T_ee", PoseStamped, self.base_T_ee_callback)
        self.world_T_ee_pub = rospy.Publisher("world_T_ee", PoseStamped, queue_size=1)

    def world_T_base_callback(self, msg: PoseStamped):
        self.world_T_base = msg.pose
        self.world_T_ee.header.stamp = msg.header.stamp
        self.update_world_T_ee()

    def base_T_ee_callback(self, msg: PoseStamped):
        self.base_T_ee = msg
        self.world_T_ee.header.stamp = msg.header.stamp
        self.update_world_T_ee()

    def update_world_T_ee(self):
        self.world_T_ee.pose = transform_pose_by_pose(self.world_T_base, self.base_T_ee)
        self.world_T_ee_pub.publish(self.world_T_ee)

if __name__ == "__main__":
    rospy.init_node("total_pose")
    total_pose = TotalPose()
    rospy.spin()