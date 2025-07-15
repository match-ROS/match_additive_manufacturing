#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import math
import tf.transformations as tr

class MoveFollowerToStartPose:
    def __init__(self):
        rospy.init_node('move_follower_tp_start_pose', anonymous=True)

        # Parameter laden
        self.leader_pose_topic = rospy.get_param('~leader_pose_topic', '/mur620c/mir_pose_simple')
        self.relative_pose = rospy.get_param('~relative_pose', {'x': 0.0, 'y': -2.0, 'phi': 0.0})
        self.robot_name = rospy.get_param('~robot_name', 'mur620b')

        self.cmd_vel_topic = f'/{self.robot_name}/cmd_vel'
        self.move_base_topic = f'/{self.robot_name}/move_base'

        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.move_base_client = actionlib.SimpleActionClient(self.move_base_topic, MoveBaseAction)
        self.move_base_client.wait_for_server()

        self.leader_pose_sub = rospy.Subscriber(self.leader_pose_topic, Pose, self.leader_pose_callback)

        rospy.loginfo("Follower ready. Waiting for leader pose...")

    def leader_pose_callback(self, pose_msg):
        leader_x = pose_msg.position.x
        leader_y = pose_msg.position.y
        _, _, leader_yaw = tr.euler_from_quaternion([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ])

        dx_rel = self.relative_pose['x']
        dy_rel = self.relative_pose['y']
        dphi_rel = self.relative_pose['phi']

        # Transformation der relativen Pose in Weltkoordinaten
        dx_world = dx_rel * math.cos(leader_yaw) - dy_rel * math.sin(leader_yaw)
        dy_world = dx_rel * math.sin(leader_yaw) + dy_rel * math.cos(leader_yaw)
        target_x = leader_x + dx_world
        target_y = leader_y + dy_world
        target_yaw = leader_yaw + dphi_rel

        q = tr.quaternion_from_euler(0, 0, target_yaw)

        # Zielpose setzen
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        goal.target_pose.pose.orientation = Quaternion(*q)

        rospy.loginfo("Sending goal to move_base...")
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()

        rospy.loginfo("Reached goal. Starting fine alignment...")
        self.fine_align(target_yaw)

    def fine_align(self, target_yaw):
        rate = rospy.Rate(10)
        angle_diff = 100.0
        while abs(angle_diff) > 0.05 and not rospy.is_shutdown():
            current_pose = rospy.wait_for_message(f'/{self.robot_name}/mir_pose_simple', Pose)
            current_yaw = tr.euler_from_quaternion([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])[2]

            angle_diff = target_yaw - current_yaw
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize

            twist = Twist()
            twist.angular.z = 0.5 * angle_diff
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop rotation
        self.cmd_vel_pub.publish(Twist())
        rospy.loginfo("Fine alignment complete.")
        rospy.signal_shutdown("Done.")

if __name__ == '__main__':
    try:
        MoveFollowerToStartPose()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
