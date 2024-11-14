#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import tf.transformations as tr

class MoveToFirstPathPoint:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('move_to_first_path_point', anonymous=True)
        
        # load paramaters
        self.robot_name = rospy.get_param('~robot_name', 'mur620a')
        self.path_topic = rospy.get_param('~path_topic', '/mir_path')

       
        # Action client for 'move_base'
        self.move_base_client = actionlib.SimpleActionClient(self.robot_name + '/move_base', MoveBaseAction)
        
        # Wait for the action server to be available
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")

        # Subscriber to 'mir_path' topic
        self.path_sub = rospy.Subscriber(self.path_topic, Path, self.path_callback)

    def path_callback(self, path_msg):
        # Check if the path contains at least one pose
        if len(path_msg.poses) == 0:
            rospy.logwarn("Received an empty path!")
            return
        
        # Extract the first pose from the path
        first_pose = path_msg.poses[0]
        rospy.loginfo(f"Moving to first pose: {first_pose.pose}")

        # Create and send the goal
        goal = MoveBaseGoal()
        goal.target_pose = first_pose  # Assign the first pose as the target
        goal.target_pose.pose.position.x = first_pose.pose.position.x #- 0.8

        self.move_base_client.send_goal(goal)
        
        # Optional: Wait for the result (blocking call)
        success = self.move_base_client.wait_for_result()
        if success:
            rospy.loginfo("Successfully reached the first pose!")
            # shutdown the node
            rospy.signal_shutdown("Successfully reached the first pose!")
        else:
            rospy.logwarn("Failed to reach the first pose.")

if __name__ == '__main__':
    try:
        mover = MoveToFirstPathPoint()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
