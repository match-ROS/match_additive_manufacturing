#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import math
import tf.transformations as tr
from geometry_msgs.msg import Twist

class MoveToFirstPathPoint:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('move_to_first_path_point', anonymous=True)

        # Load parameters
        robot_name = rospy.get_param('~robot_name', 'mur620a')
        robot_name = robot_name if isinstance(robot_name, str) else ''
        self.robot_name = "/" + robot_name if not robot_name.startswith('/') else robot_name
        raw_topic = rospy.get_param('~path_topic', 'mir_path_transformed')
        ns_param = rospy.get_param('~path_namespace', '')
        # Normalize namespace and topic to ensure namespaced subscription.
        ns_clean = ns_param.strip('/') if isinstance(ns_param, str) else ''
        # self.path_topic = raw_topic.lstrip('/') if isinstance(raw_topic, str) else str(raw_topic)
        self.path_topic = raw_topic if isinstance(raw_topic, str) else str(raw_topic)
        if ns_clean and not self.path_topic.startswith('/'):
            self.path_topic = f"/{ns_clean}/{self.path_topic}"
        elif not self.path_topic.startswith('/'):
            self.path_topic = f"/{self.path_topic}"
        self.initial_path_index = rospy.get_param('~initial_path_index', 0)
        self.robot_pose_topic = rospy.get_param('~robot_pose_topic', f'{self.robot_name}/mir_pose_simple')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', f'{self.robot_name}/cmd_vel')


        # Action client for 'move_base'
        self.move_base_client = actionlib.SimpleActionClient(self.robot_name + '/move_base', MoveBaseAction)

        # Wait for the action server to be available
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")
        rospy.wait_for_message(self.robot_pose_topic, Pose)
        rospy.loginfo("Received first robot pose message.")
        self.robot_pose_sub = rospy.Subscriber(self.robot_pose_topic, Pose, self.robot_pose_callback)
        # Start publisher for 'cmd_vel' topic
        self.twist_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        # Subscriber to 'mir_path' topic
        #self.path_sub = rospy.Subscriber(self.path_topic, Path, self.path_callback)
        rospy.loginfo(f"Waiting for path message on topic: {self.path_topic}")
        path = rospy.wait_for_message(self.path_topic, Path)
        self.path_callback(path)
        
        # wait for robot pose to be received
        

    def calculate_orientation(self, start, target):
        """Calculate the orientation quaternion pointing from start to target."""
        dx = target.pose.position.x - start.pose.position.x
        dy = target.pose.position.y - start.pose.position.y
        yaw = math.atan2(dy, dx)
        quaternion = tr.quaternion_from_euler(0, 0, yaw)
        return Quaternion(*quaternion)

    def path_callback(self, path_msg):
        # Check if the path contains at least two poses
        if len(path_msg.poses) < 2:
            rospy.logwarn("Path must contain at least two poses!")
            return

        # Extract the first pose from the path
        first_pose = path_msg.poses[self.initial_path_index]
        rospy.loginfo(f"Moving to first pose: {first_pose.pose}")

        # Calculate the target orientation based on the second point
        second_pose = path_msg.poses[self.initial_path_index+10]
        self.orientation = self.calculate_orientation(first_pose, second_pose)

        # Create and send the goal
        goal = MoveBaseGoal()
        goal.target_pose.header = first_pose.header
        goal.target_pose.pose.position = first_pose.pose.position
        goal.target_pose.pose.orientation = self.orientation

        self.move_base_client.send_goal(goal)

        while not rospy.is_shutdown():
            state = self.move_base_client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                success = True
                break
            elif state in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.PREEMPTED]:
                success = False
                break
            rospy.sleep(0.1)


        if success:
            #self.turn_robot_towards_path()
            rospy.loginfo("Successfully reached the first pose!")
            rospy.signal_shutdown("Successfully reached the first pose!")
        else:
            rospy.logwarn("Failed to reach the first pose.")

    def turn_robot_towards_path(self):
        twist = Twist()
        angle_diff = 100.0
        target_orientation = tr.euler_from_quaternion([
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w
        ])[2]
        while not rospy.is_shutdown() and angle_diff > 0.1:
            angle_diff = target_orientation - self.robot_orientation 
            if angle_diff > math.pi:
                angle_diff -= 2*math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2*math.pi
           
            twist.angular.z = angle_diff * 0.5
            self.twist_pub.publish(twist)
            rospy.sleep(0.1)
        twist.angular.z = 0.0
        self.twist_pub.publish(twist)
        rospy.sleep(1.0)
        rospy.signal_shutdown("Robot is now oriented towards the path.")


    def robot_pose_callback(self, pose_msg):
        self.robot_orientation = tr.euler_from_quaternion([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ])[2]
if __name__ == '__main__':
    try:
        mover = MoveToFirstPathPoint()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
