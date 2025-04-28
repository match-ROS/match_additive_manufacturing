#!/usr/bin/env python
import rospy
import math
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class VelocityTransformer(object):
    def __init__(self):
        # Initialize the node.
        rospy.init_node('velocity_transformer_node', anonymous=False)
        
        # Publisher for the velocity command in the map frame.
        self.pub_cmd_map = rospy.Publisher('/mur620a/ur_vel_map', Twist, queue_size=10)
        
        # Subscribe to the ground truth odometry to update the robot's current yaw.
        rospy.Subscriber('/mur620a/ground_truth', Odometry, self.ground_truth_callback)
        # Subscribe to the incoming velocity command in the base_footprint frame.
        rospy.Subscriber('/mur620a/ur_vel_local', Twist, self.cmd_vel_callback)
        
        # This variable stores the latest yaw (orientation) of the base_footprint in the map frame.
        self.current_yaw = None
        
        rospy.loginfo("Velocity Transformer Node started: converting cmd_vel from base_footprint to map frame.")

    def ground_truth_callback(self, odom_msg):
        """
        Callback to receive the ground truth odometry.
        Extracts the yaw from the robot's current orientation.
        """
        q = odom_msg.pose.pose.orientation
        # Convert quaternion to Euler angles; we only need the yaw.
        (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    def cmd_vel_callback(self, cmd_vel_msg):
        """
        Callback that transforms the velocity command from the base_footprint frame
        to the map frame using the current yaw.
        """
        if self.current_yaw is None:
            rospy.logwarn("No ground truth pose received yet; cannot transform velocity command.")
            return

        # Extract the linear velocity components in the base frame.
        base_vx = cmd_vel_msg.linear.x
        base_vy = cmd_vel_msg.linear.y

        # The rotation matrix R(yaw) transforms a vector from the base_footprint frame to the map frame:
        # [ cos(yaw)  -sin(yaw)]
        # [ sin(yaw)   cos(yaw)]
        map_vx = math.cos(self.current_yaw) * base_vx - math.sin(self.current_yaw) * base_vy
        map_vy = math.sin(self.current_yaw) * base_vx + math.cos(self.current_yaw) * base_vy

        map_wx = math.cos(self.current_yaw) * cmd_vel_msg.angular.x - math.sin(self.current_yaw) * cmd_vel_msg.angular.y
        map_wy = math.sin(self.current_yaw) * cmd_vel_msg.angular.x + math.cos(self.current_yaw) * cmd_vel_msg.angular.y
        map_wz = cmd_vel_msg.angular.z

        # Create and populate the transformed Twist message.
        transformed_cmd = Twist()
        transformed_cmd.linear.x = map_vx
        transformed_cmd.linear.y = map_vy
        transformed_cmd.linear.z = cmd_vel_msg.linear.z
        transformed_cmd.angular.x = map_wx
        transformed_cmd.angular.y = map_wy
        transformed_cmd.angular.z = map_wz

        # Publish the transformed velocity command.
        self.pub_cmd_map.publish(transformed_cmd)

if __name__ == '__main__':
    try:
        transformer = VelocityTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
