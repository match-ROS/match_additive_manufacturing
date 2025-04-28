#!/usr/bin/env python
import rospy
import math
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

def odom_callback(msg):
    # Extract the robot's pose from the odometry message.
    # The orientation (as a quaternion) is in the map frame.
    q = msg.pose.pose.orientation
    # Convert quaternion to Euler angles to obtain yaw.
    (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    
    # Prepare the transformation matrix from map to base_footprint
    # For a rotation around z by angle θ, the matrix to rotate a vector
    # from the map frame to the robot's frame is R(-θ), which is:
    #      [[ cos(θ), sin(θ)],
    #       [-sin(θ), cos(θ)]]
    #
    # Transform the linear velocity vector from the twist message.
    orig_linear_x = msg.twist.twist.linear.x
    orig_linear_y = msg.twist.twist.linear.y
    
    # Rotate the vector using R(-yaw):
    new_linear_x = math.cos(yaw) * orig_linear_x + math.sin(yaw) * orig_linear_y
    new_linear_y = -math.sin(yaw) * orig_linear_x + math.cos(yaw) * orig_linear_y

    # Create a new Odometry message to publish.
    new_msg = Odometry()
    new_msg.header = msg.header
    
    # Optionally, update the header frame_id to indicate the twist is now in the base_footprint frame.
    new_msg.child_frame_id = "mur620a/base_footprint"
    new_msg.pose = msg.pose  # Pose remains unchanged (still in map frame)
    
    # Copy twist values and update the linear part.
    # Note: Angular velocity transformation is usually not needed if only z is used.
    new_msg.twist.twist.linear.x = new_linear_x
    new_msg.twist.twist.linear.y = new_linear_y
    new_msg.twist.twist.linear.z = msg.twist.twist.linear.z  # Generally remains the same.
    
    # For angular velocities, if the motion is mostly planar (rotation around z),
    # the z-component remains unchanged. If there are components in x, y (rare for ground robots),
    # you would have to transform them appropriately.
    new_msg.twist.twist.angular = msg.twist.twist.angular

    pub.publish(new_msg)

if __name__ == '__main__':
    rospy.init_node('twist_transformer_node')
    # Publisher for the new odometry message with twist in the base_footprint frame.
    pub = rospy.Publisher('/mur620a/ground_truth_rotated', Odometry, queue_size=10)
    # Subscriber to the original ground truth odometry.
    sub = rospy.Subscriber('/mur620a/ground_truth', Odometry, odom_callback)
    
    rospy.loginfo("Twist transformer node started, transforming twist from map to mur620a/base_footprint frame.")
    rospy.spin()
