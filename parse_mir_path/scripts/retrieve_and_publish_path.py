#! /usr/bin/env python3
import sys
import os
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf.transformations as tf
import math


# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
start_index = 10

print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir+ "/component/rectangleRoundedCorners")

# Import mir_path to retrieve mirX and mirY
from print_path import xMIR
from print_path import yMIR
from print_path import nL
from print_path import xVecMIRx
from print_path import xVecMIRy

def apply_transformation(x_coords, y_coords, tx, ty, tz, rx, ry, rz):
    transformed_poses = []

    # Convert rotation from Euler angles to a quaternion
    quaternion = tf.quaternion_from_euler(rx, ry, rz)
     
    for i in range(start_index, len(x_coords)-1):
        pose_stamped = PoseStamped()
        R = tf.quaternion_matrix(quaternion)[:3, :3]

        # Original position + translation
        pose_stamped.pose.position.x = x_coords[i] + R[0, 0] * tx + R[0, 1] * ty + R[0, 2] * tz
        pose_stamped.pose.position.y = y_coords[i] + R[1, 0] * tx + R[1, 1] * ty + R[1, 2] * tz
        pose_stamped.pose.position.z = tz
        # the path should always face towards the next point
        orientation = math.atan2(y_coords[i+1] - y_coords[i], x_coords[i+1] - x_coords[i])
        q = tf.quaternion_from_euler(0, 0, orientation)

        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        
        # Set the current timestamp
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"  # Use an appropriate frame

        transformed_poses.append(pose_stamped)
    
    return transformed_poses

def publish_paths():
    rospy.init_node('path_transformer')
    
    # Publishers for the original and transformed paths
    original_pub = rospy.Publisher('/mir_path_original', Path, queue_size=10)
    transformed_pub = rospy.Publisher('/mir_path_transformed', Path, queue_size=10)
    
    # Retrieve the original path
    x_coords = xMIR.xMIR() 
    y_coords = yMIR.yMIR()
    orientation_vector_x = xVecMIRx.xVecMIRx()
    orientation_vector_y = xVecMIRy.xVecMIRy()
    layer_number = nL.nL()
    
    # Get transformation parameters from ROS params
    tx = rospy.get_param('~tx', 0.0)
    ty = rospy.get_param('~ty', 0.0)
    tz = rospy.get_param('~tz', 0.0)
    rx = rospy.get_param('~rx', 0.0)
    ry = rospy.get_param('~ry', 0.0)
    rz = rospy.get_param('~rz', 0.0)

    # Prepare Path messages
    original_path = Path()
    transformed_path = Path()
    
    # Set frame IDs for paths
    original_path.header.frame_id = "map"  # Use an appropriate frame
    transformed_path.header.frame_id = "map"
    
    # Fill original Path message
    for i in range(start_index, len(x_coords)-1):
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = x_coords[i]
        pose_stamped.pose.position.y = y_coords[i]
        pose_stamped.pose.position.z = layer_number[i]  # assuming z=0 for 2D path
        
        # the path should always face towards the next point
        #orientation = math.atan2(y_coords[i+1] - y_coords[i], x_coords[i+1] - x_coords[i])
        phi = math.atan2(orientation_vector_y[i], orientation_vector_x[i])
        q = tf.quaternion_from_euler(0, 0, phi)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"
        original_path.poses.append(pose_stamped)

    # find the center of a bounding box placed around the path
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    print(f"Center of the bounding box: ({center_x}, {center_y})")


    
    # Transform and fill transformed Path message
    transformed_path.poses = apply_transformation(x_coords, y_coords, tx, ty, tz, rx, ry, rz)
    
    set_metadata()

    rate = rospy.Rate(0.5)  # Publish at 1 Hz
    while not rospy.is_shutdown():
        # Update headers' timestamps
        original_path.header.stamp = rospy.Time.now()
        transformed_path.header.stamp = rospy.Time.now()
        
        # Publish the original and transformed paths
        original_pub.publish(original_path)
        transformed_pub.publish(transformed_path)
        rate.sleep()

def set_metadata():

    nL_ = nL.nL()

    # points per layer
    points_per_layer = [zero for zero in range(0,int(max(nL_)))]  
    print("Points per layer: ", points_per_layer)
    for i in range(len(nL_)):
        points_per_layer[int(nL_[i])-1] += 1
    #print("Points per layer: ", points_per_layer)

    rospy.set_param("/points_per_layer", points_per_layer)



if __name__ == '__main__':
    try:
        publish_paths()
    except rospy.ROSInterruptException:
        pass
