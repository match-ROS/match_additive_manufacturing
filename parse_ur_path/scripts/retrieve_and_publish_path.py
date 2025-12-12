#! /usr/bin/env python3
import sys
import os
import rospy
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path
import tf.transformations as tf
import math
from additive_manufacturing_msgs.msg import Vector3Array

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir+ "/component/rectangleRoundedCorners")


# Import mir_path to retrieve mirX and mirY
# from path import ur_path
from print_path import xTCP
from print_path import yTCP
from print_path import zTCP
from print_path import t

class PathTransfomer:
    def __init__(self):
        rospy.init_node('path_transformer')
    
        # Publishers for the original and transformed paths
        self.original_pub = rospy.Publisher('/ur_path_original', Path, queue_size=10)
        self.transformed_pub = rospy.Publisher('/ur_path_transformed', Path, queue_size=10)
        self.normals_pub = rospy.Publisher('/ur_path_normals', Vector3Array, queue_size=10)
        self.start_index = 10

        # Retrieve the original path
        self.x_coords = xTCP.xTCP()
        self.y_coords = yTCP.yTCP()
        self.z_coords = zTCP.zTCP()
        #try:
        self.timestamps = [rospy.Time.from_sec(t) for t in t.t()]
        #except AttributeError:
        #    self.timestamps = None

        
        # Get transformation parameters from ROS params
        self.tx = rospy.get_param('~tx', 0.0)
        self.ty = rospy.get_param('~ty', 0.0)
        self.tz = rospy.get_param('~tz', 0.0)
        self.rx = rospy.get_param('~rx', 0.0)
        self.ry = rospy.get_param('~ry', 0.0)
        self.rz = rospy.get_param('~rz', 0.0)

        # Prepare Path messages
        self.original_path = Path()
        self.transformed_path = Path()
        
        # Set frame IDs for paths
        self.original_path.header.frame_id = "map"  # Use an appropriate frame
        self.transformed_path.header.frame_id = "map" # TODO: set this via parameter
        
        # normals to the path
        self.normals = None
    
    def compute_normals(self):
        normals = []
        x_coords, y_coords = self.x_coords, self.y_coords
        centroid = (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))

        rospy.logwarn("computing normals")
        
        for i in range(self.start_index, len(x_coords)-1):
            # Compute the normal vector to the path at each point
            dx = x_coords[i+1] - x_coords[i-1]
            dy = y_coords[i+1] - y_coords[i-1]
            norm = math.sqrt(dx**2 + dy**2)
            normal = (dy/norm, -dx/norm)
             # Create a vector from the current point to the centroid
            # vec_to_centroid = (centroid[0] - x_coords[i], centroid[1] - y_coords[i])
            
            # # If the dot product is positive, the normal is pointing toward the centroid,
            # # else flip it to make it point inward.
            # if normal[0]*vec_to_centroid[0] + normal[1]*vec_to_centroid[1] < 0:
            #     normal = (-normal[0], -normal[1])

            normals.append(normal) # TODO: Richtung?
        normals.append(normals[-1])  # to have the same length as the path
        self.normals = Vector3Array()
        self.normals.vectors = [Vector3(x=n[0], y=n[1], z=0) for n in normals]
        return normals

    def apply_transformation(self, x_coords, y_coords, z_coords, tx, ty, tz, rx, ry, rz, timestamps=None):
        if timestamps is None:
            timestamps = [rospy.Time.from_sec(i*0.1) for i in range(len(x_coords))]
        transformed_poses = []

        # Convert rotation from Euler angles to a quaternion
        quaternion = tf.quaternion_from_euler(rx, ry, rz)
        
        for i in range(self.start_index, len(x_coords)-1):
            pose_stamped = PoseStamped()
            R = tf.quaternion_matrix(quaternion)[:3, :3]

            # Original position + translation
            pose_stamped.pose.position.x = x_coords[i] + R[0, 0] * tx + R[0, 1] * ty + R[0, 2] * tz
            pose_stamped.pose.position.y = y_coords[i] + R[1, 0] * tx + R[1, 1] * ty + R[1, 2] * tz
            pose_stamped.pose.position.z = z_coords[i] + R[2, 0] * tx + R[2, 1] * ty + R[2, 2] * tz
            # the path should always face towards the next point
            orientation = math.atan2(y_coords[i+1] - y_coords[i], x_coords[i+1] - x_coords[i])
            q = tf.quaternion_from_euler(0, 0, orientation)

            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]
            
            # Set the current timestamp
            pose_stamped.header.stamp = timestamps[i]
            pose_stamped.header.frame_id = "map"  # Use an appropriate frame

            transformed_poses.append(pose_stamped)
        
        return transformed_poses

        
    def create_paths(self):    
        # Fill original Path message
        for i in range(self.start_index,len(self.x_coords)-1):
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = self.x_coords[i]
            pose_stamped.pose.position.y = self.y_coords[i]
            pose_stamped.pose.position.z = self.z_coords[i]  
            
            orientation = math.atan2(self.y_coords[i+1] - self.y_coords[i], self.x_coords[i+1] - self.x_coords[i])
            q = tf.quaternion_from_euler(0, 0, orientation)
            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]

            # Set the current timestamp
            if self.timestamps is not None:
                pose_stamped.header.stamp = self.timestamps[i]
            else:
                pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "map"
            self.original_path.poses.append(pose_stamped)


        
        # Transform and fill transformed Path message
        self.transformed_path.poses = self.apply_transformation(self.x_coords, self.y_coords, self.z_coords, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz, self.timestamps)
        
    def publish(self):  
    
        # Update headers' timestamps
        self.original_path.header.stamp = rospy.Time.now()
        self.transformed_path.header.stamp = rospy.Time.now()
        self.normals.header.stamp = rospy.Time.now()
        
        # Publish the original and transformed paths
        self.original_pub.publish(self.original_path)
        self.transformed_pub.publish(self.transformed_path)
        self.normals_pub.publish(self.normals)
            
    def run(self):
        self.create_paths()
        self.compute_normals()
        
        rate = rospy.Rate(0.1)  # Publish at 0.1 Hz
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        path_transformer = PathTransfomer()
        path_transformer.run()
    except rospy.ROSInterruptException:
        pass
