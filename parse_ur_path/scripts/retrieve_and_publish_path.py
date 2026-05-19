#! /usr/bin/env python3
import sys
import os
import importlib
import math
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path
import tf.transformations as tf
from additive_manufacturing_msgs.msg import Vector3Array

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_COMPONENT_NAME = "rectangleRoundedCorners"

class PathTransfomer:
    REQUIRED_MODULES = ("xTCP", "yTCP", "zTCP", "t")

    def __init__(self):
        rospy.init_node('retrieve_and_publish_ur_path', anonymous=False)
    
        self.component_name = rospy.get_param('~component_name', DEFAULT_COMPONENT_NAME)
        modules = self._load_component_modules(self.component_name)

        param_namespace = rospy.get_param('~path_namespace', "") if rospy.has_param('~path_namespace') else ""
        resolved_namespace = param_namespace.strip('/') if isinstance(param_namespace, str) else ""
        node_name = rospy.get_name()
        node_namespace = rospy.get_namespace()
        if not resolved_namespace:
            resolved_namespace = node_namespace.strip('/')
        rospy.loginfo(
            "UR path node '%s' using namespace '%s' (param '%s')",
            node_name,
            node_namespace,
            param_namespace if param_namespace else "/",
        )

        def _ns_topic(base: str) -> str:
            name = base.strip('/')
            ns = resolved_namespace
            return f"/{ns}/{name}" if ns else f"/{name}"

        # Publishers for the original and transformed paths
        self.original_pub = rospy.Publisher(_ns_topic('ur_path_original'), Path, queue_size=10)
        self.transformed_pub = rospy.Publisher(_ns_topic('ur_path_transformed'), Path, queue_size=10)
        self.normals_pub = rospy.Publisher(_ns_topic('ur_path_normals'), Vector3Array, queue_size=10)
        self.start_index = 10

        # Retrieve the original path
        self.x_coords = modules["xTCP"].xTCP()
        self.y_coords = modules["yTCP"].yTCP()
        self.z_coords = modules["zTCP"].zTCP()
        t_values = modules["t"].t()
        self.timestamps = [rospy.Time.from_sec(val) for val in t_values]

        
        # Get transformation parameters from ROS params
        self.tx = rospy.get_param('~tx', 0.0)
        self.ty = rospy.get_param('~ty', 0.0)
        self.tz = rospy.get_param('~tz', 0.0)
        self.rx = rospy.get_param('~rx', 0.0)
        self.ry = rospy.get_param('~ry', 0.0)
        self.rz = rospy.get_param('~rz', 0.0)
        self.flip_normal = rospy.get_param('~flip_normal', False)
        self.flip_tangent = rospy.get_param('~flip_tangent', False)
        self.tool_z_source = rospy.get_param('~tool_z_source', 'normal')
        self.tool_x_source = rospy.get_param('~tool_x_source', 'tangent')

        # Prepare Path messages
        self.original_path = Path()
        self.transformed_path = Path()
        
        # Set frame IDs for paths
        self.original_path.header.frame_id = "map"  # Use an appropriate frame
        self.transformed_path.header.frame_id = "map" # TODO: set this via parameter
        
        # normals to the path
        self.normals = None

    def _load_component_modules(self, component_name):
        component_path = os.path.join(parent_dir, "component", component_name)
        if not os.path.isdir(component_path):
            rospy.logfatal("Component folder '%s' not found at %s", component_name, component_path)
            raise rospy.ROSInitException("Invalid component folder")

        if component_path not in sys.path:
            sys.path.append(component_path)

        loaded = {}
        for module_name in self.REQUIRED_MODULES:
            try:
                loaded[module_name] = importlib.import_module(f"print_path.{module_name}")
            except ImportError as exc:
                rospy.logfatal("Failed to import %s from component '%s': %s", module_name, component_name, exc)
                raise

        rospy.loginfo("Loaded UR print_path component '%s'", component_name)
        return loaded
    
    def compute_normals(self):
        normals = []
        rospy.logwarn("computing normals")

        if self.transformed_path.poses:
            for pose_stamped in self.transformed_path.poses:
                orientation = pose_stamped.pose.orientation
                quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=float)
                quat_norm = np.linalg.norm(quat)
                if quat_norm < 1e-6:
                    normal = np.array([0.0, 0.0, 1.0])
                else:
                    quat /= quat_norm
                    rotation = tf.quaternion_matrix(quat)
                    normal = rotation[0:3, 2]
                normals.append(normal)
        else:
            x_coords = self.x_coords
            for i in range(self.start_index, len(x_coords) - 1):
                _, normal = self._compute_tangent_and_normal(i)
                normals.append(normal)
        if normals:
            normals.append(normals[-1])  # to have the same length as the path
        self.normals = Vector3Array()
        self.normals.vectors = [Vector3(x=n[0], y=n[1], z=n[2]) for n in normals]
        return normals

    def _compute_tangent_and_normal(self, i):
        last_i = len(self.x_coords) - 2
        if i <= self.start_index:
            i_prev = i
            i_next = min(i + 1, last_i)
        elif i >= last_i:
            i_prev = max(i - 1, self.start_index)
            i_next = i
        else:
            i_prev = i - 1
            i_next = i + 1

        dx = self.x_coords[i_next] - self.x_coords[i_prev]
        dy = self.y_coords[i_next] - self.y_coords[i_prev]
        dz = self.z_coords[i_next] - self.z_coords[i_prev]

        tangent = np.array([dx, dy, dz], dtype=float)
        if self.flip_tangent:
            tangent *= -1.0
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        normal = np.array([dy, -dx, 0.0], dtype=float)
        if self.flip_normal:
            normal *= -1.0
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = normal / normal_norm

        # Enforce orthogonality between tangent and normal.
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        return tangent, normal

    @staticmethod
    def _orientation_from_axes(x_axis, y_axis, z_axis):
        rotation = np.eye(4)
        rotation[0:3, 0] = x_axis
        rotation[0:3, 1] = y_axis
        rotation[0:3, 2] = z_axis
        return tf.quaternion_from_matrix(rotation)

    @staticmethod
    def _normalize_axis(vec, fallback):
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return fallback
        return vec / norm

    def apply_transformation(self, poses, tx, ty, tz, rx, ry, rz, timestamps=None):
        if timestamps is None:
            timestamps = [rospy.Time.from_sec(i*0.1) for i in range(len(poses))]
        transformed_poses = []

        # Convert rotation from Euler angles to a quaternion
        quaternion = tf.quaternion_from_euler(rx, ry, rz)
        
        for i in range(self.start_index, len(poses)-1):
            pose_stamped = PoseStamped()
            R = tf.quaternion_matrix(quaternion)[:3, :3]

            # Original position + translation
            pose_stamped.pose.position.x = poses[i].pose.position.x + R[0, 0] * tx + R[0, 1] * ty + R[0, 2] * tz
            pose_stamped.pose.position.y = poses[i].pose.position.y + R[1, 0] * tx + R[1, 1] * ty + R[1, 2] * tz
            pose_stamped.pose.position.z = poses[i].pose.position.z + R[2, 0] * tx + R[2, 1] * ty + R[2, 2] * tz
            # the path should always face towards the next point
            original_q = [poses[i].pose.orientation.x,
                          poses[i].pose.orientation.y,
                          poses[i].pose.orientation.z,
                          poses[i].pose.orientation.w]
            # Calculate desired rotation as a quaternion from rx, ry, rz
           
            if rx == 0.0 and ry == 0.0 and rz == 0.0:
                desired_quaternion = [0.0, 0.0, 0.0, 1.0]  # No rotation
            else:
                desired_quaternion = tf.quaternion_from_euler(rx, ry, rz)

            # Apply the desired quaternion to the orientation quaternion from poses
            q = tf.quaternion_multiply(original_q, desired_quaternion)

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

            tangent, normal = self._compute_tangent_and_normal(i)
            binormal = np.cross(tangent, normal)
            binormal = self._normalize_axis(binormal, np.array([0.0, 0.0, 1.0]))

            axis_map = {
                'tangent': tangent,
                'normal': normal,
                'binormal': binormal,
            }
            z_axis = axis_map.get(self.tool_z_source, normal)
            x_axis = axis_map.get(self.tool_x_source, tangent)

            z_axis = self._normalize_axis(z_axis, np.array([0.0, 0.0, 1.0]))
            x_axis = self._normalize_axis(x_axis, np.array([1.0, 0.0, 0.0]))

            if abs(np.dot(z_axis, x_axis)) > 0.95:
                fallback_x = axis_map['tangent'] if self.tool_z_source != 'tangent' else axis_map['normal']
                x_axis = self._normalize_axis(fallback_x, np.array([1.0, 0.0, 0.0]))

            y_axis = np.cross(z_axis, x_axis)
            y_axis = self._normalize_axis(y_axis, np.array([0.0, 0.0, 1.0]))
            x_axis = np.cross(y_axis, z_axis)
            x_axis = self._normalize_axis(x_axis, np.array([1.0, 0.0, 0.0]))
            q = self._orientation_from_axes(x_axis, y_axis, z_axis)
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
        self.transformed_path.poses = self.apply_transformation(self.original_path.poses, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz, self.timestamps)
        
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
        
        rate = rospy.Rate(1)  # Publish at 0.1 Hz
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        path_transformer = PathTransfomer()
        path_transformer.run()
    except rospy.ROSInterruptException:
        pass
