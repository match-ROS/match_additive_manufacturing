#! /usr/bin/env python3
import sys
import os
import importlib
import math
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf.transformations as tf
from std_msgs.msg import Float32MultiArray



# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
start_index = 10

DEFAULT_COMPONENT_NAME = "rectangleRoundedCorners"
REQUIRED_MODULES = (
    "xMIR",
    "yMIR",
    "nL",
    "xVecMIRx",
    "xVecMIRy",
    "vxMIR",
    "vyMIR",
    "t",
)

OPTIONAL_MODULES = (
    "t_optimized",
    "mir_index_offset",
)


def _load_component_modules(component_name):
    component_path = os.path.join(parent_dir, "component", component_name)
    if not os.path.isdir(component_path):
        rospy.logfatal("Component folder '%s' not found at %s", component_name, component_path)
        raise rospy.ROSInitException("Invalid component folder")

    if component_path not in sys.path:
        sys.path.append(component_path)

    modules = {}
    for module_name in REQUIRED_MODULES:
        try:
            modules[module_name] = importlib.import_module(f"print_path.{module_name}")
        except ImportError as exc:
            rospy.logfatal("Failed to import %s from component '%s': %s", module_name, component_name, exc)
            raise
    for module_name in OPTIONAL_MODULES:
        try:
            modules[module_name] = importlib.import_module(f"print_path.{module_name}")
        except ImportError:
            rospy.logwarn("Optional module %s not found in component '%s'", module_name, component_name)

    rospy.loginfo("Loaded MiR print_path component '%s'", component_name)
    return modules


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
    rospy.init_node('retrieve_and_publish_mir_path', anonymous=False)

    param_namespace = rospy.get_param('~path_namespace', "") if rospy.has_param('~path_namespace') else ""
    normalized_ns = param_namespace.strip('/') if isinstance(param_namespace, str) else ""

    node_name = rospy.get_name()
    node_namespace = rospy.get_namespace()
    resolved_namespace = normalized_ns if normalized_ns else node_namespace.strip('/')
    rospy.loginfo(
        "MiR path node '%s' using namespace '%s' (param '%s')",
        node_name,
        node_namespace,
        normalized_ns or "/",
    )

    component_name = rospy.get_param('~component_name', DEFAULT_COMPONENT_NAME)
    modules = _load_component_modules(component_name)

    def _ns_topic(base: str) -> str:
        name = base.strip('/')
        ns = resolved_namespace
        return f"/{ns}/{name}" if ns else f"/{name}"

    # Publishers for the original and transformed paths
    original_pub = rospy.Publisher(_ns_topic('mir_path_original'), Path, queue_size=10)
    transformed_pub = rospy.Publisher(_ns_topic('mir_path_transformed'), Path, queue_size=10)
    velocity_pub = rospy.Publisher(_ns_topic('mir_path_velocity'), Path, queue_size=10)
    timestamps_pub = rospy.Publisher(_ns_topic('mir_path_timestamps'), Float32MultiArray, queue_size=10)
    index_offset_pub = rospy.Publisher(_ns_topic('mir_index_offset'), Float32MultiArray, queue_size=1)

    
    # Retrieve the original path
    x_coords = modules["xMIR"].xMIR()
    y_coords = modules["yMIR"].yMIR()
    vx_coords = modules["vxMIR"].vxMIR()
    vy_coords = modules["vyMIR"].vyMIR()
    orientation_vector_x = modules["xVecMIRx"].xVecMIRx()
    orientation_vector_y = modules["xVecMIRy"].xVecMIRy()
    layer_numbers = modules["nL"].nL()
    t_coords = modules["t"].t()
    
    try:
        mir_index_offset = modules["mir_index_offset"].mir_index_offset()
        #t_coords = modules["t_optimized"].t_optimized()
    except KeyError:
        mir_index_offset = [0.0]  # Default offset if module not found
        pass

    index_offset_msg = Float32MultiArray()
    index_offset_msg.data = list(mir_index_offset)

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
    velocity_path = Path()
    
    # Set frame IDs for paths
    original_path.header.frame_id = "map"  # Use an appropriate frame
    transformed_path.header.frame_id = "map"
    velocity_path.header.frame_id = "map"
    
    # Fill original Path message
    for i in range(start_index, len(x_coords)-1):
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = x_coords[i]
        pose_stamped.pose.position.y = y_coords[i]
        pose_stamped.pose.position.z = layer_numbers[i]
        
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

        # Fill velocity Path message
        vel_stamped = PoseStamped()
        vel_stamped.pose.position.x = vx_coords[i]
        vel_stamped.pose.position.y = vy_coords[i]
        vel_stamped.pose.position.z = 0.0  # Velocity in z is zero
        vel_stamped.header.stamp = rospy.Time.now()
        vel_stamped.header.frame_id = "map"
        velocity_path.poses.append(vel_stamped)

    # find the center of a bounding box placed around the path
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    print(f"Center of the bounding box: ({center_x}, {center_y})")

    timestamps_msg = Float32MultiArray()
    timestamps_msg.data = list(t_coords)
    
    # Transform and fill transformed Path message
    transformed_path.poses = apply_transformation(x_coords, y_coords, tx, ty, tz, rx, ry, rz)
    
    set_metadata(layer_numbers, resolved_namespace)

    rate = rospy.Rate(1)  # Publish at 1 Hz
    while not rospy.is_shutdown():
        # Update headers' timestamps
        original_path.header.stamp = rospy.Time.now()
        transformed_path.header.stamp = rospy.Time.now()
        velocity_path.header.stamp = rospy.Time.now()
        
        # Publish the original and transformed paths
        original_pub.publish(original_path)
        transformed_pub.publish(transformed_path)
        velocity_pub.publish(velocity_path)
        timestamps_pub.publish(timestamps_msg)
        index_offset_pub.publish(index_offset_msg)
        rate.sleep()

def set_metadata(layer_numbers, path_namespace=""):
    if not layer_numbers:
        rospy.logwarn("No layer metadata available to publish")
        return

    normalized_layers = [max(1, int(round(val))) for val in layer_numbers]
    max_layer = max(normalized_layers)
    points_per_layer = [0 for _ in range(max_layer)]
    for layer in normalized_layers:
        idx = layer - 1
        if 0 <= idx < len(points_per_layer):
            points_per_layer[idx] += 1

    ns = path_namespace.strip('/') if isinstance(path_namespace, str) else ""
    # Always populate the legacy global parameter to avoid breaking consumers.
    rospy.set_param("/points_per_layer", points_per_layer)
    if ns:
        rospy.set_param(f"/{ns}/points_per_layer", points_per_layer)



if __name__ == '__main__':
    try:
        publish_paths()
    except rospy.ROSInterruptException:
        pass
