#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose
import os
import rospkg
import subprocess

class LayerSpawner:
    def __init__(self):
        # Initialize the node
        rospy.init_node('layer_spawner')
        self.current_layer_topic = rospy.get_param('~current_layer_topic', '/current_layer')

        # Get the base pose from parameter server
        self.base_pose = rospy.get_param('~object_center_pose', [43.289831, 39.752500, 0.0])
        self.base_pose = rospy.get_param('~object_center_pose', [0.0, 0.0, 0.0])
        self.current_layer = -1

        # Path to the STL file
        rospack = rospkg.RosPack()
        self.stl_file = rospy.get_param(
            '~stl_file',
            os.path.join(rospack.get_path('component'), 'models', 'first_layer', 'first_layer.sdf')
        )

        # Topic subscription
        rospy.Subscriber('/current_layer', Int32, self.layer_callback)

        rospy.loginfo("LayerSpawner node initialized.")

    def layer_callback(self, msg):
        # Check if the layer has increased
        if msg.data > self.current_layer:
            rospy.loginfo(f"Spawning layer {msg.data}")
            self.spawn_layer(msg.data)
            self.current_layer = msg.data

    def spawn_layer(self, layer_number):
        # Calculate the pose for the new layer
        layer_pose = Pose()
        layer_pose.position.x = self.base_pose[0]
        layer_pose.position.y = self.base_pose[1]
        layer_pose.position.z = self.base_pose[2] + layer_number * 0.05  # Assuming each layer is 0.1m thick
        layer_pose.orientation.w = 1.0

        # Spawn the model in Gazebo
        spawn_cmd = [
            'rosrun', 'gazebo_ros', 'spawn_model',
            '-file', self.stl_file,
            '-sdf',  # Change to '-urdf' if using a URDF file
            '-model', f'layer_{layer_number}',
            '-x', str(layer_pose.position.x),
            '-y', str(layer_pose.position.y),
            '-z', str(layer_pose.position.z)
        ]

        try:
            subprocess.check_call(spawn_cmd)
            rospy.loginfo(f"Layer {layer_number} spawned successfully.")
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Failed to spawn layer {layer_number}: {e}")

if __name__ == '__main__':
    try:
        spawner = LayerSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
