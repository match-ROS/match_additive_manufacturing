#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Pose, Quaternion
from std_msgs.msg import Empty, Bool
from tf.transformations import euler_from_quaternion
import tf.transformations as tr
import math
from tf import TransformBroadcaster

class PathFollowerNode:
    def __init__(self):
        rospy.init_node('path_follower_node')
        
        # Config
        self.path = []
        self.velocities = []
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.15)
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_original")
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", "/mur620a/mir_pose_simple")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mur620a/mobile_base_controller/cmd_vel")
        self.control_rate = rospy.get_param("~control_rate", 100)
        
        # Subscriber
        rospy.Subscriber(self.mir_path_topic, Path, self.path_callback)
        rospy.Subscriber(self.mir_pose_topic, Pose, self.pose_callback)
        
        # Publisher 
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # Start und Status
        rospy.Subscriber("/start_follow_path", Empty, self.start_callback)
        self.completion_pub = rospy.Publisher("/path_following_complete", Bool, queue_size=1)
        
        # Init
        self.current_pose = None
        self.is_active = False
        self.controller_output = Twist()
        self.broadcaster = TransformBroadcaster()

    def path_callback(self, msg):
        self.path = msg.poses
        self.calculate_velocities()

    def pose_callback(self, msg):
        self.current_pose = msg

    def calculate_velocities(self):
        self.velocities = []
        for i in range(len(self.path) - 1):
            p1 = self.path[i].pose.position
            p2 = self.path[i + 1].pose.position
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
            speed = distance  # Goal: reach the next point in 1 second
            self.velocities.append(speed)
        #print("Velocities: ", self.velocities)

    def start_callback(self, msg):
        if not self.path:
            rospy.logwarn("Got no path. Ignoring start command")
            return
        self.is_active = True
        rospy.loginfo("Start commnand received. Starting ...")
        self.follow_path()

    def follow_path(self):
        for idx, target_pose in enumerate(self.path):
            target_position = target_pose.pose.position
            target_orientation = target_pose.pose.orientation
            speed = self.velocities[idx]
            
            #broadcast target position
            self.broadcaster.sendTransform((target_position.x, target_position.y, target_position.z), (target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w), rospy.Time.now(), "target_position", "map")
            rate = rospy.Rate(self.control_rate)
            while not rospy.is_shutdown() and self.is_active and not self.reached_target(target_position):
                self.align_robot(target_position)
                self.move_toward_target(speed)
                rate.sleep() 
            
            if not self.is_active:
                break
        
        self.is_active = False
        self.completion_pub.publish(Bool(data=True))
        rospy.loginfo("Pfadverfolgung abgeschlossen.")

    def reached_target(self, target_position):
        if self.current_pose is None:
            return False
        
        current_position = self.current_pose.position
        distance = math.sqrt((target_position.x - current_position.x) ** 2 + (target_position.y - current_position.y) ** 2)
        #print("Distance: ", distance)
        return distance < self.distance_threshold

    def align_robot(self, target_position):
        if self.current_pose is None:
            return
        
        # current position and orientation of the robot
        current_position = self.current_pose.position
        current_orientation = self.current_pose.orientation

        # broadcast current position
        self.broadcaster.sendTransform((current_position.x, current_position.y, current_position.z), (current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w), rospy.Time.now(), "current_position", "map")

        # Richtung zum Zielpunkt als Ziel-Orientierung berechnen
        angle_to_target = math.atan2(target_position.y - current_position.y, target_position.x - current_position.x)
        target_orientation_q = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(angle_to_target / 2),
            w=math.cos(angle_to_target / 2)
        )

        # Winkel-Differenz zwischen aktueller und Ziel-Orientierung berechnen
        current_orientation_q = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
        target_orientation_q = [target_orientation_q.x, target_orientation_q.y, target_orientation_q.z, target_orientation_q.w]
        rotation_diff_q = tr.quaternion_multiply(target_orientation_q, tr.quaternion_conjugate(current_orientation_q))

        # Extrahiere die Z-Komponente der Winkelgeschwindigkeit
        angle_diff = 2 * math.atan2(rotation_diff_q[2], rotation_diff_q[3])

        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
    
        self.controller_output.angular.z = angle_diff*1.0

    def move_toward_target(self, speed):
        self.controller_output.linear.x = speed*1.0
        self.cmd_vel_pub.publish(self.controller_output)

if __name__ == '__main__':
    PathFollowerNode()
    rospy.spin()
