#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Empty, Bool
import math
from tf import TransformBroadcaster
from tf import transformations as tr
from std_msgs.msg import Int32
from copy import deepcopy

class PurePursuitNode:
    def __init__(self):
        rospy.init_node('pure_pursuit_node')
        
        # Config
        self.path = []
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.25)
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.25)
        self.search_range = rospy.get_param("~search_range", 20) # Number of points to search for lookahead point
        self.Kv = rospy.get_param("~Kv", 1.0)  # Linear speed multiplier
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_original")
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", "/mur620a/mir_pose_simple")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mur620a/mobile_base_controller/cmd_vel")
        self.trajectory_index_topic = rospy.get_param("~trajectory_index_topic", "/trajectory_index")
        self.control_rate = rospy.get_param("~control_rate", 100)
        self.dT = rospy.get_param("~dT", 0.2)
        
        # Subscriber
        rospy.Subscriber(self.mir_path_topic, Path, self.path_callback)
        rospy.Subscriber(self.mir_pose_topic, Pose, self.pose_callback)
        rospy.Subscriber(self.trajectory_index_topic, Int32, self.trajectory_index_callback)
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # Start und Status
        rospy.Subscriber("/start_follow_path", Empty, self.start_callback)
        self.completion_pub = rospy.Publisher("/path_following_complete", Bool, queue_size=1)
        
        # Init
        self.current_pose = None
        self.is_active = False
        self.broadcaster = TransformBroadcaster()
        self.last_index = 0
        self.ur_trajectory_index = 0
        self.current_lookahead_point = None
        self.current_mir_path_index = 0
        self.time_stamp_old = rospy.Time.now()

    def start_callback(self, msg):
        if not self.path:
            rospy.logwarn("Got no path. Ignoring start command")
            return
        self.is_active = True
        rospy.loginfo_once("Start command received. Starting ...")
        self.follow_path()

    def follow_path(self):

        # Berechne die Geschwindigkeiten für jeden Pfadpunkt
        self.calculate_velocities()

        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown() and self.is_active:
            # Überprüfe, ob Pfad und aktuelle Pose vorhanden sind
            if not self.path or self.current_pose is None:
                continue
            
            # Increment index if time has passed - the index is not incremented by reaching the lookahead point only time
            if rospy.Time.now() - self.time_stamp_old > rospy.Duration(self.dT):
                self.time_stamp_old = rospy.Time.now()
                self.last_index += 1

            # Berechne den Lookahead-Punkt
            lookahead_point = self.find_lookahead_point()
            if lookahead_point is None:
                rospy.loginfo_throttle(5,"No valid lookahead point found. Stopping.")
                continue

            # Berechne die Krümmung und Steuerung
            curvature = self.calculate_curvature(lookahead_point)
            self.apply_control(curvature)

            # Beende die Pfadverfolgung, wenn der Zielpunkt erreicht ist
            # if self.reached_target(self.path[-1].pose.position):
            #     self.is_active = False
            #     self.completion_pub.publish(Bool(data=True))
            #     rospy.loginfo("Pfadverfolgung abgeschlossen.")
            #     break
            


            rate.sleep()

    def find_lookahead_point(self):
        # Suche im Pfadausschnitt
        search_range = self.path[self.last_index:self.last_index + self.search_range]
        print("last_index", self.last_index)

        for idx, pose in enumerate(search_range):
            position = pose.pose.position
            distance = self.calculate_distance(self.current_pose.position, position)
            
            # Überprüfe, ob der Lookahead-Punkt weit genug entfernt ist
            if distance >= self.lookahead_distance:
                # Sende den Lookahead-Punkt an tf
                self.broadcaster.sendTransform(
                    (position.x, position.y, position.z),
                    (0, 0, 0, 1),
                    rospy.Time.now(),
                    "lookahead_point",
                    "map"
                )
                rospy.loginfo(f"Lookahead point updated: Index {self.last_index}")
                return position
        return None

    def calculate_curvature(self, lookahead_point):
        # Aktuelle Position und Orientierung
        robot_position = self.current_pose.position
        robot_yaw = self.get_yaw_from_pose(self.current_pose)

        # Relativer Lookahead-Punkt im Roboterrahmen
        dx = lookahead_point.x - robot_position.x
        dy = lookahead_point.y - robot_position.y
        transformed_x = math.cos(-robot_yaw) * dx - math.sin(-robot_yaw) * dy
        transformed_y = math.sin(-robot_yaw) * dx + math.cos(-robot_yaw) * dy

        # Berechne die Krümmung
        if transformed_y == 0:
            return 0.0
        return 2 * transformed_y / (self.lookahead_distance ** 2)

    def apply_control(self, curvature):
        # Berechne die Steuerbefehle
        index_error = self.ur_trajectory_index - (self.last_index)    
        distance_error = self.calculate_distance(self.current_pose.position, self.path[self.last_index].pose.position)

        velocity = Twist()
        velocity.linear.x = self.Kv * self.velocities[self.last_index] * (1/self.dT) + 0.1 * distance_error# * (1.0 + 0.1*index_error) 
        velocity.angular.z = velocity.linear.x * curvature
        self.cmd_vel_pub.publish(velocity)

    def calculate_velocities(self):
        self.velocities = []
        for i in range(len(self.path) - 1):
            p1 = self.path[i].pose.position
            p2 = self.path[i + 1].pose.position
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
            speed = distance  # Goal: reach the next point in 1 second
            self.velocities.append(speed)

    def path_callback(self, msg):
        self.path = msg.poses

    def pose_callback(self, msg):
        self.current_pose = msg
        # broadcast current pose
        self.broadcaster.sendTransform(
            (msg.position.x, msg.position.y, msg.position.z),
            (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
            rospy.Time.now(),
            "current_pose",
            "map"
        )


    def trajectory_index_callback(self, msg):
        self.ur_trajectory_index = msg.data

    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)

    def get_yaw_from_pose(self, pose):
        orientation = pose.orientation
        _, _, yaw = tr.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw

if __name__ == '__main__':
    PurePursuitNode()
    rospy.spin()
