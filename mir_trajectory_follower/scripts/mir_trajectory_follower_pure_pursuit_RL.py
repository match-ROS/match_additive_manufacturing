#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Empty, Bool
import math
from tf import TransformBroadcaster
from tf import transformations as tr
from std_msgs.msg import Int32, Float32
from copy import deepcopy

class PurePursuitNode:
    def __init__(self):
        rospy.init_node('pure_pursuit_node')
        
        # Config
        self.path = []
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.25)
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.15)
        self.search_range = rospy.get_param("~search_range", 20) # Number of points to search for lookahead point
        self.Kv = rospy.get_param("~Kv", 0.2)  # Linear speed multiplier
        self.K_distance = rospy.get_param("~K_distance", 0.3)  # Distance error multiplier
        self.K_idx = rospy.get_param("~K_idx", 0.01)  # Index error multiplier
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_original")
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", "/mur620a/mir_pose_simple")
        self.RL_cmd_vel_offset_topic = rospy.get_param("~RL_cmd_vel_offset_topic", "/RL_cmd_vel_offset")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mur620a/mobile_base_controller/cmd_vel")
        self.trajectory_index_topic = rospy.get_param("~trajectory_index_topic", "/trajectory_index")
        self.layer_progress_topic = rospy.get_param("~layer_progress_topic", "/layer_progress")
        self.control_rate = rospy.get_param("~control_rate", 100)
        self.dT = rospy.get_param("~dT", 0.3)
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/mir_target_pose")
        self.actual_pose_topic = rospy.get_param("~actual_pose_topic", "/mir_actual_pose")
        self.points_per_layer = rospy.get_param("/points_per_layer", [0])

        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        self.actual_pose_pub = rospy.Publisher(self.actual_pose_topic, PoseStamped, queue_size=1)
        self.layer_progress = rospy.Publisher(self.layer_progress_topic, Float32, queue_size=1)
        
        # Subscriber
        rospy.Subscriber(self.mir_pose_topic, Pose, self.pose_callback)
        rospy.Subscriber(self.trajectory_index_topic, Int32, self.trajectory_index_callback)
        rospy.Subscriber(self.RL_cmd_vel_offset_topic, Twist, self.RL_cmd_vel_offset_callback)
        
        # Start und Status
        self.completion_pub = rospy.Publisher("/path_following_complete", Bool, queue_size=1)
        
        # Init
        self.current_pose = None
        self.is_active = False
        self.broadcaster = TransformBroadcaster()
        self.ur_trajectory_index = 0
        self.current_lookahead_point = None
        self.current_mir_path_index = 0
        self.current_target_index = 0
        self.time_stamp_old = rospy.Time.now()
        self.RL_cmd_vel_offset = Twist()
        self.current_layer = 0

        # Start
        self.path = rospy.wait_for_message(self.mir_path_topic, Path).poses
        self.ur_trajectory_index = rospy.wait_for_message(self.trajectory_index_topic, Int32).data
        self.current_mir_path_index = self.ur_trajectory_index
        self.follow_path()
        

    def reached_target(self, target_position):
        if self.current_pose is None:
            return False
        
        current_position = self.current_pose.position
        distance = math.sqrt((target_position.x - current_position.x) ** 2 + (target_position.y - current_position.y) ** 2)
        #print("Distance: ", distance)
        return distance < self.distance_threshold

    def follow_path(self):

        # Berechne die Geschwindigkeiten für jeden Pfadpunkt
        self.calculate_velocities()
        while not rospy.is_shutdown() and self.is_active == False:
            rospy.loginfo_throttle(5, "Waiting for trajectory index.")
            rospy.sleep(0.01)

        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            # Überprüfe, ob Pfad und aktuelle Pose vorhanden sind
            if not self.path or self.current_pose is None:
                continue
            
            # check if the current path index is reaced 
            if self.reached_target(self.path[self.current_mir_path_index].pose.position):
                self.current_mir_path_index += 1
                # check if trajectory is finished
                if self.current_mir_path_index >= len(self.path):
                    self.is_active = False
                    self.completion_pub.publish(Bool(data=True))
                    rospy.loginfo("Pfadverfolgung abgeschlossen.")
                    break

            # Berechne den Lookahead-Punkt
            lookahead_point = self.find_lookahead_point()
            if lookahead_point is None:
                rospy.loginfo_throttle(5,"No valid lookahead point found. Stopping.")
                continue

            # Berechne die Krümmung und Steuerung
            curvature = self.calculate_curvature(lookahead_point)
            self.apply_control(curvature)

            # update layer progress
            self.publish_layer_progress()
            self.publish_actual_and_target_pose()

            # Beende die Pfadverfolgung, wenn der Zielpunkt erreicht ist
            # if self.reached_target(self.path[-1].pose.position):
            #     self.is_active = False
            #     self.completion_pub.publish(Bool(data=True))
            #     rospy.loginfo("Pfadverfolgung abgeschlossen.")
            #     break
            


            rate.sleep()

    def find_lookahead_point(self):
        # Suche im Pfadausschnitt
        search_range = self.path[self.current_mir_path_index:self.current_mir_path_index + self.search_range]
        #print("last_index", self.ur_trajectory_index)

        for idx, pose in enumerate(search_range):
            position = pose.pose.position
            distance = self.calculate_distance(self.current_pose.position, position)
            
            # Überprüfe, ob der Lookahead-Punkt weit genug entfernt ist
            if (distance >= self.lookahead_distance and idx + self.current_mir_path_index >= self.current_target_index) or idx == len(search_range) - 1:
                # Sende den Lookahead-Punkt an tf
                self.broadcaster.sendTransform(
                    (position.x, position.y, position.z),
                    (0, 0, 0, 1),
                    rospy.Time.now(),
                    "lookahead_point",
                    "map"
                )
                self.current_target_index = idx + self.current_mir_path_index
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
        index_error = self.ur_trajectory_index - (self.current_mir_path_index-1) # -1 because of the current_mir_path_index is the next point
        #print("index_error", index_error)   
        distance_error = self.calculate_distance(self.current_pose.position, self.path[self.current_mir_path_index].pose.position)



        # broadcast target point
        self.broadcast_target_point(self.path[self.current_mir_path_index].pose.position)

        velocity = Twist()
        target_vel = self.Kv * self.velocities[self.current_mir_path_index] * (1/self.dT) + self.K_distance * distance_error + self.K_idx * index_error
        velocity.linear.x = max(0.0, target_vel )  # min 0.0 to avoid negative speeds
        #velocity.linear.x *= max(0.0, (1.0 + 0.1*index_error))
         
        velocity.angular.z = velocity.linear.x * curvature

        # Add RL offset
        velocity.linear.x += self.RL_cmd_vel_offset.linear.x
        velocity.angular.z += self.RL_cmd_vel_offset.angular.z

        self.cmd_vel_pub.publish(velocity)



    def broadcast_target_point(self, point):
        self.broadcaster.sendTransform(
            (point.x, point.y, point.z),
            (0, 0, 0, 1),
            rospy.Time.now(),
            "target_point",
            "map"
        )

    def calculate_velocities(self):
        self.velocities = []
        for i in range(len(self.path) - 1):
            p1 = self.path[i].pose.position
            p2 = self.path[i + 1].pose.position
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
            speed = distance  # Goal: reach the next point in 1 second
            self.velocities.append(speed)

    def pose_callback(self, msg):
        self.current_pose = msg
        # broadcast current pose
        now = rospy.Time.now()
        if now == self.time_stamp_old:
            return
        self.time_stamp_old = now
        self.broadcaster.sendTransform(
            (msg.position.x, msg.position.y, msg.position.z),
            (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
            rospy.Time.now(),
            "current_pose",
            "map"
        )

    def trajectory_index_callback(self, msg):
        self.ur_trajectory_index = msg.data
        if self.ur_trajectory_index > 0:
            self.is_active = True

    def calculate_distance(self, pos1, pos2):
        # compute direction
        orientation = math.atan2(pos2.y - pos1.y, pos2.x - pos1.x)
        if orientation < math.pi / 2 and orientation > -math.pi / 2:
            direction = -1  # mir is in front of the target point so we need to go backwards
        else:
            direction = 1
        # compute distance
        print("direction", direction)
        distance = math.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)
            
        return distance * direction
    

    def get_yaw_from_pose(self, pose):
        orientation = pose.orientation
        _, _, yaw = tr.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw
    
    def RL_cmd_vel_offset_callback(self, msg):
        self.RL_cmd_vel_offset = msg

    def publish_actual_and_target_pose(self):
        actual_pose = PoseStamped()
        actual_pose.pose = deepcopy(self.current_pose)
        actual_pose.header.frame_id = "map"
        actual_pose.header.stamp = rospy.Time.now()
        self.actual_pose_pub.publish(actual_pose)

        target_pose = PoseStamped()
        target_pose.pose.position = self.path[self.current_mir_path_index].pose.position
        target_pose.header.frame_id = "map"
        target_pose.header.stamp = rospy.Time.now()
        self.target_pose_pub.publish(target_pose)

    def publish_layer_progress(self):
        progress = Float32()
        points_in_current_layer = self.points_per_layer[self.current_layer]
        points_in_previous_layers = sum(self.points_per_layer[:self.current_layer])
          
        if points_in_current_layer == 0:
            progress.data = 0
        else: 
            progress.data = (self.current_mir_path_index - points_in_previous_layers) / points_in_current_layer
        self.layer_progress.publish(progress)

if __name__ == '__main__':
    PurePursuitNode()
    rospy.spin()
