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
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.1)
        self.lateral_distance_threshold = rospy.get_param("~lateral_distance_threshold", 0.2)
        self.tangent_distance_threshold = rospy.get_param("~tangent_distance_threshold", 0.02)
        self.search_range = rospy.get_param("~search_range", 5) # Number of points to search for lookahead point
        self.Kv = rospy.get_param("~Kv", 1.0)  # Linear speed multiplier
        self.K_distance = rospy.get_param("~K_distance", 0.0)  # Distance error multiplier
        self.K_orientation = rospy.get_param("~K_orientation", 0.5)  # Orientation error multiplier
        self.K_idx = rospy.get_param("~K_idx", 0.01)  # Index error multiplier
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_original")
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", "/mur620a/mir_pose_simple")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mur620a/mobile_base_controller/cmd_vel")
        self.trajectory_index_topic = rospy.get_param("~trajectory_index_topic", "/trajectory_index")
        self.layer_progress_topic = rospy.get_param("~layer_progress_topic", "/layer_progress")
        self.control_rate = rospy.get_param("~control_rate", 100)
        self.dT = rospy.get_param("~dT", 0.5)
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/mir_target_pose")
        self.actual_pose_topic = rospy.get_param("~actual_pose_topic", "/mir_actual_pose")
        self.points_per_layer = rospy.get_param("/points_per_layer", [0])
        self.override_topic = rospy.get_param("~override_topic", "/velocity_override")
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        self.actual_pose_pub = rospy.Publisher(self.actual_pose_topic, PoseStamped, queue_size=1)
        self.layer_progress = rospy.Publisher(self.layer_progress_topic, Float32, queue_size=1)
        
        # Subscriber
        rospy.Subscriber(self.mir_pose_topic, Pose, self.pose_callback)
        rospy.Subscriber(self.trajectory_index_topic, Int32, self.trajectory_index_callback)
        rospy.Subscriber(self.override_topic, Float32, self.override_callback)
        
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
        self.current_layer = 0
        self.override = 1.0  # Default override value
        self.current_sub_step = 0

        # Start
        self.path = rospy.wait_for_message(self.mir_path_topic, Path).poses
        rospy.loginfo("Got path with {} points.".format(len(self.path)))
        self.ur_trajectory_index = rospy.wait_for_message(self.trajectory_index_topic, Int32).data
        rospy.loginfo("Starting from trajectory index: {}".format(self.ur_trajectory_index))
        self.current_mir_path_index = self.ur_trajectory_index
        self.follow_path()
        

    def reached_target(self, target_position):
        if self.current_pose is None:
            return False
        
        current_position = self.current_pose.position
        distance = math.sqrt((target_position.x - current_position.x) ** 2 + (target_position.y - current_position.y) ** 2)
        # compute lateral and tangential distance
        orientation = self.get_yaw_from_pose(self.current_pose)
        direction = math.atan2(target_position.y - current_position.y, target_position.x - current_position.x)
        lateral_distance = distance * math.sin(direction - orientation)
        tangent_distance = distance * math.cos(direction - orientation)

        return abs(tangent_distance) < self.tangent_distance_threshold and abs(lateral_distance) < self.lateral_distance_threshold

    def follow_path(self):

        # Berechne die Geschwindigkeiten für jeden Pfadpunkt
        self.calculate_path_velocities()
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
                self.current_sub_step = 0
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
                
            # Berechne Fortschritt im Erreichen des aktuellen Pfadpunktes
            self.calculate_sub_step_progress()

            self.apply_control()

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


    def calculate_sub_step_progress(self):
        # compute number of control cycles per path point
        cycles_per_point = int(self.dT * self.control_rate)
        self.current_sub_step_progress = min(1.0, self.current_sub_step / cycles_per_point)  # Ensure progress does not exceed 1.0
        self.current_sub_step += 1
        

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

    def calculate_orientation_error(self, current_pose, target_pose):
        # Berechne den Winkel zwischen der aktuellen Pose und der Zielpose
        current_yaw = self.get_yaw_from_pose(current_pose)
        target_yaw = self.get_yaw_from_pose(target_pose)

        # Normalisiere den Winkelunterschied
        angle_diff = target_yaw - current_yaw
        #angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalisiere den Winkel
        return angle_diff  # Rückgabe des normalisierten Winkelunterschieds

    def apply_control(self):
        # Berechne die Steuerbefehle
        index_error = self.ur_trajectory_index - (self.current_mir_path_index-1) # -1 because of the current_mir_path_index is the next point
 
        distance_error = self.calculate_distance(self.current_pose.position, self.path[self.current_mir_path_index].pose.position)
        orientation_error = self.calculate_orientation_error(self.current_pose, self.path[self.current_mir_path_index].pose)

        # broadcast target point
        self.broadcast_target_point(self.path[self.current_mir_path_index].pose.position)
        rospy.loginfo_throttle(1, f"Current index: {self.current_mir_path_index}, Target index: {self.ur_trajectory_index}, Index error: {index_error}, Distance error: {distance_error}, Orientation error: {orientation_error}")
        velocity = Twist()
        target_vel = self.Kv * self.path_velocities_lin[self.current_mir_path_index] + self.K_distance * distance_error + self.K_idx * index_error

        velocity.linear.x = max(0.0, target_vel ) * self.override  # min 0.0 to avoid negative speeds

        velocity.angular.z =  self.K_orientation * orientation_error + self.Kv * (self.path_velocities_ang[self.current_mir_path_index])

        print(f"Lin vel: {velocity.linear.x}, Ang vel: {velocity.angular.z}")
        self.cmd_vel_pub.publish(velocity)



    def broadcast_target_point(self, point):
        self.broadcaster.sendTransform(
            (point.x, point.y, point.z),
            (0, 0, 0, 1),
            rospy.Time.now(),
            "target_point",
            "map"
        )

    def calculate_path_velocities(self):
        self.path_velocities_lin = [0.0]
        self.path_velocities_ang = [0.0]

        for i in range(len(self.path) - 1):
            p1 = self.path[i].pose.position
            p2 = self.path[i + 1].pose.position
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
            angular_diff_unwrapped = self.get_yaw_from_pose(self.path[i + 1].pose) - self.get_yaw_from_pose(self.path[i].pose)
            angular_diff = math.atan2(math.sin(angular_diff_unwrapped), math.cos(angular_diff_unwrapped))  # Normalize angle
            
            self.path_velocities_lin.append(distance * (1.0/self.dT))  # Linear velocity
            self.path_velocities_ang.append(angular_diff * (1.0/self.dT))


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

    def calculate_distance(self, current_pose, target_pose):
        # Compute if the mir is in front or behind the target point
        mir_yaw = self.get_yaw_from_pose(self.current_pose)
        target_direction = math.atan2(target_pose.y - current_pose.y, target_pose.x - current_pose.x)
        angle_diff = target_direction - mir_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize angle

        # If the angle difference is between -pi/2 and pi/2, mir is facing towards the target (in front)
        # Otherwise, it's behind
        if -math.pi/2 <= angle_diff <= math.pi/2:
            direction = 1
        else:
            direction = -1
        # Calculate the distance
        distance = math.sqrt((target_pose.x - current_pose.x) ** 2 + (target_pose.y - current_pose.y) ** 2)

        return distance * direction 
    

    def get_yaw_from_pose(self, pose):
        orientation = pose.orientation
        _, _, yaw = tr.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw
    
    def override_callback(self, msg):
        # Override the current velocity with the received value
        self.override = msg.data

    def publish_actual_and_target_pose(self):
        actual_pose = PoseStamped()
        actual_pose.pose = deepcopy(self.current_pose)
        actual_pose.header.frame_id = "map"
        actual_pose.header.stamp = rospy.Time.now()
        self.actual_pose_pub.publish(actual_pose)

        target_pose = PoseStamped()
        target_pose.pose.position = self.path[self.ur_trajectory_index].pose.position
        target_pose.pose.orientation = self.path[self.ur_trajectory_index].pose.orientation
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
