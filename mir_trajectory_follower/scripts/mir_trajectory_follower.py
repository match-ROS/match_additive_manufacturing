#! /usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Empty, Bool, Int32, Float32, Float32MultiArray
import math
from tf import TransformBroadcaster
from tf import transformations as tr
from std_msgs.msg import Int32, Float32
from copy import deepcopy
import time
import numpy as np

def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class RateLimiter:
    def __init__(self, max_rate_per_s, dt, init_val=0.0):
        self.max_rate = max_rate_per_s
        self.dt = dt
        self.y = init_val
        self.initialized = False
    def reset(self, val=0.0):
        self.y = val; self.initialized = False
    def step(self, x):
        if not self.initialized:
            self.y = x; self.initialized = True
            return self.y
        max_step = self.max_rate * self.dt
        self.y += max(-max_step, min(max_step, x - self.y))
        return self.y

class PurePursuitNode:
    def __init__(self):
        rospy.init_node('pure_pursuit_node')
        
        # Config
        self.path = []
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.1)
        self.lateral_distance_threshold = rospy.get_param("~lateral_distance_threshold", 0.35)
        self.tangent_distance_threshold = rospy.get_param("~tangent_distance_threshold", 0.04)
        self.search_range = rospy.get_param("~search_range", 5) # Number of points to search for lookahead point
        self.Kv = rospy.get_param("~Kv", 1.0)  # Linear speed multiplier
        self.Kw = rospy.get_param("~Kw", 1.0)  # Angular speed multiplier
        self.Ky = rospy.get_param("~Ky", 0.3)  # Lateral error multiplier
        self.K_distance = rospy.get_param("~K_distance", 0.0)  # Distance error multiplier
        self.K_orientation = rospy.get_param("~K_orientation", 0.3)  # Orientation error multiplier
        self.K_idx = rospy.get_param("~K_idx", 0.015)  # Index error multiplier
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_original")
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", "/mur620a/mir_pose_simple")
        self.mir_path_velocity_topic = rospy.get_param("~mir_path_velocity_topic", "/mir_path_velocity")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mur620a/mobile_base_controller/cmd_vel")
        self.trajectory_index_topic = rospy.get_param("~trajectory_index_topic", "/trajectory_index")
        self.layer_progress_topic = rospy.get_param("~layer_progress_topic", "/layer_progress")
        self.control_rate = rospy.get_param("~control_rate", 100)
        self.dT = rospy.get_param("~dT", 0.3)  # Time between trajectory points in seconds
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/mir_target_pose")
        self.actual_pose_topic = rospy.get_param("~actual_pose_topic", "/mir_actual_pose")
        self.points_per_layer = rospy.get_param("/points_per_layer", [0])
        self.override_topic = rospy.get_param("~override_topic", "/velocity_override")
        self.velocity_filter_coeff = rospy.get_param("~velocity_filter_coeff", 0.95)
        self.smooth_window_sec = rospy.get_param("~vel_smooth_window_sec", 2.0)  # Glättungsfenster in Sekunden für Pfadgeschwindigkeit
        self.linear_velocity_limit = rospy.get_param("~linear_velocity_limit", 0.7)  # Maximale lineare Geschwindigkeit
        self.angular_velocity_limit = rospy.get_param("~angular_velocity_limit", 1.2)  # Maximale Winkelgeschwindigkeit
        self.mir_path_timestamps_topic = rospy.get_param("~mir_path_timestamps_topic", "/mir_path_timestamps")
        self.start_condition_topic = rospy.get_param("~start_condition_topic", "/start_condition")
        self.wait_for_start_condition = rospy.get_param("~wait_for_start_condition", True)
        self.initial_path_index = self._parse_initial_path_index(rospy.get_param("~initial_path_index", -1))

        # Fehlergrenzen. Beim Überschreiten wird die Pfadverfolgung abgebrochen
        self.max_distance_error = rospy.get_param("~max_distance_error", 1.0)  # Maximaler Abstandsfehler
        self.max_orientation_error = rospy.get_param("~max_orientation_error", 1.5)  # Maximaler Orientierungsfehler in Radiant
        self.max_index_error = rospy.get_param("~max_index_error", 50) # Maximaler Indexfehler

        self.dt_ctrl = 1.0/float(self.control_rate)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        self.actual_pose_pub = rospy.Publisher(self.actual_pose_topic, PoseStamped, queue_size=1)
        self.layer_progress = rospy.Publisher(self.layer_progress_topic, Float32, queue_size=1)
        
        # Subscriber
        rospy.Subscriber(self.mir_pose_topic, Pose, self.pose_callback)
        rospy.Subscriber(self.trajectory_index_topic, Int32, self.trajectory_index_callback)
        rospy.Subscriber(self.override_topic, Float32, self.override_callback)
        rospy.Subscriber(self.mir_path_velocity_topic, Path, self.velocity_path_callback)
        rospy.Subscriber(self.start_condition_topic, Bool, self.start_condition_callback, queue_size=1)
        
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
        self.mir_path_velocity = None
        self.mir_velocity_old = Twist()
        self.filtered_velocity = Twist()
        self.timestamps = None
        self.dT_list = None
        self.control_enabled = not self.wait_for_start_condition

        # Timestamps passend zum Pfad einlesen
        ts_msg = rospy.wait_for_message(self.mir_path_timestamps_topic, Float32MultiArray)
        self.timestamps = list(ts_msg.data)
        rospy.loginfo("Got timestamp vector with {} entries.".format(len(self.timestamps)))

        # Start
        self.path = rospy.wait_for_message(self.mir_path_topic, Path).poses
        rospy.loginfo("Got path with {} points.".format(len(self.path)))
        self.ur_trajectory_index = self._resolve_initial_path_index()
        rospy.loginfo("Starting from trajectory index: {}".format(self.ur_trajectory_index))
        self.current_mir_path_index = self.ur_trajectory_index
        self._update_active_state()
        self.follow_path()
        

    def reached_target(self, target_position):
        if self.current_pose is None:
            return False
        
        current_position = self.current_pose.position
        distance = math.sqrt((target_position.x - current_position.x) ** 2 + (target_position.y - current_position.y) ** 2)
        # compute lateral and tangential distance
        orientation = self.get_yaw_from_pose(self.current_pose)
        direction = math.atan2(target_position.y - current_position.y, target_position.x - current_position.x)
        self.lateral_distance = distance * math.sin(direction - orientation)
        tangent_distance = distance * math.cos(direction - orientation)

        return abs(tangent_distance) < self.tangent_distance_threshold and abs(self.lateral_distance) < self.lateral_distance_threshold

    def follow_path(self):
        # Berechne die Geschwindigkeiten für jeden Pfadpunkt
        self.calculate_path_velocities()
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if not self.control_enabled:
                rospy.loginfo_throttle(5, f"Waiting for start condition on {self.start_condition_topic}.")
                rate.sleep()
                continue

            if not self.is_active:
                rospy.loginfo_throttle(5, "Waiting for trajectory index.")
                rate.sleep()
                continue

            # Überprüfe, ob Pfad und aktuelle Pose vorhanden sind
            if not self.path or self.current_pose is None:
                rate.sleep()
                continue
            
            # check if the current path index is reaced 
            if self.reached_target(self.path[self.current_mir_path_index].pose.position):
                self.current_mir_path_index += 1
                self.current_sub_step = 0
                # check if trajectory is finished
                if self.current_mir_path_index >= len(self.path):
                    self.is_active = False
                    self.completion_pub.publish(Bool(data=True))
                    rospy.loginfo("Pfadverfolgung abgeschlossen 1.")
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
            #self.publish_actual_and_target_pose()
            
            rate.sleep()


    def calculate_sub_step_progress(self):
        # index-spezifisches dT verwenden (falls vorhanden)
        if self.dT_list is not None and len(self.dT_list) > 0:
            idx = min(self.current_mir_path_index, len(self.dT_list) - 1)
            dT_local = max(1e-3, self.dT_list[idx])
        else:
            dT_local = self.dT

        # compute number of control cycles per path point
        cycles_per_point = max(1, int(round(dT_local * self.control_rate)))
        self.current_sub_step_progress = min(1.0, float(self.current_sub_step) / cycles_per_point)
        self.current_sub_step += 1
        

    def filter_velocity(self, raw_velocity):
        # Implement a simple low-pass filter for velocity
        self.filtered_velocity.linear.x = raw_velocity.linear.x * (1 - self.velocity_filter_coeff) + self.filtered_velocity.linear.x * self.velocity_filter_coeff
        self.filtered_velocity.angular.z = raw_velocity.angular.z * (1 - self.velocity_filter_coeff) + self.filtered_velocity.angular.z * self.velocity_filter_coeff
        return self.filtered_velocity

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
        # Berechne den Winkel zwischen der aktuellen Pose und der Zielpose in quaternion
        # compute relative quaternion (target * conj(current)) and extract yaw
        q_curr = [current_pose.orientation.x,
                  current_pose.orientation.y,
                  current_pose.orientation.z,
                  current_pose.orientation.w]
        q_tgt = [target_pose.orientation.x,
                 target_pose.orientation.y,
                 target_pose.orientation.z,
                 target_pose.orientation.w]

        q_rel = tr.quaternion_multiply(q_tgt, tr.quaternion_conjugate(q_curr))
        _, _, yaw = tr.euler_from_quaternion(q_rel)
        return yaw

    def apply_control(self):
        # Berechne die Steuerbefehle
        index_error = self.ur_trajectory_index - (self.current_mir_path_index-1)

        distance_error = self.calculate_distance(
            self.current_pose.position,
            self.path[self.current_mir_path_index].pose.position
        )

        raw_ang_err = self.calculate_orientation_error(
            self.current_pose,
            self.path[self.current_mir_path_index].pose
        )
        orientation_error = wrap_to_pi(raw_ang_err)

        # broadcast target point
        self.broadcast_target_point(self.path[self.current_mir_path_index].pose.position)
        rospy.loginfo_throttle(3, f"Current index: {self.current_mir_path_index}, Target index: {self.ur_trajectory_index}, Index error: {index_error}, Distance error: {distance_error}, Orientation error: {orientation_error}")
        velocity = Twist()
        feedforward_v  = self.path_velocities_lin[self.current_mir_path_index]
        feedforward_w  = self.path_velocities_ang[self.current_mir_path_index]

        target_v = self.Kv*feedforward_v + self.K_distance*distance_error * self.current_sub_step_progress + self.K_idx*index_error
        target_w = math.sin(self.K_orientation*orientation_error) * self.current_sub_step_progress + self.Kw*feedforward_w + self.Ky * self.lateral_distance * np.sign(target_v)

        #print(f"Kv*feedforward_v: {self.Kv*feedforward_v}, K_distance*distance_error: {self.K_distance*distance_error * self.current_sub_step_progress}, K_idx*index_error: {self.K_idx*index_error}, K_orientation*orientation_error: {self.K_orientation*orientation_error * self.current_sub_step_progress}")
        #print(f"target_v: {target_v}, K_distance*distance_error: {self.K_distance*distance_error * self.current_sub_step_progress}, K_idx*index_error: {self.K_idx*index_error}, lateral_distance: {self.lateral_distance}, target_w: {target_w}")
        # Keine Rückwärtsfahrt (optional)
        target_v = max(0.0, target_v) * self.override

        # Begrenzung der Geschwindigkeiten
        target_v = max(-self.linear_velocity_limit, min(self.linear_velocity_limit, target_v))
        target_w = max(-self.angular_velocity_limit, min(self.angular_velocity_limit, target_w))


        self.filter_velocity(Twist(linear=Twist().linear.__class__(x=target_v), angular=Twist().angular.__class__(z=target_w)))
        velocity.linear.x, velocity.angular.z = self.filtered_velocity.linear.x, self.filtered_velocity.angular.z

        #print(f"Publishing cmd_vel: linear.x={velocity.linear.x}, angular.z={velocity.angular.z}")

        # Überprüfe Fehlergrenzen
        result = self.check_error_thresholds(distance_error, orientation_error, index_error)
        if not result:
            rospy.logwarn_throttle(1,"Error thresholds exceeded. Stopping path following.")
            velocity.linear.x = 0.0
            velocity.angular.z = 0.0
            self.is_active = False

        #print(f"Publishing cmd_vel: linear.x={velocity.linear.x}, angular.z={velocity.angular.z}")

        self.cmd_vel_pub.publish(velocity)


    def check_error_thresholds(self, distance_error, orientation_error, index_error):
        if abs(distance_error) > self.max_distance_error:
            rospy.logwarn_throttle(5,f"Distance error {distance_error} exceeds maximum threshold {self.max_distance_error}. Stopping path following.")
            return False
        if abs(orientation_error) > self.max_orientation_error:
            rospy.logwarn_throttle(5,f"Orientation error {orientation_error} exceeds maximum threshold {self.max_orientation_error}. Stopping path following.")
            return False
        if abs(index_error) > self.max_index_error:
            rospy.logwarn_throttle(5,f"Index error {index_error} exceeds maximum threshold {self.max_index_error}. Stopping path following.")
            return False
        return True

    def broadcast_target_point(self, point):
        self.broadcaster.sendTransform(
            (point.x, point.y, point.z),
            (0, 0, 0, 1),
            rospy.Time.now(),
            "target_point",
            "map"
        )

    def calculate_path_velocities(self):
        # Falls keine Timestamps vorhanden sind, auf konstantes dT zurückfallen
        if self.timestamps is None or len(self.timestamps) < 2:
            rospy.logwarn("No or insufficient timestamps received. Using constant dT = {}".format(self.dT))
            self.dT_list = [self.dT] * len(self.path)
        else:
            # dT-Liste aus Timestamps berechnen
            self.dT_list = [self.dT]  # erstes Element als Fallback
            for i in range(len(self.path) - 1):
                if i + 1 < len(self.timestamps):
                    dt = self.timestamps[i + 1] - self.timestamps[i]
                else:
                    dt = self.dT_list[-1]
                # numerisch absichern
                dt = max(1e-3, float(dt))
                self.dT_list.append(dt)

        self.path_velocities_lin = [0.0]
        self.path_velocities_ang = [0.0]

        for i in range(len(self.path) - 1):
            p1 = self.path[i].pose.position
            p2 = self.path[i + 1].pose.position
            distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

            angular_diff_unwrapped = self.get_yaw_from_pose(self.path[i + 1].pose) - self.get_yaw_from_pose(self.path[i].pose)
            angular_diff = math.atan2(math.sin(angular_diff_unwrapped), math.cos(angular_diff_unwrapped))

            # dT für Segment i -> i+1 (wir hängen v bei Punkt i+1 an)
            dt_seg = self.dT_list[i + 1]

            self.path_velocities_lin.append(distance / dt_seg)
            self.path_velocities_ang.append(angular_diff / dt_seg)

        lin = np.array(self.path_velocities_lin, dtype=float)
        ang = np.array(self.path_velocities_ang, dtype=float)

        # mittleres dT für Fenstergröße
        dt_mean = float(sum(self.dT_list)) / max(1, len(self.dT_list))
        win = max(1, int(round(self.smooth_window_sec / dt_mean)))
        if win % 2 == 0:
            win += 1

        lin_s = self.zero_phase_moving_average(lin, win)
        ang_s = self.zero_phase_moving_average(ang, win)

        self.path_velocities_lin = lin_s.tolist()
        self.path_velocities_ang = ang_s.tolist()



    def zero_phase_moving_average(self,x, window_size):
        if window_size <= 1:
            return np.asarray(x, dtype=float)
        kernel = np.ones(window_size, dtype=float) / window_size
        y = np.convolve(x, kernel, mode='same')          # forward
        y = np.convolve(y[::-1], kernel, mode='same')[::-1]  # backward
        return y

    def pose_callback(self, msg):
        self.current_pose = msg

        # broadcast current pose (unverändert)
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
        self._update_active_state()

    def start_condition_callback(self, msg: Bool):
        new_state = bool(msg.data) or not self.wait_for_start_condition
        if new_state == self.control_enabled:
            return

        self.control_enabled = new_state
        if not self.control_enabled:
            rospy.loginfo("Start condition reset – pausing MIR trajectory follower.")
            self.is_active = False
            self.cmd_vel_pub.publish(Twist())
        else:
            rospy.loginfo("Start condition fulfilled – MIR trajectory follower ready.")
        self._update_active_state()

    def _update_active_state(self):
        should_be_active = self.control_enabled and self.ur_trajectory_index > 0
        if should_be_active and not self.is_active:
            rospy.loginfo("Trajectory index available and start condition met – starting MIR path following.")
        self.is_active = should_be_active

    @staticmethod
    def _parse_initial_path_index(raw_value):
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, f"Invalid initial_path_index value '{raw_value}', ignoring.")
            return None
        return idx if idx >= 0 else None

    def _resolve_initial_path_index(self):
        if self.initial_path_index is not None:
            clamped = min(max(self.initial_path_index, 0), max(len(self.path) - 1, 0))
            if clamped != self.initial_path_index:
                rospy.logwarn("Initial path index %d clamped to %d based on path length.", self.initial_path_index, clamped)
            else:
                rospy.loginfo("Using initial path index %d from parameter.", clamped)
            return clamped

        rospy.loginfo("Waiting for first trajectory index on %s", self.trajectory_index_topic)
        msg = rospy.wait_for_message(self.trajectory_index_topic, Int32)
        return max(0, msg.data)

    def calculate_distance(self, current_pose, target_pose):
        # Compute if the mir is in front or behind the target point
        mir_yaw = self.get_yaw_from_pose(self.current_pose)
        target_direction = math.atan2(target_pose.y - current_pose.y, target_pose.x - current_pose.x)
        angle_diff = target_direction - mir_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # If the angle difference is between -pi/2 and pi/2, mir is facing towards the target (in front)
        # Otherwise, it's behind
        if -math.pi/2 <= angle_diff <= math.pi/2:
            direction = -1
        else:
            direction = 1
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

    def velocity_path_callback(self, msg):
        self.mir_path_velocity = msg


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
